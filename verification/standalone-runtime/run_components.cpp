#ifdef TRANSCRIBRO_ORT_REFERENCE
#include "onnxruntime_cxx_api.h"
#else
#include "medium.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <vector>

namespace {
double now(clockid_t clock = CLOCK_MONOTONIC) {
  timespec value{};
  clock_gettime(clock, &value);
  return double(value.tv_sec) + double(value.tv_nsec) / 1e9;
}
struct Input {
  std::vector<int64_t> shape;
  std::vector<uint64_t> data;
  uint32_t type;
  explicit Input(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("missing input");
    const auto bytes = f.tellg();
    if (bytes < 16 || bytes > 512 * 1024 * 1024)
      throw std::runtime_error("input size");
    f.seekg(0);
    char magic[8];
    uint32_t rank;
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(&type), 4);
    f.read(reinterpret_cast<char *>(&rank), 4);
    if (std::memcmp(magic, "TRINPUT1", 8) || rank > 8 ||
        (type != 1 && type != 7))
      throw std::runtime_error("input header");
    shape.resize(rank);
    f.read(reinterpret_cast<char *>(shape.data()), rank * 8);
    size_t elements = 1;
    for (auto dim : shape) {
      if (dim < 0 || dim > 10000000 ||
          (dim && elements > 100000000 / size_t(dim)))
        throw std::runtime_error("input shape");
      elements *= size_t(dim);
    }
    const size_t payload = elements * (type == 1 ? 4 : 8);
    if (payload + 16 + rank * 8 != size_t(bytes))
      throw std::runtime_error("input extent");
    data.resize((payload + 7) / 8);
    f.read(reinterpret_cast<char *>(data.data()), payload);
    if (!f)
      throw std::runtime_error("input read");
  }
  const float *floats() const {
    if (type != 1)
      throw std::runtime_error("float input required");
    return reinterpret_cast<const float *>(data.data());
  }
  const int64_t *integers() const {
    if (type != 7)
      throw std::runtime_error("integer input required");
    return reinterpret_cast<const int64_t *>(data.data());
  }
};
#ifdef TRANSCRIBRO_TRACE
void trace_file(void *context, const char *label, const float *data,
                size_t count) {
  std::string name(label);
  std::replace(name.begin(), name.end(), '/', '~');
  std::ofstream f(*static_cast<std::filesystem::path *>(context) /
                      (name + ".f32"),
                  std::ios::binary);
  f.write(reinterpret_cast<const char *>(data), count * sizeof(float));
  if (!f)
    std::abort();
}
#endif
} // namespace

int main(int argc, char **argv) try {
  if (argc != 5)
    throw std::runtime_error(
        "usage: run_components ARCHIVE KIND INPUTS OUTPUT");
  const std::string kind = argv[2];
  const std::filesystem::path source = argv[3], dest = argv[4];
  const double start = now();
#ifdef TRANSCRIBRO_ORT_REFERENCE
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "component-reference");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.DisableMemPattern();
  Ort::Session model(env, argv[1], options);
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#else
  char error[1024]{};
  std::unique_ptr<void, decltype(&tm_destroy)> model(
      tm_create(argv[1], error, sizeof(error)), tm_destroy);
  if (!model)
    throw std::runtime_error(error);
#endif
  const double loaded = now();
  std::vector<std::filesystem::path> cases;
  for (const auto &entry : std::filesystem::directory_iterator(source))
    if (entry.is_directory())
      cases.push_back(entry.path());
  std::sort(cases.begin(), cases.end(), [](const auto &a, const auto &b) {
    return std::stoi(a.filename()) < std::stoi(b.filename());
  });
  std::cout << "{\"load_ms\":" << (loaded - start) * 1000 << ",\"cases\":[";
  for (size_t index = 0; index < cases.size(); ++index) {
    std::ifstream list(cases[index] / "inputs.txt");
    std::vector<std::string> names;
    std::vector<Input> inputs;
    std::string name;
    while (std::getline(list, name)) {
      names.push_back(name);
      inputs.emplace_back(cases[index] /
                          ("input-" + std::to_string(inputs.size()) + ".bin"));
    }
    std::vector<std::vector<float>> outputs;
    if (kind == "encoder" && names == std::vector<std::string>{"features"}) {
      if (inputs[0].shape.size() != 3 || inputs[0].shape[0] != 1 ||
          inputs[0].shape[2] != 768)
        throw std::runtime_error("encoder input shape");
      outputs.emplace_back(inputs[0].shape[1] * 768);
    } else if (kind == "decoder_kv" &&
               names == std::vector<std::string>{"token", "k_self", "v_self",
                                                 "out_k_cross",
                                                 "out_v_cross"}) {
      outputs.emplace_back(inputs[0].shape.at(1) * 32768);
      for (size_t i = 0; i < 2; ++i)
        outputs.emplace_back(14 * 640 *
                             (inputs[1].shape.at(3) + inputs[0].shape[1]));
      for (size_t i = 3; i < 5; ++i)
        outputs.emplace_back(inputs[i].floats(),
                             inputs[i].floats() +
                                 14 * 640 * inputs[i].shape.at(3));
    } else
      throw std::runtime_error("unsupported component inputs");
    auto run = [&] {
#ifdef TRANSCRIBRO_ORT_REFERENCE
      std::vector<const char *> input_names, output_names;
      std::vector<Ort::Value> values;
      std::vector<Ort::AllocatedStringPtr> owned_names;
      Ort::AllocatorWithDefaultOptions allocator;
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto &input = inputs[i];
        input_names.push_back(names[i].c_str());
        values.push_back(Ort::Value::CreateTensor(
            memory, input.data.data(), input.data.size() * sizeof(uint64_t),
            input.shape.data(), input.shape.size(), ONNXTensorElementDataType(input.type)));
      }
      for (size_t i = 0; i < model.GetOutputCount(); ++i) {
        owned_names.push_back(model.GetOutputNameAllocated(i, allocator));
        output_names.push_back(owned_names.back().get());
      }
      auto results = model.Run(Ort::RunOptions{nullptr}, input_names.data(),
                               values.data(), values.size(), output_names.data(),
                               output_names.size());
      if (results.size() != outputs.size())
        throw std::runtime_error("reference output count");
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (results[i].GetTensorTypeAndShapeInfo().GetElementCount() != outputs[i].size())
          throw std::runtime_error("reference output extent");
        std::memcpy(outputs[i].data(), results[i].GetTensorData<float>(),
                    outputs[i].size() * sizeof(float));
      }
#else
      int code;
      if (kind == "encoder")
        code = tm_encoder(model.get(), inputs[0].floats(), inputs[0].shape[1],
                          outputs[0].data());
      else
        code = tm_decoder(
            model.get(), inputs[0].integers(), inputs[0].shape[1],
            inputs[1].floats(), inputs[2].floats(), inputs[1].shape[3],
            inputs[3].floats(), inputs[4].floats(), inputs[3].shape[3],
            outputs[0].data(), outputs[1].data(), outputs[2].data());
      if (code)
        throw std::runtime_error(tm_error());
#endif
    };
#ifdef TRANSCRIBRO_TRACE
    auto trace_directory = dest / cases[index].filename() / "trace";
    std::filesystem::create_directories(trace_directory);
    tm_set_trace(trace_file, &trace_directory);
#endif
    const auto wall = now(), cpu = now(CLOCK_PROCESS_CPUTIME_ID);
    run();
#ifdef TRANSCRIBRO_TRACE
    tm_set_trace(nullptr, nullptr);
#endif
    const auto wall_ms = (now() - wall) * 1000,
               cpu_ms = (now(CLOCK_PROCESS_CPUTIME_ID) - cpu) * 1000;
    const auto directory = dest / cases[index].filename();
    std::filesystem::create_directories(directory);
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::ofstream f(directory / ("output-" + std::to_string(i) + ".bin"),
                      std::ios::binary);
      f.write(reinterpret_cast<const char *>(outputs[i].data()),
              outputs[i].size() * sizeof(float));
      if (!f)
        throw std::runtime_error("output write");
    }
    if (index)
      std::cout << ',';
    std::cout << "{\"case\":\"" << cases[index].filename().string()
              << "\",\"wall_ms\":" << wall_ms << ",\"cpu_ms\":" << cpu_ms
              << '}';
    std::cout.flush();
  }
  rusage resources{};
  getrusage(RUSAGE_SELF, &resources);
  std::cout << "],\"max_rss_kib\":" << resources.ru_maxrss << "}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
