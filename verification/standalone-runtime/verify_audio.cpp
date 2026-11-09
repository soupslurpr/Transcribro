// Evaluation executable only: the standalone library itself does not link ORT.
#include "medium.h"
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read_pcm(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || file.tellg() > 16000 * 4 * 120)
    throw std::runtime_error("PCM file extent");
  const size_t size = file.tellg();
  if (size % 4)
    throw std::runtime_error("PCM alignment");
  std::vector<float> pcm(size / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(pcm.data()), size);
  if (!file)
    throw std::runtime_error("PCM read");
  return pcm;
}
Ort::Value tensor(const Ort::MemoryInfo &memory, float *data, size_t count,
                  const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<float>(memory, data, count, shape.data(),
                                         shape.size());
}
Ort::Value integer_tensor(const Ort::MemoryInfo &memory, int64_t *data,
                          const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<int64_t>(memory, data, 1, shape.data(),
                                           shape.size());
}
struct Comparison {
  size_t values = 0, different = 0;
  float maximum = 0;
  void check(const float *a, const float *b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
        throw std::runtime_error("nonfinite audio result");
      maximum = std::max(maximum, std::abs(a[i] - b[i]));
      if (std::memcmp(a + i, b + i, sizeof(float)))
        ++different;
    }
    values += size;
  }
  void print() const {
    std::cout << "{\"values\":" << values << ",\"different\":" << different
              << ",\"maximum\":" << maximum << '}';
  }
};
} // namespace

int main(int argc, char **argv) try {
  if (argc != 5 && argc != 6)
    throw std::runtime_error(
        "usage: verify_audio ARCHIVE MODEL_DIR SILERO_ORT PCM_F32 [--stream-blocks]");
  char error[1024];
  std::unique_ptr<void, decltype(&tm_destroy)> model(
      tm_create(argv[1], error, sizeof(error)), tm_destroy);
  if (!model)
    throw std::runtime_error(error);
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "standalone-audio-verification");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.DisableMemPattern();
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::string directory = argv[2];
  Ort::Session vad(env, argv[3], options);
  Ort::Session frontend(env, (directory + "/frontend.model.ort").c_str(),
                        options);
  Ort::Session weight_session(
      env, (directory + "/frontend.weights.ort").c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> weight_names;
  std::vector<const char *> names;
  for (size_t i = 0; i < weight_session.GetOutputCount(); ++i)
    weight_names.emplace_back(
        weight_session.GetOutputNameAllocated(i, allocator).get());
  for (const auto &name : weight_names)
    names.push_back(name.c_str());
  auto weights = weight_session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr,
                                    0, names.data(), names.size());
  auto audio = read_pcm(argv[4]);
  tm_vad_state candidate_vad{};
  std::array<float, 64> context{};
  std::array<float, 256> state{};
  std::array<float, 576> vad_audio{};
  int64_t sr = 16000;
  Comparison vad_probability, vad_state;
  for (size_t at = 0; at + 512 <= audio.size(); at += 512) {
    std::copy(context.begin(), context.end(), vad_audio.begin());
    std::copy_n(audio.data() + at, 512, vad_audio.data() + 64);
    std::vector<Ort::Value> inputs;
    inputs.push_back(tensor(memory, vad_audio.data(), 576, {1, 576}));
    inputs.push_back(tensor(memory, state.data(), 256, {2, 1, 128}));
    inputs.push_back(integer_tensor(memory, &sr, {}));
    const char *in_names[]{"input", "state", "sr"},
        *out_names[]{"output", "stateN"};
    auto out = vad.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3,
                       out_names, 2);
    float probability;
    if (tm_vad(model.get(), audio.data() + at, &candidate_vad, &probability))
      throw std::runtime_error(tm_error());
    vad_probability.check(out[0].GetTensorData<float>(), &probability, 1);
    vad_state.check(out[1].GetTensorData<float>(), candidate_vad.hidden, 128);
    vad_state.check(out[1].GetTensorData<float>() + 128, candidate_vad.cell,
                    128);
    std::copy_n(out[1].GetTensorData<float>(), 256, state.data());
    std::copy_n(audio.data() + at + 448, 64, context.data());
  }
  const bool stream_blocks = argc == 6 && std::string(argv[5]) == "--stream-blocks";
  tm_frontend_state candidate{}, reference{};
  std::vector<float> stream_features;
  Comparison features, recurrent, pair_features;
  const size_t sizes[]{640, 1600, 2560, 5120, 1293, 3200, 16000};
  size_t at = 0, step = 0;
  while (at + (stream_blocks ? 1280 : 640) <= audio.size()) {
    const size_t size = stream_blocks ? 1280 : std::min(sizes[step++ % 7], audio.size() - at);
    std::vector<Ort::Value> inputs;
    inputs.push_back(
        tensor(memory, audio.data() + at, size, {1, int64_t(size)}));
    inputs.push_back(tensor(memory, reference.sample_buffer, 79, {1, 79}));
    inputs.push_back(integer_tensor(memory, &reference.sample_len, {1}));
    inputs.push_back(tensor(memory, reference.conv1, 768 * 4, {1, 768, 4}));
    inputs.push_back(tensor(memory, reference.conv2, 1536 * 4, {1, 1536, 4}));
    inputs.push_back(integer_tensor(memory, &reference.frame_count, {1}));
    std::vector<const char *> in_names{"audio_chunk",  "sample_buffer",
                                       "sample_len",   "conv1_buffer",
                                       "conv2_buffer", "frame_count"};
    for (size_t i = 0; i < weights.size(); ++i) {
      auto shape = weights[i].GetTensorTypeAndShapeInfo();
      inputs.push_back(tensor(memory, weights[i].GetTensorMutableData<float>(),
                              shape.GetElementCount(), shape.GetShape()));
      in_names.push_back(names[i]);
    }
    const char *out_names[]{"features",         "sample_buffer_out",
                            "sample_len_out",   "conv1_buffer_out",
                            "conv2_buffer_out", "frame_count_out"};
    auto out = frontend.Run(Ort::RunOptions{nullptr}, in_names.data(),
                            inputs.data(), inputs.size(), out_names, 6);
    const size_t capacity = (size + 319) / 320;
    std::vector<float> actual(capacity * 768);
    size_t count;
    if (tm_frontend(model.get(), audio.data() + at, size, &candidate,
                    actual.data(), capacity, &count))
      throw std::runtime_error(tm_error());
    if (out[0].GetTensorTypeAndShapeInfo().GetElementCount() != count * 768)
      throw std::runtime_error("frontend feature extent");
    features.check(out[0].GetTensorData<float>(), actual.data(), count * 768);
    if (stream_blocks)
      stream_features.insert(stream_features.end(), out[0].GetTensorData<float>(),
                             out[0].GetTensorData<float>() + count * 768);
    recurrent.check(out[1].GetTensorData<float>(), candidate.sample_buffer, 79);
    recurrent.check(out[3].GetTensorData<float>(), candidate.conv1, 768 * 4);
    recurrent.check(out[4].GetTensorData<float>(), candidate.conv2, 1536 * 4);
    if (out[2].GetTensorData<int64_t>()[0] != candidate.sample_len ||
        out[5].GetTensorData<int64_t>()[0] != candidate.frame_count)
      throw std::runtime_error("frontend integer state differs");
    std::copy_n(out[1].GetTensorData<float>(), 79, reference.sample_buffer);
    reference.sample_len = out[2].GetTensorData<int64_t>()[0];
    std::copy_n(out[3].GetTensorData<float>(), 768 * 4, reference.conv1);
    std::copy_n(out[4].GetTensorData<float>(), 1536 * 4, reference.conv2);
    reference.frame_count = out[5].GetTensorData<int64_t>()[0];
    at += size;
  }
  if (stream_blocks) {
    tm_frontend_state paired{};
    size_t position = 0, frames = 0;
    while (position < at) {
      const size_t size = std::min(size_t(2560), at - position);
      float actual[8 * 768];
      size_t count;
      if (tm_frontend(model.get(), audio.data() + position, size, &paired, actual, 8, &count))
        throw std::runtime_error(tm_error());
      pair_features.check(stream_features.data() + frames * 768, actual, count * 768);
      frames += count;
      position += size;
    }
    std::cout << "{\"paired_frontend\":";
    pair_features.print();
    std::cout << ",\"remaining\":";
  }
  std::cout << "{\"vad_probability\":";
  vad_probability.print();
  std::cout << ",\"vad_state\":";
  vad_state.print();
  std::cout << ",\"features\":";
  features.print();
  std::cout << ",\"frontend_state\":";
  recurrent.print();
  std::cout << (stream_blocks ? "}}\n" : "}\n");
  return pair_features.different || vad_probability.different || vad_state.different ||
                 features.different || recurrent.different
             ? 2
             : 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
