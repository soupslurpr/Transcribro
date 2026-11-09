// Common native-only benchmark driver. Reference C API declarations are from
// the pinned Moonshine source (MIT); see CREDITS and licenses/Moonshine.txt.
#include "moonshine-c-api.h"
#include "stream.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
double now(clockid_t clock = CLOCK_MONOTONIC) {
  timespec t{};
  clock_gettime(clock, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::string quoted(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << unsigned(c);
    else
      out << c;
  }
  out << '"';
  return out.str();
}
size_t pss() {
  std::ifstream file("/proc/self/smaps_rollup");
  std::string line;
  while (std::getline(file, line))
    if (line.rfind("Pss:", 0) == 0)
      return std::stoull(line.substr(4));
  throw std::runtime_error("cannot read process PSS");
}
struct Mapping {
  void *data = nullptr;
  size_t size = 0;
  explicit Mapping(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "model open");
    struct stat st{};
    if (fstat(fd, &st) || st.st_size <= 0 || st.st_size > 512 * 1024 * 1024) {
      close(fd);
      throw std::runtime_error("model extent");
    }
    size = size_t(st.st_size);
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    require(data != MAP_FAILED, "model mapping");
  }
  ~Mapping() {
    if (data && data != MAP_FAILED)
      munmap(data, size);
  }
};
struct Engine {
  virtual ~Engine() = default;
  virtual void reset() = 0;
  virtual void push(const float *, size_t) = 0;
  virtual std::string update(bool) = 0;
};
struct Native : Engine {
  void *handle;
  Native(const char *archive, const char *tokenizer) {
    char error[1024]{};
    handle = ts_create(archive, tokenizer, error, sizeof(error));
    require(handle, error);
  }
  ~Native() override { ts_destroy(handle); }
  void reset() override {
    // Corpus recordings are independent sessions. Retaining the detector's
    // hidden state would make results depend on preceding clips and batches.
    if (ts_reset(handle, 0))
      throw std::runtime_error(ts_error());
  }
  void push(const float *audio, size_t samples) override {
    if (ts_push(handle, audio, samples))
      throw std::runtime_error(ts_error());
  }
  std::string update(bool final) override {
    if (ts_update(handle, final))
      throw std::runtime_error(ts_error());
    return ts_text(handle);
  }
};
struct Reference : Engine {
  void *library = nullptr;
  int32_t handle = -1, stream = -1;
  std::vector<std::unique_ptr<Mapping>> files;
  decltype(&moonshine_load_transcriber_from_memory_files) load;
  decltype(&moonshine_free_transcriber) release;
  decltype(&moonshine_create_stream) create;
  decltype(&moonshine_free_stream) free_stream;
  decltype(&moonshine_start_stream) start;
  decltype(&moonshine_stop_stream) stop;
  decltype(&moonshine_transcribe_add_audio_to_stream) add;
  decltype(&moonshine_transcribe_stream) transcribe;
  template <class T> void symbol(T &target, const char *name) {
    target = reinterpret_cast<T>(dlsym(library, name));
    require(target, name);
  }
  Reference(const char *path, const char *directory) {
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(dlerror());
    symbol(load, "moonshine_load_transcriber_from_memory_files");
    symbol(release, "moonshine_free_transcriber");
    symbol(create, "moonshine_create_stream");
    symbol(free_stream, "moonshine_free_stream");
    symbol(start, "moonshine_start_stream");
    symbol(stop, "moonshine_stop_stream");
    symbol(add, "moonshine_transcribe_add_audio_to_stream");
    symbol(transcribe, "moonshine_transcribe_stream");
    const char *names[]{"adapter.ort",           "cross_kv.ort",
                        "decoder_kv.ort",        "encoder.ort",
                        "frontend.model.ort",    "frontend.weights.ort",
                        "streaming_config.json", "tokenizer.bin"};
    std::array<const uint8_t *, 8> data;
    std::array<uint64_t, 8> sizes;
    for (size_t i = 0; i < 8; ++i) {
      files.push_back(
          std::make_unique<Mapping>(std::string(directory) + '/' + names[i]));
      data[i] = static_cast<const uint8_t *>(files.back()->data);
      sizes[i] = files.back()->size;
    }
    const moonshine_option_t options[]{{"return_audio_data", "false"},
                                       {"log_output_text", "false"},
                                       {"transcription_interval", "1.0"},
                                       {"vad_threshold", "0.3"},
                                       {"vad_window_duration", "1.0"},
                                       {"vad_look_behind_sample_count", "8192"},
                                       {"max_tokens_per_second", "20"},
                                       {"max_preview_tokens_per_second", "6.5"},
                                       {"decode_incomplete_lines", "true"}};
    handle = load(names, data.data(), sizes.data(), 8, 5, options, 9,
                  MOONSHINE_HEADER_VERSION);
    require(handle >= 0, "reference load failed");
  }
  ~Reference() override {
    if (stream >= 0)
      free_stream(handle, stream);
    if (handle >= 0)
      release(handle);
    if (library)
      dlclose(library);
  }
  void reset() override {
    if (stream >= 0)
      require(free_stream(handle, stream) == 0, "reference free stream");
    stream = create(handle, 0);
    require(stream >= 0 && start(handle, stream) == 0,
            "reference start stream");
  }
  void push(const float *audio, size_t samples) override {
    require(add(handle, stream, audio, samples, 16000, 0) == 0,
            "reference push");
  }
  std::string update(bool final) override {
    if (final)
      require(stop(handle, stream) == 0, "reference stop");
    transcript_t *transcript = nullptr;
    require(transcribe(handle, stream, 0, &transcript) == 0 && transcript,
            "reference update");
    std::string text;
    for (size_t i = 0; i < transcript->line_count; ++i) {
      if (i)
        text += ' ';
      if (transcript->lines[i].text)
        text += transcript->lines[i].text;
    }
    auto first = text.find_first_not_of(" \t\r\n\v\f"),
         last = text.find_last_not_of(" \t\r\n\v\f");
    return first == std::string::npos ? ""
                                      : text.substr(first, last - first + 1);
  }
};
std::vector<float> read_pcm(const std::string &path, size_t &samples) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  // Cover the app's 30-minute audio-file limit as well as five-minute
  // dictation.
  require(bool(file) && file.tellg() >= 0 && file.tellg() <= 16000 * 4 * 1800,
          "PCM extent");
  const size_t bytes = size_t(file.tellg());
  require(bytes % 4 == 0, "PCM alignment");
  samples = bytes / 4;
  std::vector<float> data((samples + 511) / 512 * 512 + 8192, 0);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(data.data()), bytes);
  require(bool(file), "PCM read");
  return data;
}
} // namespace
int main(int argc, char **argv) try {
  require(argc == 6, "usage: run_stream MODE ARCHIVE_OR_MODELS "
                     "TOKENIZER_OR_LIBRARY PCM_LIST OUTPUT_JSON");
  const std::string mode = argv[1];
  require(mode == "native" || mode == "reference" || mode == "reference-independent", "invalid mode");
  // Match the Android engine, which sets this before constructing Moonshine.
  require(setenv("MOONSHINE_ORT_SINGLE_THREAD", "1", 1) == 0,
          "reference thread configuration");
  const double load_start = now(), load_cpu = now(CLOCK_PROCESS_CPUTIME_ID);
  std::unique_ptr<Engine> engine;
  if (mode == "native")
    engine = std::make_unique<Native>(argv[2], argv[3]);
  else if (mode == "reference")
    engine = std::make_unique<Reference>(argv[3], argv[2]);
  const double load_seconds = now() - load_start,
               load_cpu_seconds = now(CLOCK_PROCESS_CPUTIME_ID) - load_cpu;
  const size_t load_pss = pss();
  std::ifstream list(argv[4]);
  require(bool(list), "PCM list missing");
  std::ofstream out(argv[5]);
  require(bool(out), "output file");
  out << std::setprecision(10) << "{\"mode\":" << quoted(mode)
      << ",\"independent_recordings\":" << (mode != "reference" ? "true" : "false")
      << ",\"load_seconds\":" << load_seconds
      << ",\"load_cpu_seconds\":" << load_cpu_seconds
      << ",\"load_pss_kib\":" << load_pss << ",\"clips\":[";
  std::string path;
  size_t index = 0, peak_pss = load_pss;
  while (std::getline(list, path)) {
    if (path.empty())
      continue;
    // Moonshine's inner Silero is a deliberately leaked process-global singleton.
    // Recreating a transcriber does not reset it. Fork from a parent that has
    // never loaded Moonshine so every reference recording starts with zero VAD.
    pid_t child = -1;
    if (mode == "reference-independent") {
      if (index++) out << ',';
      out.flush();
      child = fork();
      require(child >= 0, "reference process creation");
      if (child > 0) {
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                    WEXITSTATUS(status) == 0, "reference recording failed");
        std::cout << index << ' ' << path << '\n' << std::flush;
        continue;
      }
      engine = std::make_unique<Reference>(argv[3], argv[2]);
    }
    size_t samples;
    auto audio = read_pcm(path, samples);
    engine->reset();
    const double started = now(), cpu_started = now(CLOCK_PROCESS_CPUTIME_ID);
    size_t pushed = 0, analyzed = 0;
    std::string text;
    if (mode != "reference-independent" && index++)
      out << ',';
    out << "{\"path\":" << quoted(path)
        << ",\"audio_seconds\":" << double(samples) / 16000 << ",\"updates\":[";
    bool comma = false;
    const size_t padded = audio.size() - 8192;
    for (size_t end = std::min(size_t(8192), audio.size());;) {
      const bool final = end == audio.size();
      engine->push(audio.data() + pushed, end - pushed);
      pushed = end;
      if (end - analyzed >= 16000 || final) {
        const double wall = now(), cpu = now(CLOCK_PROCESS_CPUTIME_ID);
        text = engine->update(final);
        const double elapsed = now() - wall,
                     cpu_seconds = now(CLOCK_PROCESS_CPUTIME_ID) - cpu;
        const size_t memory = pss();
        peak_pss = std::max(peak_pss, memory);
        if (comma)
          out << ',';
        comma = true;
        out << "{\"samples\":" << end
            << ",\"final\":" << (final ? "true" : "false")
            << ",\"wall_seconds\":" << elapsed
            << ",\"cpu_seconds\":" << cpu_seconds << ",\"pss_kib\":" << memory
            << ",\"text\":" << quoted(text) << '}';
        analyzed = end;
      }
      if (final)
        break;
      end = end + 8192 <= padded ? end + 8192 : audio.size();
    }
    out << "],\"wall_seconds\":" << now() - started
        << ",\"cpu_seconds\":" << now(CLOCK_PROCESS_CPUTIME_ID) - cpu_started
        << ",\"text\":" << quoted(text) << '}';
    out.flush();
    if (child == 0) {
      out.flush();
      engine.reset();
      _exit(0);
    }
    std::cout << index << ' ' << path << '\n' << std::flush;
  }
  rusage resources{};
  getrusage(RUSAGE_SELF, &resources);
  std::ifstream maps("/proc/self/maps");
  std::string mapped((std::istreambuf_iterator<char>(maps)), {});
  const bool ort_loaded = mode == "reference-independent" ||
                          mapped.find("libonnxruntime.so") != std::string::npos;
  require(mode != "native" || !ort_loaded, "native execution loaded ORT");
  out << "],\"peak_sampled_pss_kib\":" << peak_pss
      << ",\"max_rss_kib\":" << resources.ru_maxrss
      << ",\"ort_loaded\":" << (ort_loaded ? "true" : "false") << "}\n";
  require(bool(out), "output write");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
