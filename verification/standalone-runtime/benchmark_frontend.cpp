// Compare batched frontend execution to the established 80 ms boundaries.
#include "medium.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
double now(clockid_t clock) {
  timespec value{};
  clock_gettime(clock, &value);
  return double(value.tv_sec) + double(value.tv_nsec) / 1e9;
}
bool equal_state(const tm_frontend_state &a, const tm_frontend_state &b) {
  return a.sample_len == b.sample_len && a.frame_count == b.frame_count &&
         !std::memcmp(a.sample_buffer, b.sample_buffer,
                      sizeof(a.sample_buffer)) &&
         !std::memcmp(a.conv1, b.conv1, sizeof(a.conv1)) &&
         !std::memcmp(a.conv2, b.conv2, sizeof(a.conv2));
}
std::vector<float> read_pcm(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file) && file.tellg() > 0 && file.tellg() <= 16000 * 4 * 120,
          "bounded PCM fixture required");
  const size_t bytes = size_t(file.tellg());
  require(bytes % 4 == 0, "PCM extent");
  std::vector<float> audio((bytes / 4 + 1279) / 1280 * 1280, 0);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(audio.data()), bytes);
  require(bool(file), "PCM read");
  return audio;
}
void run(void *model, const float *audio, size_t samples,
         tm_frontend_state &state, float *output) {
  size_t frames = 0;
  if (tm_frontend(model, audio, samples, &state, output, samples / 320,
                  &frames))
    throw std::runtime_error(tm_error());
  require(frames == samples / 320, "frontend shape");
}
} // namespace

int main(int argc, char **argv) try {
  require(argc == 5,
          "usage: benchmark_frontend ARCHIVE PCM_LIST ROUNDS OUTPUT_JSON");
  const int rounds = std::stoi(argv[3]);
  require(rounds > 0 && rounds <= 10, "bounded rounds required");
  char error[1024]{};
  std::unique_ptr<void, decltype(&tm_destroy)> model(
      tm_create(argv[1], error, sizeof(error)), tm_destroy);
  require(bool(model), error);
  std::ifstream list(argv[2]);
  require(bool(list), "PCM list missing");
  std::ofstream output(argv[4]);
  require(bool(output), "cannot create report");
  output << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  std::string path;
  while (std::getline(list, path)) {
    require(path.find_first_of("\"\\\r") == std::string::npos,
            "fixture path cannot contain JSON escapes");
    auto audio = read_pcm(path);
    std::vector<float> expected(audio.size() / 320 * 768);
    std::vector<tm_frontend_state> states;
    tm_frontend_state state{};
    for (size_t at = 0; at < audio.size(); at += 1280) {
      run(model.get(), audio.data() + at, 1280, state,
          expected.data() + at / 320 * 768);
      states.push_back(state);
    }
    for (int repeat = 0; repeat < rounds; ++repeat) {
      std::array<size_t, 8> groups{1, 2, 3, 4, 6, 8, 12, 24};
      if (repeat % 2)
        std::reverse(groups.begin(), groups.end());
      for (const size_t blocks : groups) {
        state = {};
        std::vector<float> actual(blocks * 4 * 768);
        size_t differences = 0, state_differences = 0;
        double maximum = 0, cpu = 0, wall = 0;
        for (size_t at = 0; at < audio.size(); at += blocks * 1280) {
          const size_t count = std::min(blocks * 1280, audio.size() - at);
          const double start_wall = now(CLOCK_MONOTONIC);
          const double start_cpu = now(CLOCK_THREAD_CPUTIME_ID);
          run(model.get(), audio.data() + at, count, state, actual.data());
          cpu += now(CLOCK_THREAD_CPUTIME_ID) - start_cpu;
          wall += now(CLOCK_MONOTONIC) - start_wall;
          const float *reference = expected.data() + at / 320 * 768;
          for (size_t i = 0; i < count / 320 * 768; ++i) {
            require(std::isfinite(actual[i]), "nonfinite feature");
            differences +=
                std::memcmp(actual.data() + i, reference + i, 4) != 0;
            maximum =
                std::max(maximum, std::abs(double(actual[i]) - reference[i]));
          }
          state_differences +=
              !equal_state(state, states[(at + count) / 1280 - 1]);
        }
        output << (first ? "" : ",") << "{\"clip\":\"" << path
               << "\",\"round\":" << repeat << ",\"blocks\":" << blocks
               << ",\"samples\":" << audio.size()
               << ",\"different_features\":" << differences
               << ",\"different_states\":" << state_differences
               << ",\"maximum_error\":" << maximum << ",\"cpu_seconds\":" << cpu
               << ",\"wall_seconds\":" << wall << '}';
        output.flush();
        first = false;
        std::cout << path << ' ' << repeat << ' ' << blocks << ' '
                  << differences << ' ' << state_differences << '\n';
      }
    }
  }
  output << "]}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
