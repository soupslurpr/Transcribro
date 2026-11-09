#include "local_attention.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <time.h>
namespace {
double cpu() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
uint32_t random_state = 179;
float value() {
  random_state = 1664525 * random_state + 1013904223;
  return float(int32_t(random_state % 200001) - 100000) / 32768;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::runtime_error("usage: benchmark_local_attention OUTPUT_JSON");
  std::ofstream output(argv[1]);
  output << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (int mask : {2, 3})
    for (size_t rows : {1, 4, 17, 32, 64, 65, 127, 128, 129, 255, 256, 257, 384,
                        500, 513, 1024}) {
      std::vector<float> q(rows * 64), k(q.size()), v(q.size()),
          reference(q.size()), actual(q.size()), scores(rows * rows), scratch;
      for (auto *array : {&q, &k, &v})
        for (auto &x : *array)
          x = value();
      auto original = [&](float *result) {
        MlasGemm(CblasNoTrans, CblasTrans, rows, rows, 64, 0.1249999925494194f,
                 q.data(), 64, k.data(), 64, 0.f, scores.data(), rows, nullptr);
        for (size_t r = 0; r < rows; ++r) {
          for (size_t c = 0; c < rows; ++c) {
            const bool allowed =
                mask == 2 ? int64_t(r) - int64_t(c) >= -4 && r <= c + 16
                          : r >= c && r <= c + 16;
            if (!allowed)
              scores[r * rows + c] = -INFINITY;
          }
          MlasComputeSoftmax(scores.data() + r * rows, scores.data() + r * rows,
                             1, rows, false, false, 0.f, nullptr);
        }
        MlasGemm(CblasNoTrans, CblasNoTrans, rows, 64, rows, 1.f, scores.data(),
                 rows, v.data(), 64, 0.f, result, 64, nullptr);
      };
      original(reference.data());
      for (int round = 0; round < 3; ++round)
        for (int order = 0; order < 2; ++order) {
          const bool candidate = (round + order) % 2;
          auto run = [&] {
            if (candidate)
              transcribro::standalone::local_attention_head(
                  q.data(), k.data(), v.data(), rows, mask, 0.1249999925494194f,
                  actual.data(), scratch);
            else
              original(actual.data());
          };
          run();
          const double start = cpu();
          for (int i = 0; i < 4; ++i)
            run();
          const double elapsed = (cpu() - start) / 4;
          size_t differences = 0;
          double maximum_error = 0;
          for (size_t i = 0; i < actual.size(); ++i) {
            differences += std::memcmp(&actual[i], &reference[i], 4) != 0;
            maximum_error = std::max(
                maximum_error, std::abs(double(actual[i]) - reference[i]));
          }
          if (!first)
            output << ',';
          first = false;
          output << "{\"mask\":" << mask << ",\"rows\":" << rows
                 << ",\"round\":" << round
                 << ",\"candidate\":" << (candidate ? "true" : "false")
                 << ",\"differences\":" << differences
                 << ",\"maximum_error\":" << maximum_error
                 << ",\"cpu_seconds\":" << elapsed << '}';
        }
    }
  output << "]}\n";
  if (!output)
    throw std::runtime_error("output write");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
