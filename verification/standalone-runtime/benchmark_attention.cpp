#include "mlas.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <vector>
namespace {
double cpu() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
uint32_t random_state = 739;
float value() {
  random_state = 1664525 * random_state + 1013904223;
  return float(int32_t(random_state % 200001) - 100000) / 32768;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 2)
    throw std::runtime_error("usage: benchmark_attention OUTPUT_JSON");
  std::ofstream output(argv[1]);
  output << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (int kind = 0; kind < 2; ++kind)
    for (size_t memory :
         {1, 17, 64, 65, 127, 128, 129, 255, 256, 257, 384, 500, 513}) {
      const size_t n = kind == 0 ? memory : 64, k = kind == 0 ? 64 : memory;
      const auto transpose = kind == 0 ? CblasTrans : CblasNoTrans;
      const float scale = kind == 0 ? .125f : 1.f;
      std::vector<float> weights(memory * 64);
      for (auto &w : weights)
        w = value();
      const size_t size = MlasGemmPackBSize(CblasNoTrans, transpose, n, k);
      std::vector<uint8_t> storage(size + 64);
      void *packed = reinterpret_cast<void *>(
          (reinterpret_cast<uintptr_t>(storage.data()) + 63) & ~uintptr_t(63));
      const double start = cpu();
      MlasGemmPackB(CblasNoTrans, transpose, n, k, weights.data(), 64, packed);
      const double pack_seconds = cpu() - start;
      for (size_t rows : {1, 2, 4, 8, 16, 64}) {
        std::vector<float> x(rows * k), reference(rows * n), actual(rows * n);
        for (auto &a : x)
          a = value();
        MlasGemm(CblasNoTrans, transpose, rows, n, k, scale, x.data(), k,
                 weights.data(), 64, 0.f, reference.data(), n, nullptr);
        for (int round = 0; round < 3; ++round)
          for (int order = 0; order < 2; ++order) {
            const bool candidate = (round + order) % 2;
            auto run = [&] {
              if (candidate)
                MlasGemm(CblasNoTrans, rows, n, k, scale, x.data(), k, packed,
                         0.f, actual.data(), n, nullptr);
              else
                MlasGemm(CblasNoTrans, transpose, rows, n, k, scale, x.data(),
                         k, weights.data(), 64, 0.f, actual.data(), n, nullptr);
            };
            run();
            const double began = cpu();
            for (int repeat = 0; repeat < 32; ++repeat)
              run();
            const double elapsed = (cpu() - began) / 32;
            size_t differences = 0;
            for (size_t i = 0; i < actual.size(); ++i)
              differences += std::memcmp(&actual[i], &reference[i], 4) != 0;
            if (!first)
              output << ',';
            first = false;
            output << "{\"kind\":\"" << (kind == 0 ? "query_key" : "value")
                   << "\",\"memory\":" << memory << ",\"rows\":" << rows
                   << ",\"round\":" << round
                   << ",\"candidate\":" << (candidate ? "true" : "false")
                   << ",\"differences\":" << differences
                   << ",\"packed_bytes\":" << size
                   << ",\"pack_seconds\":" << pack_seconds
                   << ",\"cpu_seconds\":" << elapsed << '}';
          }
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
