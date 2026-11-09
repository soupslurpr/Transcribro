#include "fixture.h"
#include "compact_head.h"
#include "token_selection.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <time.h>
using namespace transcribro::research;
using transcribro::standalone::TokenSelection;
namespace {
double seconds(clockid_t clock) {
  timespec t{};
  clock_gettime(clock, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void edge_cases() {
  constexpr size_t width = 256, rows = 8;
  std::vector<float> values(rows * width, -1.f);
  const auto inf = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  values[0] = nan;
  values[width + 64] = nan;
  values[width + 65] = 2;
  values[2 * width + 63] = values[2 * width + 64] = 2;
  values[3 * width + 129] = inf;
  std::fill_n(values.data() + 4 * width, width, -inf);
  values[5 * width] = -0.f;
  values[5 * width + 64] = 0.f;
  std::fill_n(values.data() + 6 * width, width, nan);
  values[7 * width + 255] = 3;
  for (size_t columns : {size_t(1), size_t(64), size_t(128), width}) {
    TokenSelection selection;
    std::array<int64_t, rows> predictions{};
    selection.run(
        rows, width, columns,
        [&](size_t c, float *block) {
          for (size_t r = 0; r < rows; ++r)
            std::copy_n(values.data() + r * width + c, columns,
                        block + r * columns);
        },
        predictions.data());
    for (size_t r = 0; r < rows; ++r) {
      const auto *begin = values.data() + r * width;
      require(predictions[r] == std::max_element(begin, begin + width) - begin,
              "token tie/NaN/infinity behavior changed");
    }
  }
}
} // namespace
int main(int argc, char **argv) try {
  require(argc == 3, "usage: benchmark_token_selection HEAD_BIN OUTPUT_JSON");
  edge_cases();
  std::ifstream file(argv[1], std::ios::binary);
  const auto magic = read_values<char>(file, 8);
  require(std::memcmp(magic.data(), "TRHEAD1\0", 8) == 0, "head magic");
  const auto h = read_values<uint32_t>(file, 4);
  const size_t k = h[0], n = h[1], cases = h[2];
  require(k == 640 && n == 32768 && cases == 64 && h[3] <= 255,
          "head geometry");
  const float scale = read_values<float>(file, 1)[0];
  const auto original = read_values<uint8_t>(file, k * n);
  auto x = read_values<float>(file, k * cases);
  x.resize(k * 257);
  for (size_t i = k * cases; i < x.size(); ++i)
    x[i] = x[i % (k * cases)];
  // The actual archive layout; the candidate borrows it without another copy.
  std::vector<uint8_t> packed(k * n);
  for (size_t c = 0; c < n; ++c)
    for (size_t j = 0; j < k; ++j) {
      const size_t at = (c / 64) * (640 * 64) + (j / 128) * (128 * 64) +
                        (c % 64 / 16) * (128 * 16) + (j % 128) * 16 + c % 16;
      packed[at] = original[c * k + j];
    }
  transcribro::standalone::CompactHead<false> head(
      static_cast<const uint8_t *>(packed.data()), scale, int32_t(h[3]));
  std::ofstream out(argv[2]);
  out << std::setprecision(12) << "{\"edge_cases_passed\":true,\"cases\":[";
  bool first = true;
  for (size_t rows : {1, 2, 3, 4, 5, 7, 8, 16, 31, 63, 64, 65, 127, 128, 129,
                      255, 256, 257}) {
    std::vector<float> reference(rows * n), actual(rows * n);
    std::vector<int64_t> expected(rows), predictions(rows);
    TokenSelection selection;
    for (size_t offset = 0; offset + rows <= std::max(cases, rows);
         offset += rows) {
      const auto *input = x.data() + offset * k;
      auto baseline = [&] {
        head.run(input, rows, reference.data());
        for (size_t r = 0; r < rows; ++r) {
          const auto *begin = reference.data() + r * n;
          expected[r] = std::max_element(begin, begin + n) - begin;
        }
      };
      auto candidate = [&] {
        selection.run(
            rows, n, 64,
            [&](size_t c, float *block) {
              head.run_columns(input, rows, block, 64, c, c + 64);
            },
            predictions.data());
      };
      baseline();
      selection.run(
          rows, n, 64,
          [&](size_t c, float *block) {
            head.run_columns(input, rows, block, 64, c, c + 64);
            for (size_t r = 0; r < rows; ++r)
              std::copy_n(block + r * 64, 64, actual.data() + r * n + c);
          },
          predictions.data());
      require(std::memcmp(reference.data(), actual.data(),
                          reference.size() * sizeof(float)) == 0,
              "tiled score bits changed");
      require(expected == predictions, "tiled prediction changed");
      for (int round = 0; round < (offset == 0 ? 3 : 1); ++round)
        for (int order = 0; order < 2; ++order) {
          const bool fused = (round + order) % 2;
          const double wall = seconds(CLOCK_MONOTONIC),
                       cpu = seconds(CLOCK_PROCESS_CPUTIME_ID);
          if (fused)
            candidate();
          else
            baseline();
          const double cpu_used = seconds(CLOCK_PROCESS_CPUTIME_ID) - cpu,
                       wall_used = seconds(CLOCK_MONOTONIC) - wall;
          require(expected == predictions, "timed prediction changed");
          if (!first)
            out << ',';
          first = false;
          out << "{\"rows\":" << rows << ",\"offset\":" << offset
              << ",\"round\":" << round
              << ",\"fused\":" << (fused ? "true" : "false")
              << ",\"all_scores_exact\":true,\"predictions_exact\":true"
              << ",\"selection_capacity_bytes\":" << selection.capacity_bytes()
              << ",\"full_scores_bytes\":" << reference.size() * sizeof(float)
              << ",\"cpu_seconds\":" << cpu_used
              << ",\"wall_seconds\":" << wall_used << '}';
        }
    }
  }
  out << "]}\n";
  require(bool(out), "output write");
  std::cout << "All score bits, predictions and tie/NaN cases passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
