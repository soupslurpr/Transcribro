#pragma once
#include "mlas.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace transcribro::standalone {
// Separate 128-element K panels preserve
// the unmodified multi-row SGEMM's partial sums. The one-row ARM path keeps
// one accumulator per output across every panel, as the reference GEMV does.
class PackedHead {
  static constexpr size_t width = 32768, depth = 640, panel = 128;
  std::vector<float> storage;
  float *weights;
  size_t columns;

public:
  template <class Weight>
  explicit PackedHead(Weight value, size_t column_block = 0)
      : storage(width * depth + 16), columns(column_block) {
    if (columns != 0 && columns != 64 && columns != 128)
      throw std::invalid_argument("unsupported head panel width");
    weights = reinterpret_cast<float *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + 63) & ~uintptr_t(63));
    alignas(64) std::array<float, panel * panel> tile;
    const size_t block = columns ? columns : panel;
    if (MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, block, panel) !=
        block * panel * sizeof(float))
      throw std::runtime_error("unexpected pinned FP32 packing layout");
    for (size_t k = 0; k < depth; k += panel)
      for (size_t c = 0; c < width; c += block) {
        for (size_t j = 0; j < panel; ++j)
          for (size_t i = 0; i < block; ++i)
            tile[j * block + i] = value(k + j, c + i);
        MlasGemmPackB(CblasNoTrans, CblasNoTrans, block, panel, tile.data(),
                      block,
                      weights + (columns ? c * depth + k * columns
                                         : k * width + c * panel));
      }
  }
  PackedHead(const PackedHead &) = delete;
  PackedHead &operator=(const PackedHead &) = delete;
  void run(const float *input, size_t rows, float *output) const {
    run_columns(input, rows, output, 0, width);
  }
  void run_columns(const float *input, size_t rows, float *output, size_t first,
                   size_t last) const {
    const size_t alignment = columns ? columns : 64;
    if (first > last || last > width || first % alignment || last % alignment)
      throw std::invalid_argument("head column extent");
#if defined(__aarch64__)
    if (rows == 1) {
      for (size_t c = first; c < last; c += 64) {
        float32x4_t sum[4][4];
        for (auto &group : sum)
          for (auto &v : group)
            v = vdupq_n_f32(0);
        for (size_t k = 0; k < depth; k += panel) {
          const float *w =
              weights + (columns ? (c / columns) * columns * depth +
                                       k * columns + (c % columns) * panel
                                 : k * width + c * panel);
          for (size_t j = 0; j < panel; ++j) {
            const auto a = vdupq_n_f32(input[k + j]);
            for (size_t group = 0; group < 4; ++group)
              for (size_t lane = 0; lane < 4; ++lane)
                sum[group][lane] = vfmaq_f32(
                    sum[group][lane],
                    vld1q_f32(w + group * panel * 16 + j * 16 + lane * 4), a);
          }
        }
        for (size_t group = 0; group < 4; ++group)
          for (size_t lane = 0; lane < 4; ++lane)
            vst1q_f32(output + c + group * 16 + lane * 4, sum[group][lane]);
      }
      return;
    }
#endif
    if (columns) {
      for (size_t c = first; c < last; c += columns)
        for (size_t k = 0; k < depth; k += panel)
          MlasGemm(CblasNoTrans, rows, columns, panel, 1.0f, input + k, depth,
                   weights + c * depth + k * columns, k == 0 ? 0.0f : 1.0f,
                   output + c, width, nullptr);
    } else {
      for (size_t k = 0; k < depth; k += panel)
        MlasGemm(CblasNoTrans, rows, last - first, panel, 1.0f, input + k,
                 depth, weights + k * width + first * panel,
                 k == 0 ? 0.0f : 1.0f, output + first, width, nullptr);
    }
  }
};
} // namespace transcribro::standalone
