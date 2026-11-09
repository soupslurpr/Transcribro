#pragma once
#include "mlas.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace transcribro::standalone {
// Keep the full softmax row, including its original lane alignment. Skip
// only QK entries overwritten by the mask and zero-probability AV terms.
// AV still rounds and adds partial sums at the reference's K=256 boundaries.
inline void local_attention_head(const float *queries, const float *keys,
                                 const float *values, size_t rows, int mask,
                                 float scale, float *output,
                                 std::vector<float> &scores) {
  scores.resize(rows);
  std::array<float, 4 * 24> window;
  for (size_t group = 0; group < rows; group += 4) {
    const size_t group_rows = std::min(size_t(4), rows - group);
    const size_t window_start = group > 16 ? group - 16 : 0;
    const size_t window_end =
        std::min(rows, group + group_rows + (mask == 2 ? 4 : 0));
    const size_t window_size = window_end - window_start;
    MlasGemm(CblasNoTrans, CblasTrans, group_rows, window_size, 64, scale,
             queries + group * 64, 64, keys + window_start * 64, 64, 0.f,
             window.data(), window_size, nullptr);
    for (size_t r = group; r < group + group_rows; ++r) {
      const size_t first = r > 16 ? r - 16 : 0;
      const size_t last = std::min(rows, r + (mask == 2 ? 5 : 1));
      std::fill(scores.begin(), scores.end(), -INFINITY);
      std::copy_n(window.data() + (r - group) * window_size + first -
                      window_start,
                  last - first, scores.data() + first);
      MlasComputeSoftmax(scores.data(), scores.data(), 1, rows, false, false,
                         0.f, nullptr);
      for (size_t part = 0; part < rows; part += 256) {
        const size_t begin = std::max(part, first);
        const size_t end = std::min(part + 256, last);
#if defined(__aarch64__)
        float32x4_t sums[16];
        for (auto &s : sums)
          s = vdupq_n_f32(0);
        for (size_t k = begin; k < end; ++k) {
          const auto probability = vdupq_n_f32(scores[k]);
          for (size_t i = 0; i < 16; ++i)
            sums[i] = vfmaq_f32(sums[i], vld1q_f32(values + k * 64 + i * 4),
                                probability);
        }
        for (size_t i = 0; i < 16; ++i) {
          float *destination = output + r * 64 + i * 4;
          vst1q_f32(destination,
                    part == 0 ? sums[i]
                              : vaddq_f32(vld1q_f32(destination), sums[i]));
        }
#else
        for (size_t i = 0; i < 64; ++i) {
          float sum = 0;
          for (size_t k = begin; k < end; ++k)
            sum = std::fma(values[k * 64 + i], scores[k], sum);
          if (part == 0)
            output[r * 64 + i] = sum;
          else
            output[r * 64 + i] += sum;
        }
#endif
      }
    }
  }
}
} // namespace transcribro::standalone
