#pragma once
#include "mlas.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace transcribro::standalone {
// Reconstruct the original FP32 values at use, without another quantization
// step or a permanent expanded output matrix. Borrowed weights must outlive
// this head; the model archive owns them in the Android runtime.
template <bool Centered> class CompactHead {
  static constexpr size_t width = 32768, depth = 640, panel = 128, columns = 64;
  using Stored = std::conditional_t<Centered, int16_t, uint8_t>;
  std::vector<Stored> storage;
  const Stored *weights;
  float scale;
  int32_t zero;

#if defined(__aarch64__)
  static int16x8_t load(const Stored *source, int16x8_t offset) {
    if constexpr (Centered)
      return vld1q_s16(source);
    else
      return vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(source))),
                       offset);
  }
#endif
  void expand(const Stored *source, float *destination, size_t count) const {
    size_t i = 0;
#if defined(__aarch64__)
    const auto offset = vdupq_n_s16(int16_t(zero));
    const auto factor = vdupq_n_f32(scale);
    for (; i + 8 <= count; i += 8) {
      const auto values = load(source + i, offset);
      vst1q_f32(
          destination + i,
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(values))), factor));
      vst1q_f32(
          destination + i + 4,
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(values))), factor));
    }
#endif
    for (; i < count; ++i)
      destination[i] = float(int(source[i]) - (Centered ? 0 : zero)) * scale;
  }

public:
  template <class Weight>
  CompactHead(Weight value, float scale, int32_t zero)
      : storage(width * depth), weights(storage.data()), scale(scale),
        zero(zero) {
    if (MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, columns, panel) !=
        columns * panel * sizeof(float))
      throw std::runtime_error("unexpected pinned compact head packing");
    alignas(64) std::array<float, panel * columns> tile, packed;
    for (size_t c = 0; c < width; c += columns)
      for (size_t k = 0; k < depth; k += panel) {
        for (size_t j = 0; j < panel; ++j)
          for (size_t i = 0; i < columns; ++i)
            tile[j * columns + i] = float(value(k + j, c + i));
        MlasGemmPackB(CblasNoTrans, CblasNoTrans, columns, panel, tile.data(),
                      columns, packed.data());
        for (size_t i = 0; i < packed.size(); ++i)
          storage[c * depth + k * columns + i] =
              Stored(int(packed[i]) - (Centered ? zero : 0));
      }
  }

  CompactHead(const Stored *packed, float scale, int32_t zero)
      : weights(packed), scale(scale), zero(zero) {
    if (!packed)
      throw std::invalid_argument("null packed head");
  }
  CompactHead(const CompactHead &) = delete;
  CompactHead &operator=(const CompactHead &) = delete;

  void run(const float *input, size_t rows, float *output) const {
    run_columns(input, rows, output, width, 0, width);
  }

  void run_columns(const float *input, size_t rows, float *output,
                   size_t stride, size_t begin, size_t end) const {
    if (!input || !output || !rows || rows > 1024 || begin >= end ||
        end > width || begin % columns || end % columns || stride < end - begin)
      throw std::invalid_argument("compact head extent");
#if defined(__aarch64__)
    if (rows == 1) {
      const auto offset = vdupq_n_s16(int16_t(zero));
      const auto factor = vdupq_n_f32(scale);
      for (size_t c = begin; c < end; c += columns) {
        float32x4_t sums[4][4];
        for (auto &group : sums)
          for (auto &sum : group)
            sum = vdupq_n_f32(0);
        for (size_t k = 0; k < depth; k += panel) {
          const Stored *w = weights + c * depth + k * columns;
          for (size_t j = 0; j < panel; ++j) {
            const auto a = vdupq_n_f32(input[k + j]);
            for (size_t group = 0; group < 4; ++group)
              for (size_t pair = 0; pair < 2; ++pair) {
                const auto values =
                    load(w + group * panel * 16 + j * 16 + pair * 8, offset);
                const auto low = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(values))), factor);
                const auto high = vmulq_f32(
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(values))), factor);
                sums[group][pair * 2] =
                    vfmaq_f32(sums[group][pair * 2], low, a);
                sums[group][pair * 2 + 1] =
                    vfmaq_f32(sums[group][pair * 2 + 1], high, a);
              }
          }
        }
        for (size_t group = 0; group < 4; ++group)
          for (size_t lane = 0; lane < 4; ++lane)
            vst1q_f32(output + c - begin + group * 16 + lane * 4,
                      sums[group][lane]);
      }
      return;
    }
#endif
    alignas(64) std::array<float, panel * columns> tile;
    for (size_t c = begin; c < end; c += columns)
      for (size_t k = 0; k < depth; k += panel) {
        expand(weights + c * depth + k * columns, tile.data(), tile.size());
        MlasGemm(CblasNoTrans, rows, columns, panel, 1.f, input + k, depth,
                 tile.data(), k == 0 ? 0.f : 1.f, output + c - begin, stride,
                 nullptr);
      }
  }
};
} // namespace transcribro::standalone
