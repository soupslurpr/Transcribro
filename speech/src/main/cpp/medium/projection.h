// Experimental execution of unchanged uint8 affine Medium projections.
// No model requantization. Build without fast-math and FP contraction.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <stdexcept>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

namespace transcribro::research {

struct Quantized {
  std::pmr::vector<int8_t> values;
  std::pmr::vector<int32_t> centered_sums;
  float scale = 1;
  int32_t zero = 0;
  bool tiled = false;
  explicit Quantized(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : values(memory), centered_sums(memory) {}
};

// Preserve ORT's existing DynamicQuantizeLinear output for shared Q/K/V
// inputs. Only change its unsigned representation to the kernel's layout.
inline void import_quantized(const uint8_t *x, size_t rows, size_t k,
                             float scale, int32_t zero, Quantized &q) {
  if (!rows || rows > 1024 || !k || k > 16384)
    throw std::invalid_argument("unsupported quantized activation dimensions");
  if (!std::isfinite(scale) || scale <= 0 || zero < 0 || zero > 255)
    throw std::invalid_argument("invalid quantized activation metadata");
  q.scale = scale;
  q.zero = zero;
  q.tiled = false;
  q.values.resize(rows * k);
  q.centered_sums.resize(rows);
  for (size_t r = 0; r < rows; ++r) {
    int32_t sum = 0;
    size_t j = 0;
#if defined(__aarch64__)
    auto sums = vdupq_n_u32(0);
    for (; j + 16 <= k; j += 16) {
      const auto bytes = vld1q_u8(x + r * k + j);
      vst1q_s8(q.values.data() + r * k + j,
               vreinterpretq_s8_u8(veorq_u8(bytes, vdupq_n_u8(128))));
      sums = vpadalq_u16(sums, vpaddlq_u8(bytes));
    }
    sum = int32_t(vaddvq_u32(sums));
#endif
    for (; j < k; ++j) {
      const auto value = x[r * k + j];
      q.values[r * k + j] = int8_t(int32_t(value) - 128);
      sum += value;
    }
    q.centered_sums[r] = sum - int32_t(k) * zero;
  }
}

inline size_t activation_offset(size_t r, size_t j, size_t k) {
  return (r / 8) * k * 8 + (j / 8) * 64 + (r % 8 / 2) * 16 + (r % 2) * 8 +
         j % 8;
}

inline bool tiled_activations_available() {
#if defined(__aarch64__)
  return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
  return false;
#endif
}

inline void quantize(const float *x, size_t rows, size_t k, Quantized &q,
                     bool tiled = false) {
  if (!rows || rows > 1024 || !k || k > 16384)
    throw std::invalid_argument("unsupported activation dimensions");
  const size_t count = rows * k;
  float low = 0, high = 0;
  size_t first = 0;
#if defined(__aarch64__)
  auto lows = vdupq_n_f32(0), highs = vdupq_n_f32(0);
  for (; first + 4 <= count; first += 4) {
    const auto value = vld1q_f32(x + first);
    lows = vminq_f32(lows, value);
    highs = vmaxq_f32(highs, value);
  }
  low = vminvq_f32(lows);
  high = vmaxvq_f32(highs);
  if (!std::isfinite(low) || !std::isfinite(high))
    throw std::invalid_argument("nonfinite activation");
#endif
  for (size_t i = first; i < count; ++i) {
    if (!std::isfinite(x[i]))
      throw std::invalid_argument("nonfinite activation");
    low = std::min(low, x[i]);
    high = std::max(high, x[i]);
  }
  q.scale = (high - low) / 255.0f;
  if (q.scale == 0)
    q.scale = 1;
  if (!std::isfinite(q.scale) || q.scale <= 0)
    throw std::invalid_argument("unrepresentable activation range");
  q.zero = int32_t(std::clamp(std::nearbyint(-low / q.scale), 0.0f, 255.0f));
  // Division is part of the reference arithmetic. Multiplying by a rounded
  // reciprocal can move an exact half across a quantization boundary.
  q.tiled = tiled && rows >= 8 && k % 16 == 0;
  q.values.resize(q.tiled ? (rows + 7) / 8 * 8 * k : count);
  if (q.tiled && rows % 8)
    std::fill(q.values.begin() + (rows / 8) * 8 * k, q.values.end(),
              int8_t(q.zero));
  q.centered_sums.assign(rows, 0);
  for (size_t r = 0; r < rows; ++r) {
    int32_t sum = 0;
    size_t j = 0;
#if defined(__aarch64__)
    auto sums = vdupq_n_u32(0);
    for (; j + 16 <= k; j += 16) {
      uint16x4_t chunks[4];
      for (size_t group = 0; group < 4; ++group) {
        auto value = vdivq_f32(vld1q_f32(x + r * k + j + group * 4),
                               vdupq_n_f32(q.scale));
        value = vaddq_f32(vrndnq_f32(value), vdupq_n_f32(float(q.zero)));
        value = vminq_f32(vmaxq_f32(value, vdupq_n_f32(0)), vdupq_n_f32(255));
        chunks[group] = vmovn_u32(vcvtq_u32_f32(value));
      }
      const auto first8 = vmovn_u16(vcombine_u16(chunks[0], chunks[1]));
      const auto next8 = vmovn_u16(vcombine_u16(chunks[2], chunks[3]));
      const auto bytes = vcombine_u8(first8, next8);
      if (q.tiled) {
        auto *destination = reinterpret_cast<uint8_t *>(q.values.data()) +
                            activation_offset(r, j, k);
        vst1_u8(destination, first8);
        vst1_u8(destination + 64, next8);
      } else {
        vst1q_s8(q.values.data() + r * k + j,
                 vreinterpretq_s8_u8(veorq_u8(bytes, vdupq_n_u8(128))));
      }
      sums = vpadalq_u16(sums, vpaddlq_u8(bytes));
    }
    sum = int32_t(vaddvq_u32(sums)) - int32_t(j) * q.zero;
#endif
    for (; j < k; ++j) {
      const int32_t a = int32_t(
          std::clamp(std::nearbyint(x[r * k + j] / q.scale) + float(q.zero),
                     0.0f, 255.0f));
      if (q.tiled)
        q.values[activation_offset(r, j, k)] = int8_t(a);
      else
        q.values[r * k + j] = int8_t(a - 128);
      sum += a - q.zero;
    }
    q.centered_sums[r] = sum;
  }
}

class Projection {
public:
  // Matrix shape is [K,N]. Tiles keep four K values for each of 16 outputs.
  // Per-output affine offsets are retained, including non-symmetric weights.
  Projection(size_t k, size_t n, const uint8_t *weights, const float *scales,
             const uint8_t *zeros, const float *bias, size_t layout = 4)
      : k_(k), n_(n), layout_(layout) {
    if (!k || !n || k > 16384 || n > 32768 || k % 16 || n % 16)
      throw std::invalid_argument(
          "projection requires bounded 16-aligned dimensions");
    weights_.resize(k * n);
    sums_.assign(n, 0);
    zeros_.resize(n);
    scales_.assign(scales, scales + n);
    bias_.assign(bias, bias + n);
    for (size_t c = 0; c < n; ++c) {
      if (!std::isfinite(scales[c]) || scales[c] <= 0 ||
          !std::isfinite(bias[c]))
        throw std::invalid_argument("invalid projection scale or bias");
      zeros_[c] = int32_t(zeros[c]) - 128;
    }
    size_t offset = 0;
    if (layout != 4 && layout != 8)
      throw std::invalid_argument("projection layout");
    if (layout == 4) {
      for (size_t c = 0; c < n; c += 16)
        for (size_t j = 0; j < k; j += 4)
          for (size_t lane = 0; lane < 16; ++lane)
            for (size_t inner = 0; inner < 4; ++inner) {
              const int32_t w =
                  int32_t(weights[(j + inner) * n + c + lane]) - 128;
              weights_[offset++] = int8_t(w);
              sums_[c + lane] += w;
            }
    } else {
      for (size_t c = 0; c < n; c += 16)
        for (size_t j = 0; j < k; j += 8)
          for (size_t pair = 0; pair < 16; pair += 2)
            for (size_t lane = 0; lane < 2; ++lane)
              for (size_t a = 0; a < 8; ++a) {
                const int32_t w =
                    int32_t(weights[(j + a) * n + c + pair + lane]) - 128;
                weights_[offset++] = int8_t(w);
                sums_[c + pair + lane] += w;
              }
    }
    weights_data_ = weights_.data();
    sums_data_ = sums_.data();
    scales_data_ = scales_.data();
    bias_data_ = bias_.data();
  }

  // Borrows immutable, already packed session initializers. The session owns
  // them until after kernel destruction. Bias/zero metadata is small and owned.
  Projection(size_t k, size_t n, const int8_t *packed, const int32_t *sums,
             const float *scales, const uint8_t *zeros, const float *bias,
             size_t layout = 4)
      : k_(k), n_(n), layout_(layout), weights_data_(packed), sums_data_(sums),
        scales_data_(scales) {
    if (!k || !n || k > 16384 || n > 32768 || k % 16 || n % 16)
      throw std::invalid_argument("packed projection dimensions");
    if (layout != 4 && layout != 8)
      throw std::invalid_argument("packed projection layout");
    zeros_.resize(n);
    bias_.assign(bias, bias + n);
    bias_data_ = bias_.data();
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(scales[i]) || scales[i] <= 0 ||
          !std::isfinite(bias[i]))
        throw std::invalid_argument("packed projection metadata");
      zeros_[i] = int32_t(zeros[i]) - 128;
    }
  }
  Projection(const Projection &) = delete;
  Projection &operator=(const Projection &) = delete;
  Projection(Projection &&) = default;
  Projection &operator=(Projection &&) = default;

  size_t input_size() const { return k_; }
  size_t output_size() const { return n_; }
  size_t storage_bytes() const { return k_ * n_ + n_ * 16; }

  void multiply(const Quantized &q, size_t rows, float *out,
                bool scalar = false) const {
    if (!rows || rows > 1024 ||
        q.values.size() != (q.tiled ? (rows + 7) / 8 * 8 : rows) * k_ ||
        q.centered_sums.size() != rows)
      throw std::invalid_argument("quantized input shape mismatch");
#if defined(__aarch64__)
    if (!scalar && q.tiled && layout_ == 8 && tiled_activations_available()) {
      i8mm8<true>(q, rows, out);
      return;
    }
    if (!scalar && !q.tiled && (getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) {
      if (layout_ == 8) {
        if (rows > 1 && (getauxval(AT_HWCAP2) & HWCAP2_I8MM)) {
          if (rows >= 8)
            i8mm8<false>(q, rows, out);
          else
            i8mm(q, rows, out);
        } else
          dotprod8(q, rows, out);
      } else
        dotprod(q, rows, out);
      return;
    }
#else
    (void)scalar;
#endif
#if defined(__x86_64__)
    if (!scalar && !q.tiled && __builtin_cpu_supports("avx512vnni") &&
        __builtin_cpu_supports("avx512vbmi")) {
      vnni(q, rows, out);
      return;
    }
#endif
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < n_; ++c) {
        int32_t acc = 0;
        for (size_t j = 0; j < k_; ++j) {
          const size_t offset =
              layout_ == 4
                  ? (c / 16) * k_ * 16 + (j / 4) * 64 + (c % 16) * 4 + j % 4
                  : (c / 16) * k_ * 16 + (j / 8) * 128 + (c % 16 / 2) * 16 +
                        (c % 2) * 8 + j % 8;
          const int32_t a =
              q.tiled
                  ? int32_t(uint8_t(q.values[activation_offset(r, j, k_)])) -
                        128
                  : int32_t(q.values[r * k_ + j]);
          acc += a * int32_t(weights_data_[offset]);
        }
        acc += (128 - q.zero) * sums_data_[c] - q.centered_sums[r] * zeros_[c];
        out[r * n_ + c] =
            float(acc) * (q.scale * scales_data_[c]) + bias_data_[c];
      }
  }

private:
  size_t k_, n_, layout_ = 4;
  std::vector<int8_t> weights_;
  std::vector<int32_t> sums_, zeros_;
  std::vector<float> scales_, bias_;
  const int8_t *weights_data_ = nullptr;
  const int32_t *sums_data_ = nullptr;
  const float *scales_data_ = nullptr;
  const float *bias_data_ = nullptr;

#if defined(__x86_64__)
  __attribute__((target("avx512f,avx512bw,avx512vnni,avx512vbmi"))) void
  vnni(const Quantized &q, size_t rows, float *out) const {
    alignas(64) uint8_t indices[64];
    for (size_t c = 0; c < 16; ++c)
      for (size_t j = 0; j < 4; ++j)
        indices[c * 4 + j] = uint8_t((c / 2) * 16 + (c % 2) * 8 + j);
    const auto lo_indices = _mm512_load_si512(indices);
    const auto hi_indices = _mm512_add_epi8(lo_indices, _mm512_set1_epi8(4));
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < n_; c += 16) {
        auto sum = _mm512_setzero_si512();
        const int8_t *w = weights_data_ + c * k_;
        for (size_t j = 0; j < k_; j += layout_) {
          uint32_t word;
          std::memcpy(&word, q.values.data() + r * k_ + j, 4);
          word ^= 0x80808080u;
          const auto a = _mm512_set1_epi32(int32_t(word));
          if (layout_ == 4)
            sum = _mm512_dpbusd_epi32(sum, a, _mm512_loadu_si512(w + j * 16));
          else {
            const auto first = _mm512_loadu_si512(w + j * 16),
                       second = _mm512_loadu_si512(w + j * 16 + 64);
            sum = _mm512_dpbusd_epi32(
                sum, a, _mm512_permutex2var_epi8(first, lo_indices, second));
            std::memcpy(&word, q.values.data() + r * k_ + j + 4, 4);
            word ^= 0x80808080u;
            sum = _mm512_dpbusd_epi32(
                sum, _mm512_set1_epi32(int32_t(word)),
                _mm512_permutex2var_epi8(first, hi_indices, second));
          }
        }
        sum = _mm512_sub_epi32(
            sum, _mm512_mullo_epi32(_mm512_loadu_si512(sums_data_ + c),
                                    _mm512_set1_epi32(q.zero)));
        sum = _mm512_sub_epi32(
            sum, _mm512_mullo_epi32(_mm512_loadu_si512(zeros_.data() + c),
                                    _mm512_set1_epi32(q.centered_sums[r])));
        const auto scales = _mm512_mul_ps(_mm512_loadu_ps(scales_data_ + c),
                                          _mm512_set1_ps(q.scale));
        const auto values =
            _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(sum), scales),
                          _mm512_loadu_ps(bias_data_ + c));
        _mm512_storeu_ps(out + r * n_ + c, values);
      }
  }
#endif

#if defined(__aarch64__)
  __attribute__((target("dotprod"))) void
  dotprod8(const Quantized &q, size_t rows, float *out) const {
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < n_; c += 16) {
        int32x4_t acc[8];
        for (auto &a : acc)
          a = vdupq_n_s32(0);
        const auto *w = weights_data_ + c * k_;
        for (size_t j = 0; j < k_; j += 8) {
          const auto a8 = vld1_s8(q.values.data() + r * k_ + j);
          const auto a = vcombine_s8(a8, a8);
          for (size_t pair = 0; pair < 8; ++pair)
            acc[pair] =
                vdotq_s32(acc[pair], vld1q_s8(w + j * 16 + pair * 16), a);
        }
        for (size_t pair = 0; pair < 8; ++pair) {
          const size_t col = c + pair * 2;
          auto sum =
              vpadd_s32(vget_low_s32(acc[pair]), vget_high_s32(acc[pair]));
          sum = vadd_s32(sum,
                         vmul_n_s32(vld1_s32(sums_data_ + col), 128 - q.zero));
          sum = vsub_s32(sum, vmul_n_s32(vld1_s32(zeros_.data() + col),
                                         q.centered_sums[r]));
          const auto s = vmul_n_f32(vld1_f32(scales_data_ + col), q.scale);
          vst1_f32(out + r * n_ + col, vadd_f32(vmul_f32(vcvt_f32_s32(sum), s),
                                                vld1_f32(bias_data_ + col)));
        }
      }
  }

  __attribute__((target("i8mm"))) void i8mm(const Quantized &q, size_t rows,
                                            float *out,
                                            size_t first_row = 0) const {
    const auto offset = vdupq_n_u8(128);
    for (size_t r = first_row; r < rows; r += 4)
      for (size_t c = 0; c < n_; c += 16) {
        int32x4_t acc[2][8];
        for (auto &group : acc)
          for (auto &a : group)
            a = vdupq_n_s32(0);
        for (size_t j = 0; j < k_; j += 8) {
          const auto pad = vdup_n_s8(int8_t(q.zero - 128));
          const auto a0 = vld1_s8(q.values.data() + r * k_ + j);
          const auto a1 =
              r + 1 < rows ? vld1_s8(q.values.data() + (r + 1) * k_ + j) : pad;
          const auto a2 =
              r + 2 < rows ? vld1_s8(q.values.data() + (r + 2) * k_ + j) : pad;
          const auto a3 =
              r + 3 < rows ? vld1_s8(q.values.data() + (r + 3) * k_ + j) : pad;
          const auto first =
              veorq_u8(vreinterpretq_u8_s8(vcombine_s8(a0, a1)), offset);
          const auto second =
              veorq_u8(vreinterpretq_u8_s8(vcombine_s8(a2, a3)), offset);
          for (size_t pair = 0; pair < 8; ++pair) {
            const auto w =
                vld1q_s8(weights_data_ + c * k_ + j * 16 + pair * 16);
            acc[0][pair] = vusmmlaq_s32(acc[0][pair], first, w);
            if (r + 2 < rows)
              acc[1][pair] = vusmmlaq_s32(acc[1][pair], second, w);
          }
        }
        for (size_t group = 0; group < 2 && r + group * 2 < rows; ++group) {
          const size_t row = r + group * 2;
          const auto row_sums = vcombine_s32(
              vdup_n_s32(q.centered_sums[row]),
              vdup_n_s32(row + 1 < rows ? q.centered_sums[row + 1] : 0));
          for (size_t pair = 0; pair < 8; ++pair) {
            const size_t col = c + pair * 2;
            const auto wsum = vld1_s32(sums_data_ + col),
                       wzero = vld1_s32(zeros_.data() + col);
            auto sum = vsubq_s32(acc[group][pair],
                                 vmulq_n_s32(vcombine_s32(wsum, wsum), q.zero));
            sum =
                vsubq_s32(sum, vmulq_s32(vcombine_s32(wzero, wzero), row_sums));
            const auto s = vmul_n_f32(vld1_f32(scales_data_ + col), q.scale),
                       b = vld1_f32(bias_data_ + col);
            const auto values =
                vaddq_f32(vmulq_f32(vcvtq_f32_s32(sum), vcombine_f32(s, s)),
                          vcombine_f32(b, b));
            vst1_f32(out + row * n_ + col, vget_low_f32(values));
            if (row + 1 < rows)
              vst1_f32(out + (row + 1) * n_ + col, vget_high_f32(values));
          }
        }
      }
  }

  template <bool Tiled>
  __attribute__((target("i8mm"))) void i8mm8(const Quantized &q, size_t rows,
                                             float *out) const {
    const size_t full_rows = Tiled ? (rows + 7) / 8 * 8 : rows / 8 * 8;
    const auto offset = vdupq_n_u8(128);
    for (size_t r = 0; r < full_rows; r += 8)
      for (size_t c = 0; c < n_; c += 8) {
        int32x4_t acc[4][4];
        for (auto &group : acc)
          for (auto &a : group)
            a = vdupq_n_s32(0);
        for (size_t j = 0; j < k_; j += 8) {
          uint8x16_t a[4];
          for (size_t group = 0; group < 4; ++group) {
            const size_t row = r + group * 2;
            if constexpr (Tiled)
              a[group] =
                  vld1q_u8(reinterpret_cast<const uint8_t *>(q.values.data()) +
                           r * k_ + j * 8 + group * 16);
            else
              a[group] =
                  veorq_u8(vreinterpretq_u8_s8(vcombine_s8(
                               vld1_s8(q.values.data() + row * k_ + j),
                               vld1_s8(q.values.data() + (row + 1) * k_ + j))),
                           offset);
          }
          for (size_t pair = 0; pair < 4; ++pair) {
            const auto w = vld1q_s8(weights_data_ + (c / 16) * k_ * 16 +
                                    j * 16 + (c % 16) * 8 + pair * 16);
            for (size_t group = 0; group < 4; ++group)
              acc[group][pair] = vusmmlaq_s32(acc[group][pair], a[group], w);
          }
        }
        for (size_t group = 0; group < 4 && r + group * 2 < rows; ++group) {
          const size_t row = r + group * 2;
          const auto row_sums = vcombine_s32(
              vdup_n_s32(q.centered_sums[row]),
              vdup_n_s32(row + 1 < rows ? q.centered_sums[row + 1] : 0));
          for (size_t pair = 0; pair < 4; ++pair) {
            const size_t col = c + pair * 2;
            const auto wsum = vld1_s32(sums_data_ + col),
                       wzero = vld1_s32(zeros_.data() + col);
            auto sum = vsubq_s32(acc[group][pair],
                                 vmulq_n_s32(vcombine_s32(wsum, wsum), q.zero));
            sum =
                vsubq_s32(sum, vmulq_s32(vcombine_s32(wzero, wzero), row_sums));
            const auto s = vmul_n_f32(vld1_f32(scales_data_ + col), q.scale),
                       b = vld1_f32(bias_data_ + col);
            const auto values =
                vaddq_f32(vmulq_f32(vcvtq_f32_s32(sum), vcombine_f32(s, s)),
                          vcombine_f32(b, b));
            vst1_f32(out + row * n_ + col, vget_low_f32(values));
            if (row + 1 < rows)
              vst1_f32(out + (row + 1) * n_ + col, vget_high_f32(values));
          }
        }
      }
    if (full_rows < rows)
      i8mm(q, rows, out, full_rows);
  }

  __attribute__((target("dotprod"))) void
  dotprod(const Quantized &q, size_t rows, float *out) const {
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < n_; c += 16) {
        auto s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0;
        const int8_t *w = weights_data_ + c * k_;
        for (size_t j = 0; j < k_; j += 16) {
          const auto a = vld1q_s8(q.values.data() + r * k_ + j);
#define TRANSCRIBRO_DOT_LANE(lane)                                             \
  s0 = vdotq_laneq_s32(s0, vld1q_s8(w), a, lane);                              \
  s1 = vdotq_laneq_s32(s1, vld1q_s8(w + 16), a, lane);                         \
  s2 = vdotq_laneq_s32(s2, vld1q_s8(w + 32), a, lane);                         \
  s3 = vdotq_laneq_s32(s3, vld1q_s8(w + 48), a, lane);                         \
  w += 64
          TRANSCRIBRO_DOT_LANE(0);
          TRANSCRIBRO_DOT_LANE(1);
          TRANSCRIBRO_DOT_LANE(2);
          TRANSCRIBRO_DOT_LANE(3);
#undef TRANSCRIBRO_DOT_LANE
        }
        const int32x4_t sums[] = {s0, s1, s2, s3};
        for (size_t group = 0; group < 4; ++group) {
          const size_t offset = c + group * 4;
          auto corrected =
              vaddq_s32(sums[group], vmulq_n_s32(vld1q_s32(sums_data_ + offset),
                                                 128 - q.zero));
          corrected = vsubq_s32(corrected,
                                vmulq_n_s32(vld1q_s32(zeros_.data() + offset),
                                            q.centered_sums[r]));
          const auto scales =
              vmulq_n_f32(vld1q_f32(scales_data_ + offset), q.scale);
          const auto result =
              vaddq_f32(vmulq_f32(vcvtq_f32_s32(corrected), scales),
                        vld1q_f32(bias_data_ + offset));
          vst1q_f32(out + r * n_ + offset, result);
        }
      }
  }
#endif
};
} // namespace transcribro::research
