#pragma once
#include "mlas.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace transcribro {
// Baseline x86 CPUs lack the selected projection kernel's AVX512/VNNI path.
// Reconstruct original unsigned codes once, then let pinned MLAS dispatch
// its exact integer GEMM. The archive and final floating-point arithmetic
// remain identical to Projection::multiply; no new quantization occurs.
class X86Projection {
public:
  X86Projection(size_t k, size_t n, const int8_t *weights,
                const float *scales, const uint8_t *zeros, const float *bias)
      : k_(k), n_(n), scales_(scales), zeros_(zeros), bias_(bias) {
    std::vector<uint8_t> original(k * n);
    for (size_t c = 0; c < n; ++c)
      for (size_t j = 0; j < k; ++j) {
        const size_t offset = (c / 16) * k * 16 + (j / 8) * 128 +
                              (c % 16 / 2) * 16 + (c % 2) * 8 + j % 8;
        original[j * n + c] = uint8_t(int32_t(weights[offset]) + 128);
      }
    const size_t bytes = MlasGemmPackBSize(n, k, false, false);
    packed_ = bytes != 0;
    if (packed_) {
      weights_.resize((bytes + 7) / 8);
      MlasGemmPackB(n, k, original.data(), n, false, false, weights_.data());
    } else {
      weights_.resize((original.size() + 7) / 8);
      std::memcpy(weights_.data(), original.data(), original.size());
    }
  }

  template <typename Quantized>
  void multiply(const Quantized &q, size_t rows, float *out,
                std::vector<uint8_t> &activations,
                std::vector<int32_t> &accumulators) const {
    if (!rows || rows > 1024 || q.tiled || q.values.size() != rows * k_ ||
        q.centered_sums.size() != rows)
      throw std::invalid_argument("x86 projection input shape");
    activations.resize(rows * k_);
    for (size_t i = 0; i < activations.size(); ++i)
      activations[i] = uint8_t(int32_t(q.values[i]) + 128);
    accumulators.resize(rows * n_);
    MLAS_GEMM_QUANT_SHAPE_PARAMS shape;
    shape.M = rows;
    shape.N = n_;
    shape.K = k_;
    MLAS_GEMM_QUANT_DATA_PARAMS data;
    data.A = activations.data();
    data.lda = k_;
    data.ZeroPointA = uint8_t(q.zero);
    data.B = weights_.data();
    data.ldb = n_;
    data.ZeroPointB = zeros_;
    data.BIsPacked = packed_;
    data.PerColumnZeroPoints = true;
    data.C = accumulators.data();
    data.ldc = n_;
    MlasGemm(shape, data, nullptr);
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < n_; ++c)
        out[r * n_ + c] = float(accumulators[r * n_ + c]) *
                             (q.scale * scales_[c]) + bias_[c];
  }

private:
  size_t k_, n_;
  const float *scales_;
  const uint8_t *zeros_;
  const float *bias_;
  bool packed_;
  std::vector<uint64_t> weights_;
};
} // namespace transcribro
