#include "projection.h"
#if defined(__x86_64__)
#include "projection_x86.h"
#endif
#include <cstring>
#include <iostream>
#include <limits>
using namespace transcribro::research;
static uint32_t state = 137;
static uint32_t random_value() {
  state = 1664525 * state + 1013904223;
  return state;
}
static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid input accepted");
}
int main() try {
  size_t checked = 0;
  for (auto shape :
       {std::pair<size_t, size_t>{16, 16}, {32, 48}, {64, 80}, {640, 16}}) {
    const auto [k, n] = shape;
    std::vector<uint8_t> w(k * n), z(n);
    std::vector<float> s(n), b(n);
    for (auto &x : w)
      x = uint8_t(random_value() >> 16);
    for (size_t i = 0; i < n; ++i) {
      z[i] = uint8_t(random_value() >> 16);
      s[i] = float((random_value() % 1000) + 1) / 8192;
      b[i] = float(int(random_value() % 100) - 50) / 16;
    }
    for (size_t layout : {size_t(4), size_t(8)}) {
      std::vector<int8_t> packed(k * n);
      std::vector<int32_t> sums(n, 0);
      for (size_t c = 0; c < n; ++c)
        for (size_t j = 0; j < k; ++j) {
          const size_t offset =
              layout == 4
                  ? (c / 16) * k * 16 + (j / 4) * 64 + (c % 16) * 4 + j % 4
                  : (c / 16) * k * 16 + (j / 8) * 128 + (c % 16 / 2) * 16 +
                        (c % 2) * 8 + j % 8;
          packed[offset] = int8_t(int(w[j * n + c]) - 128);
          sums[c] += int(w[j * n + c]) - 128;
        }
      Projection owned(k, n, w.data(), s.data(), z.data(), b.data(), layout);
      Projection borrowed(k, n, packed.data(), sums.data(), s.data(), z.data(),
                          b.data(), layout);
      for (size_t rows :
           {size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(7),
            size_t(8), size_t(9), size_t(16), size_t(31), size_t(37),
            size_t(63), size_t(128), size_t(257), size_t(1023), size_t(1024)}) {
        std::vector<float> input(rows * k), actual(rows * n), other(rows * n);
        for (auto &v : input)
          v = float(int(random_value() % 200001) - 100000) / 4096;
        Quantized q;
        quantize(input.data(), rows, k, q);
        std::vector<uint8_t> original_bytes(q.values.size());
        for (size_t i = 0; i < original_bytes.size(); ++i)
          original_bytes[i] = uint8_t(int32_t(q.values[i]) + 128);
        Quantized imported;
        import_quantized(original_bytes.data(), rows, k, q.scale, q.zero,
                         imported);
        require(imported.values == q.values &&
                    imported.centered_sums == q.centered_sums &&
                    imported.scale == q.scale && imported.zero == q.zero,
                "imported activations differ");
        owned.multiply(q, rows, actual.data());
        borrowed.multiply(imported, rows, other.data());
        require(std::memcmp(actual.data(), other.data(),
                            actual.size() * sizeof(float)) == 0,
                "borrowed/owned differ");
#if defined(__x86_64__)
        if (layout == 8) {
          transcribro::X86Projection baseline(k, n, packed.data(), s.data(), z.data(), b.data());
          std::vector<uint8_t> activations;
          std::vector<int32_t> accumulators;
          baseline.multiply(q, rows, other.data(), activations, accumulators);
          require(std::memcmp(actual.data(), other.data(), actual.size() * sizeof(float)) == 0,
                  "MLAS baseline projection differs");
        }
#endif
        Quantized tiled;
        quantize(input.data(), rows, k, tiled, true);
        require(tiled.scale == q.scale && tiled.zero == q.zero &&
                    tiled.centered_sums == q.centered_sums,
                "tiled quantization metadata differs");
        for (size_t r = 0; r < rows; ++r)
          for (size_t j = 0; j < k; ++j) {
            const int value =
                tiled.tiled
                    ? int(uint8_t(tiled.values[activation_offset(r, j, k)]))
                    : int(tiled.values[r * k + j]) + 128;
            require(value == int(q.values[r * k + j]) + 128,
                    "tiled quantization value differs");
          }
        borrowed.multiply(tiled, rows, other.data());
        require(std::memcmp(actual.data(), other.data(),
                            actual.size() * sizeof(float)) == 0,
                "tiled projection differs");
        for (size_t r = 0; r < rows; ++r)
          for (size_t c = 0; c < n; ++c) {
            int32_t total = 0;
            for (size_t j = 0; j < k; ++j)
              total += (int(q.values[r * k + j]) + 128 - q.zero) *
                       (int(w[j * n + c]) - int(z[c]));
            const float expected = float(total) * (q.scale * s[c]) + b[c];
            require(std::memcmp(&expected, &actual[r * n + c], sizeof(float)) ==
                        0,
                    "independent matrix result differs");
            ++checked;
          }
      }
    }
  }
  Quantized q;
  std::vector<uint8_t> bytes(32, 128);
  for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
    rejects([&] { import_quantized(bytes.data(), 1, 32, scale, 128, q); });
  rejects([&] { import_quantized(bytes.data(), 1, 32, 1.0f, -1, q); });
  rejects([&] { import_quantized(bytes.data(), 1, 32, 1.0f, 256, q); });
  rejects([&] { import_quantized(bytes.data(), 0, 32, 1.0f, 128, q); });
  std::vector<float> input(32, 1);
  rejects([&] { quantize(input.data(), 0, 16, q); });
  rejects([&] { quantize(input.data(), 1, 16385, q); });
  for (size_t index : {size_t(0), size_t(7), size_t(31)}) {
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
      input[index] = value;
      rejects([&] { quantize(input.data(), 1, input.size(), q); });
      input[index] = 1;
    }
  }
  // A real decoder boundary: division gives 39.5, reciprocal multiplication
  // gives 39.499996. The latter changed a completed transcript in EdAcc.
  input.assign(32, 0);
  input[0] = -6.404613f;
  input[1] = 5.8952546f;
  input[2] = 1.9052736f;
  quantize(input.data(), 1, input.size(), q);
  require(q.zero == 133 && int(q.values[2]) + 128 == 173,
          "division rounding regression");
  std::cout << "independent matrix values checked: " << checked
            << "; invalid inputs rejected\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
