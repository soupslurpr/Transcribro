#pragma once
#include "projection.h"
#include <cstring>
#include <fstream>
#include <string>

namespace transcribro::research {
template <class T> std::vector<T> read_values(std::ifstream &f, size_t count) {
  std::vector<T> values(count);
  if (!f.read(reinterpret_cast<char *>(values.data()), count * sizeof(T)))
    throw std::runtime_error("truncated fixture");
  return values;
}
struct Fixture {
  size_t k, n, cases;
  std::vector<uint8_t> weights, zeros;
  std::vector<float> scales, bias, inputs;
  explicit Fixture(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    const auto magic = read_values<char>(file, 8);
    if (std::memcmp(magic.data(), "TRPROJ1\0", 8))
      throw std::runtime_error("fixture magic");
    const auto header = read_values<uint32_t>(file, 4);
    k = header[0];
    n = header[1];
    cases = header[2];
    if (!k || !n || !cases || k > 16384 || n > 32768 || cases > 1024 ||
        header[3])
      throw std::runtime_error("fixture dimensions");
    weights = read_values<uint8_t>(file, k * n);
    scales = read_values<float>(file, n);
    zeros = read_values<uint8_t>(file, n);
    bias = read_values<float>(file, n);
    inputs = read_values<float>(file, cases * k);
    if (file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("trailing fixture bytes");
  }
  Projection projection() const {
    return Projection(k, n, weights.data(), scales.data(), zeros.data(),
                      bias.data());
  }
};

class FeedForward {
public:
  FeedForward(const Fixture &first, const Fixture &second)
      : in_(first.projection()), out_(second.projection()) {
    if (first.n != 2 * second.k || first.k != second.n)
      throw std::runtime_error("feed-forward dimensions");
  }
  void run(const float *input, size_t rows, float *output) {
    expanded.resize(rows * in_.output_size());
    gated.resize(rows * out_.input_size());
    quantize(input, rows, in_.input_size(), input_q_);
    in_.multiply(input_q_, rows, expanded.data());
    const size_t width = out_.input_size();
    for (size_t r = 0; r < rows; ++r)
      for (size_t i = 0; i < width; ++i) {
        const float x = expanded[r * 2 * width + width + i];
        const float silu = x / (1.0f + std::exp(-x));
        gated[r * width + i] = expanded[r * 2 * width + i] * silu;
      }
    quantize(gated.data(), rows, width, hidden_q_);
    out_.multiply(hidden_q_, rows, output);
  }
  std::vector<float> expanded, gated;

private:
  Projection in_, out_;
  Quantized input_q_, hidden_q_;
};
} // namespace transcribro::research
