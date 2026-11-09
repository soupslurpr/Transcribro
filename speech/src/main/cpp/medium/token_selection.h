#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace transcribro::standalone {
// Consume vocabulary columns in order, retaining the first maximum exactly
// as std::max_element does. Scores are computed by the unchanged head kernels.
class TokenSelection {
  std::vector<float> scores, best;

public:
  template <class Compute>
  void run(size_t rows, size_t width, size_t columns, Compute compute,
           int64_t *predictions) {
    if (!rows || rows > 257 || !columns || columns > width || width > 32768 ||
        width % columns || !predictions)
      throw std::invalid_argument("token selection extent");
    scores.resize(rows * columns);
    best.resize(rows);
    for (size_t column = 0; column < width; column += columns) {
      compute(column, scores.data());
      for (size_t r = 0; r < rows; ++r) {
        const float *block = scores.data() + r * columns;
        if (column == 0) {
          best[r] = block[0];
          predictions[r] = 0;
        }
        // Scan in vocabulary order, including through NaNs. Reducing each
        // block independently would mishandle a NaN at a later block's start.
        float maximum = best[r];
        int64_t selected = predictions[r];
        for (size_t i = column == 0 ? 1 : 0; i < columns; ++i)
          if (maximum < block[i]) {
            maximum = block[i];
            selected = int64_t(column + i);
          }
        best[r] = maximum;
        predictions[r] = selected;
      }
    }
  }
  size_t capacity_bytes() const {
    return (scores.capacity() + best.capacity()) * sizeof(float);
  }
};
} // namespace transcribro::standalone
