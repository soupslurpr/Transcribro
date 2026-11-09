#pragma once
#include <cstdint>
#include <string>

namespace transcribro::standalone {
// Byte-fallback tokens can end a preview inside a UTF-8 sequence. Keep valid
// text unchanged and replace invalid bytes before exposing a UTF-8 C string.
inline std::string valid_text(const std::string &source) {
  std::string output;
  for (size_t at = 0; at < source.size();) {
    const auto c = uint8_t(source[at]);
    size_t length = c < 0x80                 ? 1
                    : c >= 0xc2 && c <= 0xdf ? 2
                    : c >= 0xe0 && c <= 0xef ? 3
                    : c >= 0xf0 && c <= 0xf4 ? 4
                                             : 0;
    bool valid = c != 0 && length && length <= source.size() - at;
    for (size_t i = 1; valid && i < length; ++i)
      valid = (uint8_t(source[at + i]) & 0xc0) == 0x80;
    if (valid && length >= 3) {
      const auto second = uint8_t(source[at + 1]);
      valid = !(c == 0xe0 && second < 0xa0) && !(c == 0xed && second >= 0xa0) &&
              !(c == 0xf0 && second < 0x90) && !(c == 0xf4 && second >= 0x90);
    }
    if (valid) {
      output.append(source, at, length);
      at += length;
    } else {
      output += '?';
      ++at;
    }
  }
  return output;
}
} // namespace transcribro::standalone
