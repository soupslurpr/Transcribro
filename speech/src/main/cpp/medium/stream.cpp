// Streaming policy adapted from Moonshine 0.1.5 (MIT), commit
// 234f60faa0eb388b01cdf7e60aca232af37aefda. See CREDITS and
// licenses/Moonshine.txt.
#include "stream.h"
#include "medium.h"
#include "text.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Vec = std::vector<float>;
using Tokens = std::vector<int64_t>;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void checked(int code) {
  if (code)
    throw std::runtime_error(tm_error());
}
using Model = std::unique_ptr<void, decltype(&tm_destroy)>;
using Cache = std::unique_ptr<void, decltype(&tm_cache_destroy)>;

struct Sequence {
  void *model;
  Cache cache{tm_cache_create(), tm_cache_destroy};
  tm_frontend_state frontend{};
  size_t processed = 0, emitted = 0;
  Vec features, memory, cross_keys, cross_values;
  Tokens tokens;
  bool cross_valid = false;
  explicit Sequence(void *model) : model(model) {
    require(bool(cache), "decoder cache allocation");
  }
  void reset() {
    frontend = {};
    processed = emitted = 0;
    features.clear();
    memory.clear();
    tokens.clear();
    cross_keys.clear();
    cross_values.clear();
    cross_valid = false;
    tm_cache_reset(cache.get());
  }
  Tokens decode(const Tokens &input) {
    Tokens predictions(input.size());
    checked(tm_decoder_cached(model, cache.get(), input.data(), input.size(),
                              cross_keys.data(), cross_values.data(),
                              memory.size() / 640, predictions.data()));
    return predictions;
  }
  void update(const Vec &audio, bool final, bool previews) {
    const bool is_new = processed == 0;
    if (processed < audio.size()) {
      std::array<float, 8 * 768> output;
      while (processed + 1280 <= audio.size()) {
        // Pair only blocks already available. This preserves the frontend's
        // arithmetic without adding any capture or preview delay.
        const size_t blocks = audio.size() - processed >= 2560 ? 2 : 1;
        size_t count;
        checked(tm_frontend(model, audio.data() + processed, blocks * 1280,
                            &frontend, output.data(), 8, &count));
        require(count == blocks * 4, "frontend block shape");
        features.insert(features.end(), output.begin(),
                        output.begin() + count * 768);
        processed += blocks * 1280;
      }
      if (final) {
        emitted = 0;
        memory.clear();
        cross_valid = false;
      }
      const size_t total = features.size() / 768;
      const size_t stable = final ? total : total > 16 ? total - 16 : 0;
      if (stable > emitted) {
        const size_t start = emitted > 224 ? emitted - 224 : 0;
        Vec encoded((total - start) * 768);
        checked(tm_encoder(model, features.data() + start * 768, total - start,
                           encoded.data()));
        const size_t previous = memory.size();
        memory.resize(previous + (stable - emitted) * 640);
        checked(tm_adapter(model, encoded.data() + (emitted - start) * 768,
                           stable - emitted, emitted,
                           memory.data() + previous));
        emitted = stable;
        cross_valid = false;
      }
    }
    if (memory.empty() || (!final && !previews))
      return;
    if (!cross_valid) {
      cross_keys.resize(memory.size() * 14);
      cross_values.resize(cross_keys.size());
      checked(tm_cross(model, memory.data(), memory.size() / 640,
                       cross_keys.data(), cross_values.data()));
      cross_valid = true;
    }
    const size_t budget = std::min(
        size_t(256), size_t(std::ceil((float(audio.size()) / 16000.0f) *
                                      (final ? 20.0f : 6.5f))));
    tm_cache_reset(cache.get());
    Tokens draft;
    for (auto token : tokens)
      if (token != 1 && token != 2 && draft.size() < budget)
        draft.push_back(token);
    if (!draft.empty() && !is_new) {
      Tokens input{1};
      input.insert(input.end(), draft.begin(), draft.end());
      const auto predictions = decode(input);
      size_t accepted = 0;
      while (accepted < draft.size() &&
             predictions[accepted] == draft[accepted])
        ++accepted;
      tokens.assign(1, 1);
      tokens.insert(tokens.end(), draft.begin(), draft.begin() + accepted);
      int64_t next;
      if (accepted == draft.size())
        next = predictions.back();
      else {
        tm_cache_reset(cache.get());
        next = decode(tokens).back();
      }
      while (next != 2 && tokens.size() - 1 < budget) {
        tokens.push_back(next);
        next = decode(Tokens{next}).back();
      }
    } else {
      tokens.assign(1, 1);
      for (size_t i = 0; i < budget; ++i) {
        const auto next = decode(Tokens{tokens.back()}).back();
        tokens.push_back(next);
        if (next == 2)
          break;
      }
    }
  }
};
struct Segment {
  Vec audio;
  bool complete = false;
  std::string text;
};
class Stream {
  Model model{nullptr, tm_destroy};
  std::unique_ptr<Sequence> sequence;
  std::vector<std::string> vocabulary;
  tm_vad_state vad{};
  std::array<float, 32> probabilities{};
  std::array<float, 8192> lookbehind{};
  Vec pending;
  std::vector<Segment> segments;
  size_t probability_index = 0, processed = 0, last_end = 0,
         sequence_index = SIZE_MAX;
  bool active = false, finished = false, failed = false;
  std::string result;

public:
  Stream(const char *archive, const char *tokenizer,
         const void *archive_data = nullptr, size_t archive_size = 0,
         const void *tokenizer_data = nullptr, size_t tokenizer_size = 0) {
    require((archive && tokenizer) || (archive_data && tokenizer_data),
            "null model sources");
    char error[1024]{};
    model.reset(archive_data ? tm_create_memory(archive_data, archive_size,
                                                error, sizeof(error))
                             : tm_create(archive, error, sizeof(error)));
    require(bool(model), error);
    sequence = std::make_unique<Sequence>(model.get());
    std::vector<unsigned char> bytes;
    if (tokenizer_data) {
      require(tokenizer_size > 0 && tokenizer_size < 1024 * 1024,
              "tokenizer memory extent");
      const auto *begin = static_cast<const unsigned char *>(tokenizer_data);
      bytes.assign(begin, begin + tokenizer_size);
    } else {
      std::ifstream file(tokenizer, std::ios::binary | std::ios::ate);
      require(bool(file) && file.tellg() > 0 && file.tellg() < 1024 * 1024,
              "tokenizer extent");
      bytes.resize(size_t(file.tellg()));
      file.seekg(0);
      file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
      require(bool(file), "tokenizer read");
    }
    for (size_t at = 0; at < bytes.size();) {
      size_t length = bytes[at++];
      if (length >= 128) {
        require(at < bytes.size(), "tokenizer length truncated");
        length = length - 128 + bytes[at++] * 128;
      }
      require(length <= bytes.size() - at, "tokenizer entry extent");
      vocabulary.emplace_back(reinterpret_cast<const char *>(bytes.data() + at),
                              length);
      at += length;
    }
    require(vocabulary.size() == 32768, "tokenizer vocabulary size");
  }
  std::string detokenize(const Tokens &tokens) const {
    std::string text;
    for (auto token : tokens) {
      require(token >= 0 && size_t(token) < vocabulary.size(),
              "invalid decoded token");
      const auto &piece = vocabulary[size_t(token)];
      require(!piece.empty(), "empty decoded token");
      if (piece.size() > 2 && piece.front() == '<' && piece.back() == '>')
        continue;
      text += piece;
    }
    size_t at = 0;
    while ((at = text.find("\xe2\x96\x81", at)) != std::string::npos)
      text.replace(at++, 3, " ");
    const auto first = text.find_first_not_of(" \t\r\n\v\f"),
               last = text.find_last_not_of(" \t\r\n\v\f");
    return first == std::string::npos
               ? ""
               : transcribro::standalone::valid_text(
                     text.substr(first, last - first + 1));
  }
  void reset(bool keep_vad) {
    if (!keep_vad)
      vad = {};
    probabilities.fill(0);
    lookbehind.fill(0);
    pending.clear();
    segments.clear();
    probability_index = processed = last_end = 0;
    sequence_index = SIZE_MAX;
    active = finished = failed = false;
    result.clear();
    sequence->reset();
  }
  void push(const float *audio, size_t samples) {
    require(!finished && !failed && (audio || !samples),
            "stream is not accepting audio");
    require(samples <= 64000 && pending.size() <= 64000 - samples,
            "pending audio limit; update stream first");
    for (size_t i = 0; i < samples; ++i)
      require(std::isfinite(audio[i]) && std::abs(audio[i]) <= 16,
              "invalid PCM sample");
    if (samples)
      pending.insert(pending.end(), audio, audio + samples);
  }
  void update(bool final, bool previews = true) {
    require(!finished && !failed, "stream is finished or failed");
    try {
      update_impl(final, previews);
    } catch (...) {
      failed = true;
      throw;
    }
  }
  void update_impl(bool final, bool previews) {
    if (final && pending.size() % 512)
      pending.resize((pending.size() + 511) / 512 * 512, 0);
    const size_t count = pending.size() / 512 * 512;
    std::vector<size_t> updated;
    auto changed = [&] {
      if (updated.empty() || updated.back() != segments.size() - 1)
        updated.push_back(segments.size() - 1);
    };
    for (size_t at = 0; at < count; at += 512) {
      require(processed <= 16000 * 60 * 60, "research session duration limit");
      const float *chunk = pending.data() + at;
      processed += 512;
      std::move(lookbehind.begin() + 512, lookbehind.end(), lookbehind.begin());
      std::copy_n(chunk, 512, lookbehind.end() - 512);
      float probability;
      checked(tm_vad(model.get(), chunk, &vad, &probability));
      probabilities[probability_index] = probability;
      probability_index = (probability_index + 1) % 32;
      float total = 0;
      for (float value : probabilities)
        total += value;
      float smoothed = total / 32.0f;
      const size_t length = active ? segments.back().audio.size() : 0;
      if (length > 160000)
        smoothed *= float(length - 160000) / 160000.0f;
      const bool voice = smoothed > .3f;
      if (voice && !active) {
        require(segments.size() < 4096, "segment limit");
        const size_t size = std::min(size_t(8192), processed - last_end);
        segments.push_back(
            {Vec(lookbehind.end() - size, lookbehind.end()), false, {}});
        changed();
      } else if (active) {
        auto &segment = segments.back();
        require(segment.audio.size() <= 320000 - 512,
                "utterance duration limit");
        segment.audio.insert(segment.audio.end(), chunk, chunk + 512);
        segment.complete = !voice;
        changed();
        if (!voice)
          last_end = processed;
      }
      active = voice;
    }
    pending.erase(pending.begin(), pending.begin() + count);
    if (final && active) {
      segments.back().complete = true;
      changed();
      active = false;
    }
    for (const size_t index : updated) {
      if (sequence_index != index) {
        sequence->reset();
        sequence_index = index;
      }
      auto &segment = segments[index];
      sequence->update(segment.audio, segment.complete, previews);
      segment.text = detokenize(sequence->tokens);
      if (segment.complete)
        Vec().swap(segment.audio);
    }
    result.clear();
    for (size_t i = 0; i < segments.size(); ++i) {
      if (i)
        result += ' ';
      result += segments[i].text;
    }
    const auto first = result.find_first_not_of(" \t\r\n\v\f"),
               last = result.find_last_not_of(" \t\r\n\v\f");
    result = first == std::string::npos
                 ? ""
                 : result.substr(first, last - first + 1);
    finished = final;
  }
  const char *text() const { return result.c_str(); }
  size_t segment_count() const { return segments.size(); }
  const char *segment_text(size_t i) const {
    return i < segments.size() ? segments[i].text.c_str() : nullptr;
  }
  bool segment_complete(size_t i) const {
    return i < segments.size() && segments[i].complete;
  }
};
thread_local std::string last_error;
template <class F> int call(void *handle, F function) noexcept {
  try {
    require(handle, "null stream");
    function(*static_cast<Stream *>(handle));
    last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return 1;
  }
}
} // namespace
extern "C" void *ts_create(const char *archive, const char *tokenizer,
                           char *error, size_t size) {
  try {
    return new Stream(archive, tokenizer);
  } catch (const std::exception &e) {
    if (error && size) {
      std::strncpy(error, e.what(), size - 1);
      error[size - 1] = 0;
    }
    return nullptr;
  }
}
extern "C" void *ts_create_memory(const void *archive, size_t archive_size,
                                  const void *tokenizer, size_t tokenizer_size,
                                  char *error, size_t error_size) {
  try {
    return new Stream(nullptr, nullptr, archive, archive_size, tokenizer,
                      tokenizer_size);
  } catch (const std::exception &e) {
    if (error && error_size) {
      std::strncpy(error, e.what(), error_size - 1);
      error[error_size - 1] = 0;
    }
    return nullptr;
  }
}
extern "C" void ts_destroy(void *stream) {
  delete static_cast<Stream *>(stream);
}
extern "C" int ts_push(void *stream, const float *audio, size_t count) {
  return call(stream, [&](Stream &s) { s.push(audio, count); });
}
extern "C" int ts_update(void *stream, int final) {
  return call(stream, [&](Stream &s) { s.update(final != 0); });
}
extern "C" int ts_update_ex(void *stream, int final, int previews) {
  return call(stream, [&](Stream &s) { s.update(final != 0, previews != 0); });
}
extern "C" size_t ts_segment_count(void *stream) {
  return stream ? static_cast<Stream *>(stream)->segment_count() : 0;
}
extern "C" const char *ts_segment_text(void *stream, size_t i) {
  return stream ? static_cast<Stream *>(stream)->segment_text(i) : nullptr;
}
extern "C" int ts_segment_complete(void *stream, size_t i) {
  return stream && static_cast<Stream *>(stream)->segment_complete(i);
}
extern "C" int ts_reset(void *stream, int keep_vad) {
  return call(stream, [&](Stream &s) { s.reset(keep_vad != 0); });
}
extern "C" const char *ts_text(void *stream) {
  return stream ? static_cast<Stream *>(stream)->text() : "";
}
extern "C" const char *ts_error() { return last_error.c_str(); }
