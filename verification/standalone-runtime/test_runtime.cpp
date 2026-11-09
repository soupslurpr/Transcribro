#include "medium.h"
#include "stream.h"
#include "text.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void checked(int status) { require(status == 0, tm_error()); }
using Model = std::unique_ptr<void, decltype(&tm_destroy)>;
Model create(const char *archive) {
  char error[1024]{};
  Model model(tm_create(archive, error, sizeof(error)), tm_destroy);
  require(bool(model), error);
  return model;
}
std::vector<float> vad_sequence(void *model) {
  tm_vad_state state{};
  std::array<float, 512> audio{};
  std::vector<float> results;
  for (size_t step = 0; step < 32; ++step) {
    for (size_t i = 0; i < 512; ++i)
      audio[i] = std::sin(float(i + step * 512) * .034f) * .1f;
    float probability;
    checked(tm_vad(model, audio.data(), &state, &probability));
    results.push_back(probability);
    results.insert(results.end(), state.hidden, state.hidden + 128);
    results.insert(results.end(), state.cell, state.cell + 128);
  }
  return results;
}
void cache_sequence(void *model) {
  std::unique_ptr<void, decltype(&tm_cache_destroy)> cache(tm_cache_create(),
                                                           tm_cache_destroy);
  require(bool(cache), "cache create");
  const size_t memory = 3;
  std::vector<float> cross(14 * 640 * memory), keys, values;
  for (size_t i = 0; i < cross.size(); ++i)
    cross[i] = std::sin(float(i) * .02f) * .05f;
  for (size_t repeat = 0; repeat < 2; ++repeat) {
    keys.clear();
    values.clear();
    size_t past = 0;
    tm_cache_reset(cache.get());
    // Fill the complete 257-token cache, including large prefill batches.
    for (const size_t count :
         {size_t(1), size_t(4), size_t(1), size_t(3), size_t(16), size_t(31),
          size_t(64), size_t(128), size_t(9)}) {
      std::vector<int64_t> input(count), predictions(count);
      for (size_t i = 0; i < count; ++i)
        input[i] = past == 0 ? 1 : int64_t(3 + (past + i) * 107);
      std::vector<float> logits(count * 32768),
          next_keys(14 * 640 * (past + count)), next_values(next_keys.size());
      checked(tm_decoder(model, input.data(), count, keys.data(), values.data(),
                         past, cross.data(), cross.data(), memory,
                         logits.data(), next_keys.data(), next_values.data()));
      checked(tm_decoder_cached(model, cache.get(), input.data(), count,
                                cross.data(), cross.data(), memory,
                                predictions.data()));
      for (size_t i = 0; i < count; ++i) {
        const float *row = logits.data() + i * 32768;
        require(predictions[i] == std::max_element(row, row + 32768) - row,
                "persistent cache changed prediction");
      }
      past += count;
      keys.swap(next_keys);
      values.swap(next_values);
    }
    require(past == 257, "full cache coverage");
    tm_cache_reset(cache.get());
    int64_t invalid = 32768, prediction;
    require(tm_decoder_cached(model, cache.get(), &invalid, 1, cross.data(),
                              cross.data(), memory, &prediction) != 0,
            "invalid token accepted");
  }
}
std::vector<char> read(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file) && file.tellg() > 0, "test fixture extent");
  std::vector<char> bytes(size_t(file.tellg()));
  file.seekg(0);
  file.read(bytes.data(), bytes.size());
  require(bool(file), "test fixture read");
  return bytes;
}
void borrowed_memory(const char *archive, const char *tokenizer,
                     const char *pcm) {
  const auto bytes = read(archive);
  void *allocation = nullptr;
  require(posix_memalign(&allocation, 64, bytes.size()) == 0,
          "aligned model allocation");
  std::unique_ptr<void, decltype(&std::free)> memory(allocation, std::free);
  std::memcpy(allocation, bytes.data(), bytes.size());
  char error[1024]{};
  require(!tm_create_memory(static_cast<char *>(allocation) + 1,
                            bytes.size() - 1, error, sizeof(error)),
          "unaligned borrowed model accepted");
  require(!tm_create_memory(allocation, 63, error, sizeof(error)),
          "truncated borrowed model accepted");
  require(!tm_create_memory(nullptr, bytes.size(), error, sizeof(error)),
          "null borrowed model accepted");
  {
    auto file_model = create(archive);
    const auto reference = vad_sequence(file_model.get());
    file_model.reset();
    Model model(
        tm_create_memory(allocation, bytes.size(), error, sizeof(error)),
        tm_destroy);
    require(bool(model), error);
    require(vad_sequence(model.get()) == reference,
            "borrowed model changed inference");
  }
  // Reuse the same caller-owned archive after destroying the first model.
  auto tokens = read(tokenizer);
  std::unique_ptr<void, decltype(&ts_destroy)> stream(
      ts_create_memory(allocation, bytes.size(), tokens.data(), tokens.size(),
                       error, sizeof(error)),
      ts_destroy);
  require(bool(stream), error);
  std::fill(tokens.begin(), tokens.end(), char(0));
  tokens.clear();
  tokens.shrink_to_fit();
  require(ts_segment_count(stream.get()) == 0 &&
              !ts_segment_text(stream.get(), 0) &&
              !ts_segment_complete(stream.get(), 0),
          "invalid segment boundary accepted");
  if (!pcm)
    return;
  const auto audio_bytes = read(pcm);
  require(audio_bytes.size() % sizeof(float) == 0, "PCM extent");
  std::vector<float> audio(audio_bytes.size() / sizeof(float));
  std::memcpy(audio.data(), audio_bytes.data(), audio_bytes.size());
  std::string reference;
  for (int previews : {1, 0, 1}) {
    require(ts_reset(stream.get(), 0) == 0, "borrowed stream reset");
    for (size_t at = 0; at < audio.size(); at += 16384) {
      const size_t count = std::min(size_t(16384), audio.size() - at);
      require(ts_push(stream.get(), audio.data() + at, count) == 0 &&
                  ts_update_ex(stream.get(), 0, previews) == 0,
              "deferred preview update");
      if (!previews)
        for (size_t i = 0; i < ts_segment_count(stream.get()); ++i)
          require(ts_segment_complete(stream.get(), i) ||
                      std::strlen(ts_segment_text(stream.get(), i)) == 0,
                  "disabled preview emitted text");
    }
    std::array<float, 8192> drain{};
    require(ts_push(stream.get(), drain.data(), drain.size()) == 0 &&
                ts_update_ex(stream.get(), 1, previews) == 0,
            "borrowed stream finalization");
    const std::string actual = ts_text(stream.get());
    require(!actual.empty(), "borrowed stream lost speech");
    if (reference.empty())
      reference = actual;
    require(actual == reference, "preview policy changed final speech");
    require(ts_segment_count(stream.get()) > 0, "missing completed segment");
    for (size_t i = 0; i < ts_segment_count(stream.get()); ++i)
      require(ts_segment_text(stream.get(), i) &&
                  ts_segment_complete(stream.get(), i),
              "final segment not completed");
  }
}
} // namespace
int main(int argc, char **argv) try {
  require(argc == 3 || argc == 4,
          "usage: test_runtime ARCHIVE TOKENIZER [PCM]");
  using transcribro::standalone::valid_text;
  require(valid_text("caf\xc3\xa9 \xf0\x9f\x98\x80") ==
              "caf\xc3\xa9 \xf0\x9f\x98\x80",
          "valid UTF-8 changed");
  require(valid_text("a\xe2\x82") == "a??", "partial UTF-8 escaped");
  require(valid_text("\xc0\x80\xed\xa0\x80\xf4\x90\x80\x80") == "?????????",
          "invalid Unicode escaped");
  require(valid_text(std::string("a\0b", 3)) == "a?b", "embedded null escaped");
  if (std::strcmp(argv[2], "--reject-archive") == 0) {
    char error[1024]{};
    void *handle = tm_create(argv[1], error, sizeof(error));
    if (handle) {
      tm_destroy(handle);
      throw std::runtime_error("invalid archive accepted");
    }
    std::cout << error << '\n';
    return 0;
  }
  auto model = create(argv[1]);
  const auto reference = vad_sequence(model.get());
  std::vector<float> first, second;
  std::exception_ptr first_error, second_error;
  std::thread a([&] {
    try {
      first = vad_sequence(model.get());
    } catch (...) {
      first_error = std::current_exception();
    }
  });
  std::thread b([&] {
    try {
      auto independent = create(argv[1]);
      second = vad_sequence(independent.get());
    } catch (...) {
      second_error = std::current_exception();
    }
  });
  std::vector<float> shared;
  std::exception_ptr shared_error;
  try {
    shared = vad_sequence(model.get());
  } catch (...) {
    shared_error = std::current_exception();
  }
  a.join();
  b.join();
  if (shared_error)
    std::rethrow_exception(shared_error);
  if (first_error)
    std::rethrow_exception(first_error);
  if (second_error)
    std::rethrow_exception(second_error);
  require(first == reference && second == reference && shared == reference,
          "VAD state leaked between calls or models");
  cache_sequence(model.get());
  float dummy[640]{};
  require(tm_encoder(model.get(), nullptr, 1, dummy) != 0,
          "null encoder accepted");
  require(tm_adapter(model.get(), dummy, 1, SIZE_MAX, dummy) != 0,
          "adapter offset overflow accepted");
  require(tm_cross(model.get(), dummy, 1025, dummy, dummy) != 0,
          "unbounded cross input accepted");
  require(tm_decoder(nullptr, nullptr, 1, nullptr, nullptr, 0, nullptr, nullptr,
                     0, nullptr, nullptr, nullptr) != 0,
          "null model accepted");
  model.reset();
  char error[1024]{};
  std::unique_ptr<void, decltype(&ts_destroy)> stream(
      ts_create(argv[1], argv[2], error, sizeof(error)), ts_destroy);
  require(bool(stream), error);
  std::array<float, 8192> silence{};
  for (size_t repeat = 0; repeat < 3; ++repeat) {
    require(ts_reset(stream.get(), 0) == 0, "stream reset");
    require(ts_push(stream.get(), silence.data(), 8192) == 0 &&
                ts_update(stream.get(), 0) == 0,
            "silent update");
    require(ts_push(stream.get(), silence.data(), 37) == 0 &&
                ts_update(stream.get(), 1) == 0,
            "partial final audio");
    require(std::strlen(ts_text(stream.get())) == 0, "silence hallucinated");
    require(ts_push(stream.get(), silence.data(), 1) != 0 &&
                ts_update(stream.get(), 1) != 0,
            "finished stream reused without reset");
  }
  require(ts_reset(stream.get(), 0) == 0, "stream reset after finish");
  silence[0] = NAN;
  require(ts_push(stream.get(), silence.data(), 1) != 0,
          "nonfinite audio accepted");
  require(ts_push(stream.get(), nullptr, 1) != 0, "null audio accepted");
  require(ts_push(stream.get(), silence.data(), SIZE_MAX) != 0,
          "unbounded audio accepted");
  stream.reset();
  borrowed_memory(argv[1], argv[2], argc == 4 ? argv[3] : nullptr);
  std::cout << "concurrent state, persistent caches, invalid boundaries, "
               "silence, finish, reset, borrowed ownership and preview policy "
               "passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
