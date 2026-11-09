#pragma once

// Diagnostic-only inclusive timing. Ordinary builds contain no clocks or
// counters. Each participating thread writes one report on thread exit.
namespace transcribro::standalone {
enum class ProfileStage {
  Frontend,
  Vad,
  Encoder,
  Adapter,
  Cross,
  Decoder,
  Projection,
  Fp32Gemm,
  Fp32Conv,
  Normalize,
  Attention,
  OutputHead,
  HeadSingle,
  HeadSmall,
  HeadLarge,
  LoadModel,
  LoadFrontendWeights,
  Count
};
}

#ifdef TRANSCRIBRO_PROFILE
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

namespace transcribro::standalone {
inline uint64_t profile_clock(clockid_t clock) {
  timespec value{};
  clock_gettime(clock, &value);
  return uint64_t(value.tv_sec) * 1000000000 + uint64_t(value.tv_nsec);
}
struct ProfileCounters {
  struct Entry {
    uint64_t calls = 0, wall_ns = 0, cpu_ns = 0;
  };
  std::array<Entry, size_t(ProfileStage::Count)> entries{};
  bool enabled = [] {
    const char *value = std::getenv("TRANSCRIBRO_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  ~ProfileCounters() {
    if (!enabled)
      return;
    constexpr const char *names[]{"frontend",
                                  "vad",
                                  "encoder",
                                  "adapter",
                                  "cross",
                                  "decoder",
                                  "projection",
                                  "fp32_gemm",
                                  "fp32_conv",
                                  "normalization",
                                  "attention",
                                  "output_head",
                                  "head_single",
                                  "head_small",
                                  "head_large",
                                  "load_model",
                                  "load_frontend_weights"};
    flockfile(stderr);
    std::fputs("{\"standalone_inclusive_profile\":{", stderr);
    for (size_t i = 0; i < entries.size(); ++i) {
      const auto &value = entries[i];
      std::fprintf(stderr,
                   "%s\"%s\":{\"calls\":%llu,\"cpu_seconds\":%.9f,"
                   "\"wall_seconds\":%.9f}",
                   i ? "," : "", names[i],
                   static_cast<unsigned long long>(value.calls),
                   double(value.cpu_ns) / 1e9, double(value.wall_ns) / 1e9);
    }
    std::fputs("}}\n", stderr);
    funlockfile(stderr);
  }
};
inline thread_local ProfileCounters profile_counters;
class ProfileScope {
  ProfileStage stage;
  uint64_t wall = 0, cpu = 0;
  bool enabled;

public:
  explicit ProfileScope(ProfileStage stage)
      : stage(stage), enabled(profile_counters.enabled) {
    if (enabled) {
      wall = profile_clock(CLOCK_MONOTONIC);
      cpu = profile_clock(CLOCK_THREAD_CPUTIME_ID);
    }
  }
  ~ProfileScope() {
    if (enabled) {
      const auto cpu_end = profile_clock(CLOCK_THREAD_CPUTIME_ID);
      const auto wall_end = profile_clock(CLOCK_MONOTONIC);
      auto &entry = profile_counters.entries[size_t(stage)];
      ++entry.calls;
      entry.cpu_ns += cpu_end - cpu;
      entry.wall_ns += wall_end - wall;
    }
  }
};
} // namespace transcribro::standalone
#else
namespace transcribro::standalone {
class ProfileScope {
public:
  explicit ProfileScope(ProfileStage) {}
};
} // namespace transcribro::standalone
#endif
