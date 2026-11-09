#include "fixture.h"
#include "compact_head.h"
#include "mlas.h"
#include "packed_head.h"
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <time.h>
using namespace transcribro::research;
namespace {
double cpu(clockid_t clock = CLOCK_PROCESS_CPUTIME_ID) {
  timespec t{};
  clock_gettime(clock, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
class Pair {
  std::mutex mutex;
  std::condition_variable changed;
  bool stop = false, ready = false, done = false;
  const transcribro::standalone::PackedHead *head = nullptr;
  const float *input = nullptr;
  float *output = nullptr;
  size_t rows = 0;
  std::thread worker;

public:
  Pair()
      : worker([&] {
          std::unique_lock<std::mutex> lock(mutex);
          for (;;) {
            changed.wait(lock, [&] { return stop || ready; });
            if (stop)
              break;
            ready = false;
            const auto *h = head;
            const auto *x = input;
            auto *y = output;
            const auto m = rows;
            lock.unlock();
            h->run_columns(x, m, y, 16384, 32768);
            lock.lock();
            done = true;
            changed.notify_one();
          }
        }) {}
  ~Pair() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    changed.notify_one();
    worker.join();
  }
  void run(const transcribro::standalone::PackedHead &h, const float *x,
           size_t m, float *y) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      head = &h;
      input = x;
      rows = m;
      output = y;
      done = false;
      ready = true;
    }
    changed.notify_one();
    h.run_columns(x, m, y, 0, 16384);
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] { return done; });
  }
};
} // namespace
int main(int argc, char **argv) try {
  if (argc < 3 || argc > 4)
    throw std::runtime_error(
        "usage: benchmark_head_layout HEAD_BIN OUTPUT_JSON [WIDTH]");
  std::vector<size_t> widths{0,   16,  32,  48,  64,  80,  96,
                             112, 128, 144, 256, 512, 1024};
  if (argc == 4) {
    std::vector<size_t> selected;
    std::istringstream arguments(argv[3]);
    std::string word;
    while (std::getline(arguments, word, ',')) {
      const size_t width = std::stoul(word);
      if (std::find(widths.begin(), widths.end(), width) == widths.end())
        throw std::runtime_error("unsupported width");
      selected.push_back(width);
    }
    if (selected.empty())
      throw std::runtime_error("empty selection");
    widths = selected;
  }
  std::ifstream file(argv[1], std::ios::binary);
  const auto magic = read_values<char>(file, 8);
  if (std::memcmp(magic.data(), "TRHEAD1\0", 8))
    throw std::runtime_error("head magic");
  const auto h = read_values<uint32_t>(file, 4);
  const size_t k = h[0], n = h[1], cases = h[2];
  if (k != 640 || n != 32768 || cases != 64 || h[3] > 255)
    throw std::runtime_error("head geometry");
  const float scale = read_values<float>(file, 1)[0];
  const auto bytes = read_values<uint8_t>(file, k * n);
  auto x = read_values<float>(file, k * cases);
  x.resize(k * 257);
  for (size_t i = k * cases; i < x.size(); ++i)
    x[i] = x[i % (k * cases)];
  std::vector<float> weights(k * n), alternate(k * n);
  for (size_t c = 0; c < n; ++c)
    for (size_t j = 0; j < k; ++j)
      weights[j * n + c] = float(int(bytes[c * k + j]) - int(h[3])) * scale;
  const auto packed_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, n, k);
  if (!packed_size)
    throw std::runtime_error("packed head unsupported");
  const size_t alignment = MlasGetPreferredBufferAlignment();
  std::vector<uint8_t> packed_storage(packed_size + alignment);
  void *packed = reinterpret_cast<void *>(
      (reinterpret_cast<uintptr_t>(packed_storage.data()) + alignment - 1) &
      ~(uintptr_t(alignment) - 1));
  MlasGemmPackB(CblasNoTrans, CblasNoTrans, n, k, weights.data(), n, packed);
  std::ofstream output(argv[2]);
  output << std::setprecision(12) << "{\"packed_bytes\":" << packed_size
         << ",\"cases\":[";
  bool first = true;
  for (size_t width : widths) {
    transcribro::standalone::PackedHead exact_packed(
        [&](size_t j, size_t c) { return weights[j * n + c]; },
        width == 32                    ? 128
        : (width == 48 || width == 80) ? 64
                                       : 0);
    auto pair = width == 80 ? std::make_unique<Pair>() : nullptr;
    auto compact =
        width == 96
            ? std::make_unique<transcribro::standalone::CompactHead<false>>(
                  [&](size_t j, size_t c) { return bytes[c * k + j]; }, scale,
                  int32_t(h[3]))
            : nullptr;
    auto centered =
        width == 112
            ? std::make_unique<transcribro::standalone::CompactHead<true>>(
                  [&](size_t j, size_t c) { return bytes[c * k + j]; }, scale,
                  int32_t(h[3]))
            : nullptr;
    std::vector<uint8_t> borrowed_codes(width == 144 ? k * n : 0);
    if (width == 144)
      for (size_t c = 0; c < n; ++c)
        for (size_t j = 0; j < k; ++j) {
          const size_t at = (c / 64) * (640 * 64) + (j / 128) * (128 * 64) +
                            (c % 64 / 16) * (128 * 16) + (j % 128) * 16 +
                            c % 16;
          borrowed_codes[at] = bytes[c * k + j];
        }
    auto borrowed =
        width == 144
            ? std::make_unique<transcribro::standalone::CompactHead<false>>(
                  static_cast<const uint8_t *>(borrowed_codes.data()), scale,
                  int32_t(h[3]))
            : nullptr;
    if (width >= 64 && width != 80 && width != 96 && width != 112 &&
        width != 144)
      for (size_t c = 0; c < n; c += width)
        for (size_t j = 0; j < k; ++j)
          std::copy_n(weights.data() + j * n + c, width,
                      alternate.data() + c * k + j * width);
    for (size_t rows : {1, 2, 3, 4, 5, 7, 8, 16, 31, 63, 64, 65, 127, 128, 129,
                        255, 256, 257}) {
      std::vector<float> reference(rows * n), actual(rows * n);
      for (size_t offset = 0; offset + rows <= std::max(cases, rows);
           offset += rows) {
        MlasGemm(CblasNoTrans, CblasNoTrans, rows, n, k, 1.f,
                 x.data() + offset * k, k, weights.data(), n, 0.f,
                 reference.data(), n, nullptr);
        for (int round = 0; round < (offset == 0 ? 3 : 1); ++round)
          for (int order = 0; order < (offset == 0 ? 2 : 1); ++order) {
            const bool candidate = offset != 0 || (round + order) % 2;
            const double wall_start = cpu(CLOCK_MONOTONIC), start = cpu();
            if (!candidate)
              MlasGemm(CblasNoTrans, CblasNoTrans, rows, n, k, 1.f,
                       x.data() + offset * k, k, weights.data(), n, 0.f,
                       actual.data(), n, nullptr);
            else if (width == 144)
              borrowed->run(x.data() + offset * k, rows, actual.data());
            else if (width == 96)
              compact->run(x.data() + offset * k, rows, actual.data());
            else if (width == 112)
              centered->run(x.data() + offset * k, rows, actual.data());
            else if (width == 80)
              pair->run(exact_packed, x.data() + offset * k, rows,
                        actual.data());
            else if (width == 16 || width == 32 || width == 48)
              exact_packed.run(x.data() + offset * k, rows, actual.data());
            else if (width == 0)
              MlasGemm(CblasNoTrans, rows, n, k, 1.f, x.data() + offset * k, k,
                       packed, 0.f, actual.data(), n, nullptr);
            else
              for (size_t c = 0; c < n; c += width)
                MlasGemm(CblasNoTrans, CblasNoTrans, rows, width, k, 1.f,
                         x.data() + offset * k, k, alternate.data() + c * k,
                         width, 0.f, actual.data() + c, n, nullptr);
            const double elapsed = cpu() - start;
            const double wall_elapsed = cpu(CLOCK_MONOTONIC) - wall_start;
            size_t differences = 0;
            double max_error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
              differences += std::memcmp(&actual[i], &reference[i], 4) != 0;
              max_error = std::max(max_error,
                                   std::abs(double(actual[i]) - reference[i]));
            }
            if (!first)
              output << ',';
            first = false;
            output << "{\"width\":" << width << ",\"rows\":" << rows
                   << ",\"layout\":\""
                   << (width == 16    ? "packed-k-major"
                       : width == 32  ? "packed-columns-128"
                       : width == 48  ? "packed-columns-64"
                       : width == 80  ? "two-workers-columns-64"
                       : width == 96  ? "compact-u8-columns-64"
                       : width == 112 ? "compact-i16-columns-64"
                       : width == 144 ? "borrowed-u8-columns-64"
                       : width == 0   ? "mlas-default-pack"
                                      : "contiguous-columns")
                   << '"' << ",\"offset\":" << offset << ",\"round\":" << round
                   << ",\"candidate\":" << (candidate ? "true" : "false")
                   << ",\"differences\":" << differences
                   << ",\"maximum_error\":" << max_error
                   << ",\"cpu_seconds\":" << elapsed
                   << ",\"wall_seconds\":" << wall_elapsed << '}';
          }
      }
    }
    std::cout << "width " << width << " complete" << std::endl;
  }
  output << "]}\n";
  if (!output)
    throw std::runtime_error("output write");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
