#include "zstd_reader.h"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>
struct Mapping {
  void *data;
  size_t size;
  explicit Mapping(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error("open");
    struct stat s{};
    if (fstat(fd, &s) || s.st_size <= 0 || s.st_size > 512 * 1024 * 1024) {
      close(fd);
      throw std::runtime_error("stat");
    }
    size = s.st_size;
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED)
      throw std::runtime_error("mmap");
  }
  ~Mapping() { munmap(data, size); }
};
double now(clockid_t c) {
  timespec t{};
  clock_gettime(c, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}
int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::runtime_error(
        "usage: benchmark_decompression ZSTD ARCHIVE OUTPUT");
  Mapping compressed(argv[1]), original(argv[2]);
  // Reject truncation, trailing data and incorrect advertised/output sizes.
  std::vector<char> checked_output(original.size), damaged(compressed.size + 1);
  memcpy(damaged.data(), compressed.data, compressed.size);
  auto reject = [&](const void *input, size_t size, size_t expected) {
    bool rejected = false;
    try {
      transcribro::decompress_model_frame(input, size, checked_output.data(),
                                          expected);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("malformed frame accepted");
  };
  for (size_t n : {size_t(0), size_t(3), size_t(10), compressed.size - 1})
    reject(compressed.data, n, original.size);
  reject(compressed.data, compressed.size, original.size - 1);
  reject(compressed.data, compressed.size, original.size + 1);
  reject(damaged.data(), damaged.size(), original.size);
  damaged[0] = 0;
  reject(damaged.data(), compressed.size, original.size);
  checked_output.clear();
  checked_output.shrink_to_fit();
  damaged.clear();
  damaged.shrink_to_fit();
  std::ofstream out(argv[3]);
  out << std::setprecision(12) << "{\"trials\":[";
  for (int i = 0; i < 16; ++i) {
    const bool direct = (i / 2 + i % 2) % 2;
    const double wall = now(CLOCK_MONOTONIC),
                 cpu = now(CLOCK_PROCESS_CPUTIME_ID);
    void *buffer = nullptr;
    if (posix_memalign(&buffer, 64, original.size))
      throw std::bad_alloc();
    std::unique_ptr<void, decltype(&free)> owned(buffer, free);
    if (direct) {
      transcribro::decompress_model_frame(compressed.data, compressed.size,
                                          buffer, original.size);
    } else {
      size_t at = 0;
      transcribro::ZstdReader reader([&](void *dest, size_t count) {
        const size_t n = std::min(count, compressed.size - at);
        memcpy(dest, static_cast<const char *>(compressed.data) + at, n);
        at += n;
        return n;
      });
      reader.read(buffer, original.size);
      reader.finish();
    }
    const double cpu_used = now(CLOCK_PROCESS_CPUTIME_ID) - cpu,
                 wall_used = now(CLOCK_MONOTONIC) - wall;
    if (memcmp(buffer, original.data, original.size))
      throw std::runtime_error("model bytes changed");
    out << (i ? "," : "") << "{\"trial\":" << i
        << ",\"direct\":" << (direct ? "true" : "false")
        << ",\"exact\":true,\"cpu_seconds\":" << cpu_used
        << ",\"wall_seconds\":" << wall_used << '}';
  }
  out << "],\"invalid_frames_rejected\":8}\n";
  if (!out)
    throw std::runtime_error("output");
  std::cout << "All decompressed bytes exact\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
