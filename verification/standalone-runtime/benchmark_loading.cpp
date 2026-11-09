#include "stream.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <time.h>
namespace {
double seconds(clockid_t clock) {
  timespec t{};
  clock_gettime(clock, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::runtime_error(
        "usage: benchmark_loading ARCHIVE TOKENIZER OUTPUT");
  std::ofstream out(argv[3]);
  out << std::setprecision(12) << "{\"model_initializations\":[";
  for (size_t i = 0; i < 10; ++i) {
    char error[1024]{};
    const double wall = seconds(CLOCK_MONOTONIC),
                 cpu = seconds(CLOCK_PROCESS_CPUTIME_ID);
    auto *stream = ts_create(argv[1], argv[2], error, sizeof(error));
    const double cpu_used = seconds(CLOCK_PROCESS_CPUTIME_ID) - cpu,
                 wall_used = seconds(CLOCK_MONOTONIC) - wall;
    if (!stream)
      throw std::runtime_error(error);
    ts_destroy(stream);
    out << (i ? "," : "") << "{\"iteration\":" << i
        << ",\"cpu_seconds\":" << cpu_used << ",\"wall_seconds\":" << wall_used
        << '}';
  }
  out << "]}\n";
  if (!out)
    throw std::runtime_error("output write");
  std::cout << "Ten fresh model initializations completed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
