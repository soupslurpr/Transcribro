#include "fixture.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <time.h>
using namespace transcribro::research;
namespace {
double cpu() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return double(t.tv_sec) + double(t.tv_nsec) / 1e9;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 3)
    throw std::runtime_error("usage: benchmark_tiled FIXTURES OUTPUT_JSON");
  std::vector<std::filesystem::path> fixtures;
  for (const auto &entry : std::filesystem::directory_iterator(argv[1]))
    if (std::filesystem::exists(entry.path() / "projection.bin"))
      fixtures.push_back(entry.path());
  std::sort(fixtures.begin(), fixtures.end());
  if (fixtures.empty())
    throw std::runtime_error("no fixtures");
  std::ofstream output(argv[2]);
  output << std::setprecision(12) << "{\"cases\":[";
  bool first = true;
  for (const auto &path : fixtures) {
    Fixture f((path / "projection.bin").string());
    Projection p(f.k, f.n, f.weights.data(), f.scales.data(), f.zeros.data(),
                 f.bias.data(), 8);
    for (size_t rows : {1, 4, 8, 9, 15, 16, 17, 31, 64, 128, 256}) {
      std::vector<float> x(rows * f.k), reference(rows * f.n),
          actual(rows * f.n);
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = f.inputs[i % f.inputs.size()];
      Quantized q;
      quantize(x.data(), rows, f.k, q);
      p.multiply(q, rows, reference.data());
      for (int round = 0; round < 3; ++round)
        for (int order = 0; order < 2; ++order) {
          const bool tiled = (order + round) % 2;
          auto run = [&] {
            quantize(x.data(), rows, f.k, q, tiled);
            p.multiply(q, rows, actual.data());
          };
          run();
          const double start = cpu();
          for (int i = 0; i < 3; ++i)
            run();
          const double elapsed = (cpu() - start) / 3;
          size_t differences = 0;
          for (size_t i = 0; i < actual.size(); ++i)
            differences += std::memcmp(&actual[i], &reference[i], 4) != 0;
          if (!first)
            output << ',';
          first = false;
          output << "{\"fixture\":\"" << path.filename().string()
                 << "\",\"rows\":" << rows << ",\"round\":" << round
                 << ",\"tiled\":" << (tiled ? "true" : "false")
                 << ",\"differences\":" << differences
                 << ",\"cpu_seconds\":" << elapsed << '}';
          if (differences)
            throw std::runtime_error("tiled projection mismatch");
        }
    }
    std::cout << path.filename().string() << " complete" << std::endl;
  }
  output << "]}\n";
  if (!output)
    throw std::runtime_error("output write");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
