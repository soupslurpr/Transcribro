#include "medium.h"
#include "projection.h"
#if defined(__x86_64__)
#include "projection_x86.h"
#endif
#include "compact_head.h"
#include "local_attention.h"
#include "mlas.h"
#include "packed_head.h"
#include "profile.h"
#include "token_selection.h"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {
using Vec = std::vector<float>;
using transcribro::research::Projection;
struct Quantized : transcribro::research::Quantized {
#if defined(__x86_64__)
  mutable std::vector<uint8_t> unsigned_activations;
  mutable std::vector<int32_t> integer_accumulators;
#endif
};
using transcribro::research::tiled_activations_available;
using transcribro::standalone::ProfileScope;
using transcribro::standalone::ProfileStage;

#ifdef TRANSCRIBRO_TRACE
thread_local tm_trace_callback trace_callback = nullptr;
thread_local void *trace_context = nullptr;
void trace(const std::string &name, const float *data, size_t count) {
  if (trace_callback)
    trace_callback(trace_context, name.c_str(), data, count);
}
#else
void trace(const std::string &, const float *, size_t) {}
#endif

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
uint64_t integer(const uint8_t *p, size_t bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < bytes; ++i)
    value |= uint64_t(p[i]) << (i * 8);
  return value;
}
struct Tensor {
  const uint8_t *data;
  uint32_t type;
  std::vector<size_t> shape;
  size_t bytes;
  const float *floats() const {
    require(type == 1, "expected float32 tensor");
    return reinterpret_cast<const float *>(data);
  }
};
uint8_t embedding_code(const Tensor &tensor, size_t token, size_t feature) {
  if (tensor.shape.size() == 2)
    return tensor.data[token * 640 + feature];
  const size_t offset =
      (token / 64) * (640 * 64) + (feature / 128) * (128 * 64) +
      (token % 64 / 16) * (128 * 16) + (feature % 128) * 16 + token % 16;
  return tensor.data[offset];
}

class Archive {
  uint8_t *mapped = nullptr;
  size_t length = 0;
  bool owns_mapping = false;
  std::unordered_map<std::string, Tensor> tensors;

public:
  explicit Archive(const char *path, const void *bytes = nullptr,
                   size_t size = 0) {
    if (bytes) {
      require(size >= 64 && size <= 1024ULL * 1024 * 1024 &&
                  reinterpret_cast<uintptr_t>(bytes) % 64 == 0,
              "archive memory extent or alignment");
      mapped = const_cast<uint8_t *>(static_cast<const uint8_t *>(bytes));
      length = size;
    } else {
      require(path, "null archive path");
      int fd = open(path, O_RDONLY | O_CLOEXEC);
      require(fd >= 0, "cannot open model archive");
      struct stat st{};
      if (fstat(fd, &st) || st.st_size < 64 ||
          st.st_size > 1024LL * 1024 * 1024) {
        close(fd);
        throw std::runtime_error("invalid archive file size");
      }
      length = size_t(st.st_size);
      void *address = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      require(address != MAP_FAILED, "cannot map archive");
      mapped = static_cast<uint8_t *>(address);
      owns_mapping = true;
    }
    try {
      require(std::memcmp(mapped, "TRMED001", 8) == 0 &&
                  integer(mapped + 8, 4) == 1,
              "unsupported archive");
      require(integer(mapped + 16, 8) == length &&
                  integer(mapped + 24, 8) == 64,
              "archive header mismatch");
      const size_t count = integer(mapped + 12, 4);
      require(count > 0 && count <= 4096 && 64 + count * 160 <= length,
              "invalid archive directory");
      size_t previous_end = 64 + count * 160;
      for (size_t i = 0; i < count; ++i) {
        const uint8_t *p = mapped + 64 + i * 160;
        const size_t name_len = strnlen(reinterpret_cast<const char *>(p), 80);
        require(name_len > 0 && name_len < 80, "invalid tensor name");
        const auto type = uint32_t(integer(p + 80, 4));
        const size_t rank = integer(p + 84, 4);
        require(rank <= 4, "invalid tensor rank");
        size_t elements = 1;
        std::vector<size_t> shape;
        for (size_t j = 0; j < rank; ++j) {
          size_t dim = integer(p + 88 + j * 8, 8);
          require(dim > 0 && dim <= 16777216 && elements <= length / dim,
                  "invalid tensor shape");
          shape.push_back(dim);
          elements *= dim;
        }
        size_t width = type == 1 || type == 6 ? 4 : type == 7 ? 8 : 1;
        require(type == 1 || type == 2 || type == 3 || type == 6 || type == 7,
                "invalid tensor type");
        const size_t offset = integer(p + 120, 8), bytes = integer(p + 128, 8);
        require(offset % 64 == 0 && offset >= previous_end &&
                    offset <= length && bytes <= length - offset &&
                    bytes == elements * width,
                "invalid tensor extent");
        previous_end = offset + bytes;
        auto result = tensors.emplace(
            std::string(reinterpret_cast<const char *>(p), name_len),
            Tensor{mapped + offset, type, std::move(shape), bytes});
        require(result.second, "duplicate tensor name");
      }
      require(previous_end == length, "unexpected trailing model bytes");
    } catch (...) {
      if (owns_mapping)
        munmap(mapped, length);
      mapped = nullptr;
      throw;
    }
  }
  ~Archive() {
    if (mapped && owns_mapping)
      munmap(mapped, length);
  }
  Archive(const Archive &) = delete;
  const Tensor &get(const std::string &name) const {
    auto it = tensors.find(name);
    require(it != tensors.end(), ("missing tensor " + name).c_str());
    return it->second;
  }
  bool has_packed_embedding() const {
    return tensors.count("decoder/embedding/packed64") != 0;
  }
  const Tensor &embedding() const {
    return get(has_packed_embedding() ? "decoder/embedding/packed64"
                                      : "decoder/embedding/quantized");
  }
  void expect(const std::string &name, uint32_t type,
              std::initializer_list<size_t> shape) const {
    const auto &tensor = get(name);
    require(tensor.type == type && tensor.shape == std::vector<size_t>(shape),
            ("unexpected tensor schema: " + name).c_str());
    if (type == 1)
      for (size_t i = 0; i < tensor.bytes / 4; ++i)
        require(std::isfinite(tensor.floats()[i]), "nonfinite model metadata");
  }
  void validate() const {
    require(tensors.size() == 1230, "unexpected Medium tensor count");
    auto matrix = [&](const std::string &name, size_t k, size_t n) {
      expect(name + "/weight", 3, {k, n});
      expect(name + "/bias", 1, {n});
      expect(name + "/scale", 1, {n});
      expect(name + "/zero", 2, {n});
      expect(name + "/sums", 6, {n});
      const auto *sums =
          reinterpret_cast<const int32_t *>(get(name + "/sums").data);
      for (size_t i = 0; i < n; ++i)
        require(sums[i] >= -128 * int(k) && sums[i] <= 127 * int(k),
                "packed sum range");
    };
    for (size_t layer = 0; layer < 14; ++layer) {
      const size_t ek[]{768, 768, 768, 640, 768, 3072},
          en[]{640, 640, 640, 768, 3072, 768};
      const size_t dk[]{640, 640, 640, 640, 640, 640, 640, 2560},
          dn[]{640, 640, 640, 640, 640, 640, 5120, 640};
      for (size_t i = 0; i < 6; ++i)
        matrix("encoder/linear/" + std::to_string(layer * 6 + i), ek[i], en[i]);
      for (size_t i = 0; i < 8; ++i)
        matrix("decoder/linear/" + std::to_string(layer * 8 + i), dk[i], dn[i]);
      for (size_t i = 0; i < 2; ++i)
        matrix("cross/" + std::to_string(layer * 2 + i), 640, 640);
    }
    matrix("adapter/projection", 768, 640);
    for (size_t i = 0; i < 29; ++i)
      expect("encoder/norm/" + std::to_string(i), 1, {768});
    for (size_t i = 0; i < 43; ++i)
      expect("decoder/norm/" + std::to_string(i), 1, {640});
    expect("decoder/rotary", 1, {16});
    for (const auto &name : {"decoder/embedding", "adapter/positions"}) {
      expect(std::string(name) + "/scale", 1, {1});
      expect(std::string(name) + "/zero_point", 2, {1});
      require(get(std::string(name) + "/scale").floats()[0] > 0,
              "invalid embedding scale");
    }
    if (has_packed_embedding())
      expect("decoder/embedding/packed64", 2, {512, 5, 512, 16});
    else
      expect("decoder/embedding/quantized", 2, {32768, 640});
    expect("adapter/positions/quantized", 2, {4096, 768});
    expect("frontend/linear/quantized", 3, {80, 768});
    for (const auto &part : {"scale", "zero_point"})
      expect(std::string("frontend/linear/") + part, 1, {1, 768});
    for (size_t i = 1; i <= 2; ++i) {
      const auto name = "frontend/conv" + std::to_string(i);
      const size_t out = i == 1 ? 1536 : 768, in = i == 1 ? 768 : 1536;
      expect(name + "/quantized", 3, {out, in, 5});
      expect(name + "/bias", 1, {out});
      expect(name + "/scale", 1, {out, 1, 1});
      expect(name + "/zero_point", 1, {out, 1, 1});
    }
    expect("vad/stft.forward_basis_buffer", 1, {258, 1, 256});
    const size_t vi[]{129, 128, 64, 64}, vo[]{128, 64, 64, 128};
    for (size_t i = 0; i < 4; ++i) {
      const auto name = "vad/encoder." + std::to_string(i) + ".reparam_conv.";
      expect(name + "weight", 1, {vo[i], vi[i], 3});
      expect(name + "bias", 1, {vo[i]});
    }
    for (const auto &name : {"ih", "hh"}) {
      expect(std::string("vad/decoder.rnn.weight_") + name, 1, {512, 128});
      expect(std::string("vad/decoder.rnn.bias_") + name, 1, {512});
    }
    expect("vad/decoder.decoder.2.weight", 1, {1, 128, 1});
    expect("vad/decoder.decoder.2.bias", 1, {1});
  }
};
struct Matrix {
  std::string name;
  std::unique_ptr<Projection> projection;
#if defined(__x86_64__)
  std::unique_ptr<transcribro::X86Projection> baseline;
#endif
  Matrix(const Archive &archive, const std::string &name) : name(name) {
    const auto &weight = archive.get(name + "/weight");
    require(weight.type == 3 && weight.shape.size() == 2,
            "invalid packed matrix");
    const size_t n = weight.shape[1];
    const auto &sum = archive.get(name + "/sums"),
               &scale = archive.get(name + "/scale"),
               &zero = archive.get(name + "/zero"),
               &bias = archive.get(name + "/bias");
    require(sum.type == 6 && sum.bytes == n * 4 && scale.bytes == n * 4 &&
                zero.type == 2 && zero.bytes == n && bias.bytes == n * 4,
            "packed matrix metadata mismatch");
    projection = std::make_unique<Projection>(
        weight.shape[0], n, reinterpret_cast<const int8_t *>(weight.data),
        reinterpret_cast<const int32_t *>(sum.data), scale.floats(), zero.data,
        bias.floats(), 8);
#if defined(__x86_64__)
    if (!__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("avx512vbmi"))
      baseline = std::make_unique<transcribro::X86Projection>(
          weight.shape[0], n, reinterpret_cast<const int8_t *>(weight.data),
          scale.floats(), zero.data, bias.floats());
#endif
  }
  void run(const float *x, size_t rows, Vec &out, Quantized &scratch) const {
    trace(name + "/input", x, rows * projection->input_size());
    transcribro::research::quantize(x, rows, projection->input_size(), scratch,
                                    tiled_activations_available());
    run(scratch, rows, out);
  }
  void run(const Quantized &q, size_t rows, Vec &out) const {
    ProfileScope profile(ProfileStage::Projection);
    out.resize(rows * projection->output_size());
#if defined(__x86_64__)
    if (baseline)
      baseline->multiply(q, rows, out.data(), q.unsigned_activations, q.integer_accumulators);
    else
#endif
      projection->multiply(q, rows, out.data());
    trace(name + "/output", out.data(), out.size());
  }
};
void gemm(size_t m, size_t n, size_t k, const float *a, const float *b,
          float *c, bool transpose_b = false, float scale = 1.0f) {
  ProfileScope profile(ProfileStage::Fp32Gemm);
  if (!m || !n)
    return;
  MlasGemm(CblasNoTrans, transpose_b ? CblasTrans : CblasNoTrans, m, n, k,
           scale, a, k, b, transpose_b ? k : n, 0.0f, c, n, nullptr);
}
void normalize(const Vec &x, size_t rows, size_t width, const float *gain,
               Vec &out, [[maybe_unused]] bool skip = false) {
  ProfileScope profile(ProfileStage::Normalize);
  out.resize(x.size());
  for (size_t r = 0; r < rows; ++r) {
    const float *source = x.data() + r * width;
    float sum = 0, squares = 0;
    for (size_t i = 0; i < width; ++i) {
      sum += source[i];
#if defined(__aarch64__)
      // The reference ARM layer-normalization kernel contracts this update.
      // Keep that choice explicit; independent graph operations must not fuse.
      // SkipLayerNormalization is compiled without that contraction.
      squares = skip ? squares + source[i] * source[i]
                     : std::fma(source[i], source[i], squares);
#else
      squares += source[i] * source[i];
#endif
    }
    const float mean = sum / float(width);
#if defined(__aarch64__)
    const float variance = skip ? squares / float(width) - mean * mean
                                : std::fma(-mean, mean, squares / float(width));
#else
    const float variance = squares / float(width) - mean * mean;
#endif
    const float deviation = std::sqrt(variance + 1.0e-5f);
    for (size_t i = 0; i < width; ++i)
      out[r * width + i] = ((source[i] - mean) / deviation) * gain[i];
  }
}
void add(Vec &x, const Vec &y) {
  require(x.size() == y.size(), "residual shape mismatch");
  for (size_t i = 0; i < x.size(); ++i)
    x[i] += y[i];
}
void heads(const Vec &x, size_t rows, Vec &out) {
  require(x.size() == rows * 640, "attention projection shape mismatch");
  out.resize(x.size());
  for (size_t h = 0; h < 10; ++h)
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(x.data() + r * 640 + h * 64, 64,
                  out.data() + (h * rows + r) * 64);
}
void unheads(const Vec &x, size_t rows, Vec &out) {
  out.resize(x.size());
  for (size_t h = 0; h < 10; ++h)
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(x.data() + (h * rows + r) * 64, 64,
                  out.data() + r * 640 + h * 64);
}
void logistic(Vec &x) { MlasComputeLogistic(x.data(), x.data(), x.size()); }
void tanh_values(float *x, size_t count) { MlasComputeTanh(x, x, count); }
void silu(Vec &x, Vec &scratch) {
  scratch = x;
  logistic(scratch);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] *= scratch[i];
}
void conv1d(const Vec &x, size_t channels, size_t length, const float *weight,
            const float *bias, size_t filters, size_t kernel, size_t stride,
            size_t pad, Vec &out, Vec &scratch) {
  ProfileScope profile(ProfileStage::Fp32Conv);
  const size_t count = (length + 2 * pad - kernel) / stride + 1;
  out.resize(filters * count);
  MLAS_CONV_PARAMETERS params;
  MLAS_ACTIVATION activation{};
  activation.ActivationKind = MlasIdentityActivation;
  const int64_t input_shape[]{int64_t(length)}, kernel_shape[]{int64_t(kernel)},
      dilation[]{1}, padding[]{int64_t(pad), int64_t(pad)},
      strides[]{int64_t(stride)}, output_shape[]{int64_t(count)};
  size_t workspace;
  MlasConvPrepare(&params, 1, 1, 1, channels, input_shape, kernel_shape,
                  dilation, padding, strides, output_shape, filters,
                  &activation, &workspace, 0, nullptr);
  scratch.resize(workspace);
  MlasConv(&params, x.data(), weight, bias, scratch.data(), out.data(),
           nullptr);
}
void attention(const Vec &queries, const float *keys, const float *values,
               size_t rows, size_t memory, float scale, int mask, size_t past,
               Vec &out, Vec &scores, size_t head_stride = 0) {
  ProfileScope profile(ProfileStage::Attention);
  require(memory > 0, "empty attention memory");
  if (!head_stride)
    head_stride = memory * 64;
  out.resize(rows * 640);
#if defined(__aarch64__)
  if (rows >= 64 && memory == rows && past == 0 && (mask == 2 || mask == 3)) {
    for (size_t h = 0; h < 10; ++h)
      transcribro::standalone::local_attention_head(
          queries.data() + h * rows * 64, keys + h * head_stride,
          values + h * head_stride, rows, mask, scale,
          out.data() + h * rows * 64, scores);
    return;
  }
#endif
  scores.resize(rows * memory);
  for (size_t h = 0; h < 10; ++h) {
    gemm(rows, memory, 64, queries.data() + h * rows * 64,
         keys + h * head_stride, scores.data(), true, scale);
    for (size_t r = 0; r < rows; ++r) {
      float *s = scores.data() + r * memory;
      float maximum = -INFINITY;
      for (size_t c = 0; c < memory; ++c) {
        bool allowed = true;
        if (mask == 1)
          allowed = c <= past + r;
        if (mask == 2)
          allowed = int64_t(r) - int64_t(c) >= -4 && r <= c + 16;
        if (mask == 3)
          allowed = r >= c && r <= c + 16;
        s[c] = allowed ? s[c] : -INFINITY;
        maximum = std::max(maximum, s[c]);
      }
      MlasComputeSoftmax(s, s, 1, memory, false, false, 0.0f, nullptr);
    }
    gemm(rows, 64, memory, scores.data(), values + h * head_stride,
         out.data() + h * rows * 64);
  }
}
struct Workspace {
  Vec hidden, normalized, q, k, v, qheads, kheads, vheads, attended, scores,
      joined, projected, expanded, gated, activation;
  Quantized quantized;
};
struct DecoderCache {
  static constexpr size_t capacity = 257;
  size_t count = 0;
  Vec keys = Vec(14 * 640 * capacity), values = Vec(14 * 640 * capacity);
  transcribro::standalone::TokenSelection selection;
};
class Medium {
  Archive archive;
  std::vector<Matrix> encoder_matrices, decoder_matrices, cross_matrices;
  Matrix adapter_matrix;
  std::vector<const float *> encoder_norms, decoder_norms;
  Vec frontend_linear, frontend_conv1, frontend_conv2;
#if defined(__aarch64__)
  std::unique_ptr<transcribro::standalone::CompactHead<false>> head;
#else
  Vec head;
#endif
  Workspace ws;

public:
  std::mutex mutex;
  explicit Medium(const char *path, const void *bytes = nullptr,
                  size_t size = 0)
      : archive(path, bytes, size),
        adapter_matrix(archive, "adapter/projection") {
    ProfileScope load_profile(ProfileStage::LoadModel);
    archive.validate();
    for (size_t i = 0; i < 84; ++i)
      encoder_matrices.emplace_back(archive,
                                    "encoder/linear/" + std::to_string(i));
    for (size_t i = 0; i < 112; ++i)
      decoder_matrices.emplace_back(archive,
                                    "decoder/linear/" + std::to_string(i));
    for (size_t i = 0; i < 28; ++i)
      cross_matrices.emplace_back(archive, "cross/" + std::to_string(i));
    for (size_t i = 0; i < 29; ++i) {
      const auto &t = archive.get("encoder/norm/" + std::to_string(i));
      require(t.bytes == 768 * 4, "encoder gain shape");
      encoder_norms.push_back(t.floats());
    }
    for (size_t i = 0; i < 43; ++i) {
      const auto &t = archive.get("decoder/norm/" + std::to_string(i));
      require(t.bytes == 640 * 4, "decoder gain shape");
      decoder_norms.push_back(t.floats());
    }
    const auto &e = archive.embedding();
    require(e.type == 2 && e.bytes == 32768 * 640, "embedding shape");
    float scale = archive.get("decoder/embedding/scale").floats()[0];
    int zero = archive.get("decoder/embedding/zero_point").data[0];
#if defined(__aarch64__)
    if (archive.has_packed_embedding())
      head = std::make_unique<transcribro::standalone::CompactHead<false>>(
          e.data, scale, zero);
    else
      head = std::make_unique<transcribro::standalone::CompactHead<false>>(
          [&](size_t j, size_t v) { return embedding_code(e, v, j); }, scale,
          zero);
#else
    head.resize(32768 * 640);
    // Keep 128 vocabulary columns contiguous through all 640 input values.
    // This retains MLAS's original accumulation order and avoids repeatedly
    // walking the entire matrix for each small output group during decoding.
    for (size_t base_v = 0; base_v < 32768; base_v += 128)
      for (size_t base_j = 0; base_j < 640; base_j += 16)
        for (size_t j = base_j; j < base_j + 16; ++j)
          for (size_t v = base_v; v < base_v + 128; ++v)
            head[base_v * 640 + j * 128 + v - base_v] =
                float(int(embedding_code(e, v, j)) - zero) * scale;
#endif
    ProfileScope frontend_load_profile(ProfileStage::LoadFrontendWeights);
    for (auto entry : {std::pair{"linear", &frontend_linear},
                       {"conv1", &frontend_conv1},
                       {"conv2", &frontend_conv2}}) {
      const std::string name = std::string("frontend/") + entry.first;
      const auto &q = archive.get(name + "/quantized");
      const float *s = archive.get(name + "/scale").floats();
      const float *offset = archive.get(name + "/zero_point").floats();
      require(q.type == 3, "frontend weight type");
      entry.second->resize(q.bytes);
      const auto *codes = reinterpret_cast<const int8_t *>(q.data);
      auto *output = entry.second->data();
      // Convolution coefficients for one output channel are contiguous.
      // Hoist channel lookup/division out of the coefficient loop; separate
      // FP32 multiply and add still reconstruct the exact original values.
      if (name == "frontend/linear") {
        for (size_t r = 0; r < 80; ++r)
          for (size_t c = 0; c < 768; ++c)
            output[r * 768 + c] = float(codes[r * 768 + c]) * s[c] + offset[c];
      } else {
        const size_t channel_size = q.shape[1] * q.shape[2];
        for (size_t c = 0; c < q.shape[0]; ++c)
          for (size_t i = 0; i < channel_size; ++i)
            output[c * channel_size + i] =
                float(codes[c * channel_size + i]) * s[c] + offset[c];
      }
    }
  }
  void frontend(const float *audio, size_t samples, tm_frontend_state &state,
                float *output, size_t capacity, size_t *frames) {
    ProfileScope profile(ProfileStage::Frontend);
    require(audio && output && frames && samples >= 640 && samples <= 320000,
            "frontend chunk extent");
    require(state.sample_len >= 0 && state.sample_len <= 79 &&
                state.frame_count >= 0 && state.frame_count <= INT64_MAX - 4096,
            "frontend state extent");
    const size_t count = (samples + size_t(state.sample_len)) / 80;
    const size_t first = (count + 1) / 2, second = (first + 1) / 2;
    require(capacity >= second, "frontend output capacity");
    ws.hidden.resize(count * 80);
    for (size_t i = 0; i < ws.hidden.size(); ++i) {
      const float x = i < size_t(state.sample_len)
                          ? state.sample_buffer[i]
                          : audio[i - size_t(state.sample_len)];
      require(std::isfinite(x), "nonfinite audio");
      ws.hidden[i] = x;
    }
    for (size_t r = 0; r < count; ++r) {
      float *x = ws.hidden.data() + r * 80;
      const float mean = Eigen::Map<const Eigen::VectorXf>(x, 80).sum() / 80.0f;
      std::array<float, 80> squares;
      for (size_t j = 0; j < 80; ++j) {
        x[j] -= mean;
        squares[j] = x[j] * x[j];
      }
      const float variance =
          Eigen::Map<const Eigen::VectorXf>(squares.data(), 80).sum() / 80.0f;
      const float stddev = std::sqrt(variance + 1.0e-6f);
      for (size_t j = 0; j < 80; ++j)
        x[j] = std::asinh((x[j] / stddev) * 0.5526678562164307f);
    }
    ws.projected.resize(count * 768);
    gemm(count, 768, 80, ws.hidden.data(), frontend_linear.data(),
         ws.projected.data());
    silu(ws.projected, ws.activation);
    ws.joined.resize(768 * (count + 4));
    for (size_t c = 0; c < 768; ++c) {
      std::copy_n(state.conv1 + c * 4, 4, ws.joined.data() + c * (count + 4));
      for (size_t t = 0; t < count; ++t)
        ws.joined[c * (count + 4) + 4 + t] = ws.projected[t * 768 + c];
    }
    conv1d(ws.joined, 768, count + 4, frontend_conv1.data(),
           archive.get("frontend/conv1/bias").floats(), 1536, 5, 2, 0,
           ws.expanded, ws.scores);
    silu(ws.expanded, ws.activation);
    ws.hidden.resize(1536 * (first + 4));
    for (size_t c = 0; c < 1536; ++c) {
      std::copy_n(state.conv2 + c * 4, 4, ws.hidden.data() + c * (first + 4));
      std::copy_n(ws.expanded.data() + c * first, first,
                  ws.hidden.data() + c * (first + 4) + 4);
    }
    conv1d(ws.hidden, 1536, first + 4, frontend_conv2.data(),
           archive.get("frontend/conv2/bias").floats(), 768, 5, 2, 0, ws.gated,
           ws.scores);
    for (size_t t = 0; t < second; ++t)
      for (size_t c = 0; c < 768; ++c)
        output[t * 768 + c] = ws.gated[c * second + t];
    for (size_t c = 0; c < 768; ++c)
      std::copy_n(ws.joined.data() + c * (count + 4) + count, 4,
                  state.conv1 + c * 4);
    for (size_t c = 0; c < 1536; ++c)
      std::copy_n(ws.hidden.data() + c * (first + 4) + first, 4,
                  state.conv2 + c * 4);
    std::fill_n(state.sample_buffer, 79, 0);
    state.sample_len = 0;
    state.frame_count += int64_t(count);
    *frames = second;
  }
  void vad(const float *audio, tm_vad_state &state, float *probability) {
    ProfileScope profile(ProfileStage::Vad);
    require(audio && probability, "null VAD input");
    ws.hidden.resize(640);
    std::copy_n(state.context, 64, ws.hidden.data());
    std::copy_n(audio, 512, ws.hidden.data() + 64);
    for (size_t i = 0; i < 576; ++i)
      require(std::isfinite(ws.hidden[i]), "nonfinite VAD audio");
    for (size_t i = 0; i < 64; ++i)
      ws.hidden[576 + i] = ws.hidden[574 - i];
    conv1d(ws.hidden, 1, 640,
           archive.get("vad/stft.forward_basis_buffer").floats(), nullptr, 258,
           256, 128, 0, ws.projected, ws.scores);
    ws.hidden.resize(129 * 4);
    for (size_t i = 0; i < 129 * 4; ++i) {
      const float a = ws.projected[i], b = ws.projected[129 * 4 + i];
      ws.hidden[i] = std::sqrt(a * a + b * b);
    }
    size_t channels = 129, length = 4;
    for (size_t i = 0; i < 4; ++i) {
      const std::string prefix =
          "vad/encoder." + std::to_string(i) + ".reparam_conv.";
      const size_t filters = i == 0 || i == 3 ? 128 : 64;
      const size_t stride = i == 1 || i == 2 ? 2 : 1;
      conv1d(ws.hidden, channels, length,
             archive.get(prefix + "weight").floats(),
             archive.get(prefix + "bias").floats(), filters, 3, stride, 1,
             ws.projected, ws.scores);
      for (auto &value : ws.projected)
        value = std::max(value, 0.0f);
      channels = filters;
      length = (length - 1) / stride + 1;
      ws.hidden.swap(ws.projected);
    }
    require(length == 1 && ws.hidden.size() == 128, "VAD encoder shape");
    ws.gated.resize(512);
    gemm(1, 512, 128, ws.hidden.data(),
         archive.get("vad/decoder.rnn.weight_ih").floats(), ws.gated.data(),
         true);
    MlasGemm(CblasNoTrans, CblasTrans, 1, 512, 128, 1.0f, state.hidden, 128,
             archive.get("vad/decoder.rnn.weight_hh").floats(), 128, 1.0f,
             ws.gated.data(), 512, nullptr);
    const float *input_bias = archive.get("vad/decoder.rnn.bias_ih").floats();
    const float *hidden_bias = archive.get("vad/decoder.rnn.bias_hh").floats();
    for (size_t i = 0; i < 512; ++i)
      ws.gated[i] += input_bias[i] + hidden_bias[i];
    tanh_values(ws.gated.data() + 256, 128);
    ws.activation.resize(384);
    std::copy_n(ws.gated.data(), 256, ws.activation.data());
    std::copy_n(ws.gated.data() + 384, 128, ws.activation.data() + 256);
    logistic(ws.activation);
    for (size_t i = 0; i < 128; ++i) {
#if defined(__aarch64__)
      state.cell[i] = std::fma(state.cell[i], ws.activation[128 + i],
                               ws.activation[i] * ws.gated[256 + i]);
#else
      state.cell[i] = state.cell[i] * ws.activation[128 + i] +
                      ws.activation[i] * ws.gated[256 + i];
#endif
    }
    std::copy_n(state.cell, 128, state.hidden);
    tanh_values(state.hidden, 128);
    ws.hidden.resize(128);
    for (size_t i = 0; i < 128; ++i) {
      state.hidden[i] *= ws.activation[256 + i];
      ws.hidden[i] = std::max(state.hidden[i], 0.0f);
    }
    conv1d(ws.hidden, 128, 1,
           archive.get("vad/decoder.decoder.2.weight").floats(),
           archive.get("vad/decoder.decoder.2.bias").floats(), 1, 1, 1, 0,
           ws.projected, ws.scores);
    logistic(ws.projected);
    *probability = ws.projected[0];
    std::copy_n(audio + 448, 64, state.context);
  }
  void encoder(const float *input, size_t rows, float *output) {
    ProfileScope profile(ProfileStage::Encoder);
    require(input && output, "null encoder buffer");
    require(rows > 0 && rows <= 1024, "encoder frame limit");
    ws.hidden.assign(input, input + rows * 768);
    for (size_t layer = 0; layer < 14; ++layer) {
      normalize(ws.hidden, rows, 768, encoder_norms[layer * 2], ws.normalized);
      trace("encoder/norm/" + std::to_string(layer * 2), ws.normalized.data(),
            ws.normalized.size());
      transcribro::research::quantize(ws.normalized.data(), rows, 768,
                                      ws.quantized,
                                      tiled_activations_available());
      encoder_matrices[layer * 6].run(ws.quantized, rows, ws.q);
      encoder_matrices[layer * 6 + 1].run(ws.quantized, rows, ws.k);
      encoder_matrices[layer * 6 + 2].run(ws.quantized, rows, ws.v);
      heads(ws.q, rows, ws.qheads);
      heads(ws.k, rows, ws.kheads);
      heads(ws.v, rows, ws.vheads);
      attention(ws.qheads, ws.kheads.data(), ws.vheads.data(), rows, rows,
                0.1249999925494194f, layer >= 2 && layer <= 11 ? 3 : 2, 0,
                ws.attended, ws.scores);
      unheads(ws.attended, rows, ws.joined);
      encoder_matrices[layer * 6 + 3].run(ws.joined.data(), rows, ws.projected,
                                          ws.quantized);
      add(ws.hidden, ws.projected);
      normalize(ws.hidden, rows, 768, encoder_norms[layer * 2 + 1],
                ws.normalized);
      encoder_matrices[layer * 6 + 4].run(ws.normalized.data(), rows,
                                          ws.expanded, ws.quantized);
      ws.activation.resize(ws.expanded.size());
      for (size_t i = 0; i < ws.expanded.size(); ++i)
        ws.activation[i] = ws.expanded[i] / 1.4142135381698608f;
      MlasComputeErf(ws.activation.data(), ws.activation.data(),
                     ws.activation.size());
      for (size_t i = 0; i < ws.expanded.size(); ++i)
        ws.expanded[i] *= 0.5f * (ws.activation[i] + 1.0f);
      encoder_matrices[layer * 6 + 5].run(ws.expanded.data(), rows,
                                          ws.projected, ws.quantized);
      add(ws.hidden, ws.projected);
    }
    normalize(ws.hidden, rows, 768, encoder_norms.back(), ws.normalized, true);
    std::copy(ws.normalized.begin(), ws.normalized.end(), output);
  }
  void adapter(const float *input, size_t rows, size_t offset, float *output) {
    ProfileScope profile(ProfileStage::Adapter);
    require(input && output, "null adapter buffer");
    require(rows > 0 && rows <= 1024 && offset <= 4096 && rows <= 4096 - offset,
            "adapter extent");
    const auto &p = archive.get("adapter/positions/quantized");
    require(p.bytes == 4096 * 768 && p.type == 2, "position table shape");
    const float scale = archive.get("adapter/positions/scale").floats()[0];
    const int zero = archive.get("adapter/positions/zero_point").data[0];
    ws.hidden.resize(rows * 768);
    for (size_t i = 0; i < ws.hidden.size(); ++i)
      ws.hidden[i] =
          input[i] + float(int(p.data[offset * 768 + i]) - zero) * scale;
    adapter_matrix.run(ws.hidden.data(), rows, ws.projected, ws.quantized);
    std::copy(ws.projected.begin(), ws.projected.end(), output);
  }
  void cross(const float *input, size_t rows, float *keys, float *values) {
    ProfileScope profile(ProfileStage::Cross);
    require(input && keys && values, "null cross-attention buffer");
    require(rows > 0 && rows <= 1024, "cross-attention frame limit");
    transcribro::research::quantize(input, rows, 640, ws.quantized,
                                    tiled_activations_available());
    for (size_t layer = 0; layer < 14; ++layer) {
      cross_matrices[layer * 2].run(ws.quantized, rows, ws.k);
      cross_matrices[layer * 2 + 1].run(ws.quantized, rows, ws.v);
      heads(ws.k, rows, ws.kheads);
      heads(ws.v, rows, ws.vheads);
      std::copy(ws.kheads.begin(), ws.kheads.end(), keys + layer * rows * 640);
      std::copy(ws.vheads.begin(), ws.vheads.end(),
                values + layer * rows * 640);
    }
  }
  void decoder(const int64_t *tokens, size_t rows, const float *old_k,
               const float *old_v, size_t past, const float *cross_k,
               const float *cross_v, size_t memory, float *logits, float *out_k,
               float *out_v, DecoderCache *cache = nullptr,
               int64_t *predictions = nullptr) {
    ProfileScope profile(ProfileStage::Decoder);
    require(tokens && cross_k && cross_v &&
                (logits || (cache && predictions)) &&
                (cache || (out_k && out_v && (!past || (old_k && old_v)))),
            "null decoder buffer");
    require(rows > 0 && rows <= 1024 && past <= 4096 && rows <= 4096 - past &&
                memory > 0 && memory <= 1024,
            "decoder extent");
    require(!cache || (past <= DecoderCache::capacity &&
                       rows <= DecoderCache::capacity - past),
            "decoder cache limit");
    const float *frequency = archive.get("decoder/rotary").floats();
    Vec cosines(rows * 32), sines(rows * 32);
    for (size_t r = 0; r < rows; ++r)
      for (size_t i = 0; i < 32; ++i)
        cosines[r * 32 + i] = float(past + r) * frequency[i / 2];
    Eigen::Map<Eigen::ArrayXf>(sines.data(), sines.size()) =
        Eigen::Map<const Eigen::ArrayXf>(cosines.data(), cosines.size()).sin();
    Eigen::Map<Eigen::ArrayXf>(cosines.data(), cosines.size()) =
        Eigen::Map<const Eigen::ArrayXf>(cosines.data(), cosines.size()).cos();
    const auto &embedding = archive.embedding();
    const float embedding_scale =
        archive.get("decoder/embedding/scale").floats()[0];
    const int embedding_zero =
        archive.get("decoder/embedding/zero_point").data[0];
    ws.hidden.resize(rows * 640);
    for (size_t r = 0; r < rows; ++r) {
      require(tokens[r] >= 0 && tokens[r] < 32768, "invalid token");
      for (size_t j = 0; j < 640; ++j)
        ws.hidden[r * 640 + j] =
            float(int(embedding_code(embedding, tokens[r], j)) -
                  embedding_zero) *
            embedding_scale;
    }
    for (size_t layer = 0; layer < 14; ++layer) {
      normalize(ws.hidden, rows, 640, decoder_norms[layer * 3], ws.normalized);
      trace("decoder/norm/" + std::to_string(layer * 3), ws.normalized.data(),
            ws.normalized.size());
      transcribro::research::quantize(ws.normalized.data(), rows, 640,
                                      ws.quantized,
                                      tiled_activations_available());
      decoder_matrices[layer * 8].run(ws.quantized, rows, ws.q);
      decoder_matrices[layer * 8 + 1].run(ws.quantized, rows, ws.k);
      decoder_matrices[layer * 8 + 2].run(ws.quantized, rows, ws.v);
      heads(ws.q, rows, ws.qheads);
      heads(ws.k, rows, ws.kheads);
      heads(ws.v, rows, ws.vheads);
      for (size_t h = 0; h < 10; ++h)
        for (size_t r = 0; r < rows; ++r)
          for (size_t pair = 0; pair < 16; ++pair) {
            const float co = cosines[r * 32 + pair * 2],
                        si = sines[r * 32 + pair * 2];
            const size_t at = (h * rows + r) * 64 + pair * 2;
            for (Vec *v : {&ws.qheads, &ws.kheads}) {
              const float a = (*v)[at], b = (*v)[at + 1];
              (*v)[at] = a * co + (-b) * si;
              (*v)[at + 1] = b * co + a * si;
            }
          }
      const size_t total = past + rows;
      const size_t stride = cache ? DecoderCache::capacity : total;
      float *key = cache ? cache->keys.data() + layer * stride * 640
                         : out_k + layer * total * 640;
      float *value = cache ? cache->values.data() + layer * stride * 640
                           : out_v + layer * total * 640;
      for (size_t h = 0; h < 10; ++h) {
        if (past && !cache) {
          std::copy_n(old_k + (layer * 10 + h) * past * 64, past * 64,
                      key + h * total * 64);
          std::copy_n(old_v + (layer * 10 + h) * past * 64, past * 64,
                      value + h * total * 64);
        }
        std::copy_n(ws.kheads.data() + h * rows * 64, rows * 64,
                    key + (h * stride + past) * 64);
        std::copy_n(ws.vheads.data() + h * rows * 64, rows * 64,
                    value + (h * stride + past) * 64);
      }
      attention(ws.qheads, key, value, rows, total, 0.125f, 1, past,
                ws.attended, ws.scores, stride * 64);
      unheads(ws.attended, rows, ws.joined);
      decoder_matrices[layer * 8 + 3].run(ws.joined.data(), rows, ws.projected,
                                          ws.quantized);
      add(ws.hidden, ws.projected);
      normalize(ws.hidden, rows, 640, decoder_norms[layer * 3 + 1],
                ws.normalized);
      decoder_matrices[layer * 8 + 4].run(ws.normalized.data(), rows, ws.q,
                                          ws.quantized);
      heads(ws.q, rows, ws.qheads);
      attention(ws.qheads, cross_k + layer * memory * 640,
                cross_v + layer * memory * 640, rows, memory, 0.125f, 0, 0,
                ws.attended, ws.scores);
      unheads(ws.attended, rows, ws.joined);
      decoder_matrices[layer * 8 + 5].run(ws.joined.data(), rows, ws.projected,
                                          ws.quantized);
      add(ws.hidden, ws.projected);
      normalize(ws.hidden, rows, 640, decoder_norms[layer * 3 + 2],
                ws.normalized);
      decoder_matrices[layer * 8 + 6].run(ws.normalized.data(), rows,
                                          ws.expanded, ws.quantized);
      const size_t width = ws.expanded.size() / rows / 2;
      ws.gated.resize(rows * width);
      ws.activation.resize(rows * width);
      for (size_t r = 0; r < rows; ++r)
        std::copy_n(ws.expanded.data() + r * width * 2 + width, width,
                    ws.activation.data() + r * width);
      MlasComputeLogistic(ws.activation.data(), ws.activation.data(),
                          ws.activation.size());
      for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < width; ++j) {
          const float x = ws.expanded[r * width * 2 + width + j];
          const float silu = x * ws.activation[r * width + j];
          ws.gated[r * width + j] = ws.expanded[r * width * 2 + j] * silu;
        }
      decoder_matrices[layer * 8 + 7].run(ws.gated.data(), rows, ws.projected,
                                          ws.quantized);
      add(ws.hidden, ws.projected);
    }
    normalize(ws.hidden, rows, 640, decoder_norms.back(), ws.normalized, true);
    ProfileScope head_profile(ProfileStage::OutputHead);
    ProfileScope head_size_profile(rows == 1   ? ProfileStage::HeadSingle
                                   : rows < 16 ? ProfileStage::HeadSmall
                                               : ProfileStage::HeadLarge);
#if defined(__aarch64__)
    if (predictions)
      cache->selection.run(
          rows, 32768, 64,
          [&](size_t column, float *block) {
            head->run_columns(ws.normalized.data(), rows, block, 64, column,
                              column + 64);
          },
          predictions);
    else
      head->run(ws.normalized.data(), rows, logits);
#else
    auto columns = [&](size_t column, float *output, size_t stride) {
      MlasGemm(CblasNoTrans, CblasNoTrans, rows, 128, 640, 1.0f,
               ws.normalized.data(), 640, head.data() + column * 640, 128, 0.0f,
               output, stride, nullptr);
    };
    if (predictions)
      cache->selection.run(
          rows, 32768, 128,
          [&](size_t column, float *block) { columns(column, block, 128); },
          predictions);
    else
      for (size_t column = 0; column < 32768; column += 128)
        columns(column, logits + column, 32768);
#endif
    if (cache)
      cache->count = past + rows;
  }
};
thread_local std::string last_error;
template <typename F> int call(void *handle, F function) noexcept {
  try {
    require(handle != nullptr, "null model");
    auto &model = *static_cast<Medium *>(handle);
    std::lock_guard<std::mutex> lock(model.mutex);
    function(model);
    last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return 1;
  }
}
} // namespace

extern "C" void *tm_create(const char *path, char *error, size_t bytes) {
  try {
    return new Medium(path);
  } catch (const std::exception &e) {
    if (error && bytes) {
      std::strncpy(error, e.what(), bytes - 1);
      error[bytes - 1] = 0;
    }
    return nullptr;
  }
}
extern "C" void *tm_create_memory(const void *archive, size_t size, char *error,
                                  size_t error_bytes) {
  try {
    require(archive, "null archive memory");
    return new Medium(nullptr, archive, size);
  } catch (const std::exception &e) {
    if (error && error_bytes) {
      std::strncpy(error, e.what(), error_bytes - 1);
      error[error_bytes - 1] = 0;
    }
    return nullptr;
  }
}
extern "C" void tm_destroy(void *model) { delete static_cast<Medium *>(model); }
extern "C" const char *tm_error() { return last_error.c_str(); }
extern "C" void *tm_cache_create() {
  try {
    return new DecoderCache;
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
extern "C" void tm_cache_destroy(void *cache) {
  delete static_cast<DecoderCache *>(cache);
}
extern "C" void tm_cache_reset(void *cache) {
  if (cache)
    static_cast<DecoderCache *>(cache)->count = 0;
}
extern "C" int tm_decoder_cached(void *model, void *cache_handle,
                                 const int64_t *tokens, size_t count,
                                 const float *cross_keys,
                                 const float *cross_values, size_t frames,
                                 int64_t *predictions) {
  return call(model, [&](Medium &m) {
    require(cache_handle && predictions && count > 0 &&
                count <= DecoderCache::capacity,
            "cached decoder input");
    auto &cache = *static_cast<DecoderCache *>(cache_handle);
    try {
      m.decoder(tokens, count, nullptr, nullptr, cache.count, cross_keys,
                cross_values, frames, nullptr, nullptr, nullptr, &cache,
                predictions);
    } catch (...) {
      cache.count = 0;
      throw;
    }
  });
}
extern "C" int tm_frontend(void *model, const float *audio, size_t samples,
                           tm_frontend_state *state, float *features,
                           size_t capacity, size_t *frames) {
  return call(model, [&](Medium &m) {
    require(state, "null frontend state");
    m.frontend(audio, samples, *state, features, capacity, frames);
  });
}
extern "C" int tm_vad(void *model, const float *audio, tm_vad_state *state,
                      float *probability) {
  return call(model, [&](Medium &m) {
    require(state, "null VAD state");
    m.vad(audio, *state, probability);
  });
}
#ifdef TRANSCRIBRO_TRACE
extern "C" void tm_set_trace(tm_trace_callback callback, void *context) {
  trace_callback = callback;
  trace_context = context;
}
#endif
extern "C" int tm_encoder(void *model, const float *input, size_t frames,
                          float *output) {
  return call(model, [&](Medium &m) { m.encoder(input, frames, output); });
}
extern "C" int tm_adapter(void *model, const float *input, size_t frames,
                          size_t offset, float *output) {
  return call(model,
              [&](Medium &m) { m.adapter(input, frames, offset, output); });
}
extern "C" int tm_cross(void *model, const float *input, size_t frames,
                        float *keys, float *values) {
  return call(model, [&](Medium &m) { m.cross(input, frames, keys, values); });
}
extern "C" int tm_decoder(void *model, const int64_t *tokens, size_t count,
                          const float *keys, const float *values, size_t past,
                          const float *cross_keys, const float *cross_values,
                          size_t frames, float *logits, float *out_keys,
                          float *out_values) {
  return call(model, [&](Medium &m) {
    m.decoder(tokens, count, keys, values, past, cross_keys, cross_values,
              frames, logits, out_keys, out_values);
  });
}
