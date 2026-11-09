#pragma once

#include <zstd.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace transcribro {
// A directly mapped asset can decode into its final allocation without the
// streaming input buffer. Require precisely one frame of the expected size.
inline void decompress_model_frame(const void* input, size_t input_size,
                                   void* output, size_t output_size) {
    if (ZSTD_getFrameContentSize(input, input_size) != output_size ||
        ZSTD_findFrameCompressedSize(input, input_size) != input_size ||
        ZSTD_decompress(output, output_size, input, input_size) != output_size)
        throw std::runtime_error("Invalid compressed speech model");
}

// Decode one bounded Zstandard frame while the container reader requests data.
// It retains only the codec window and input buffer, never a second whole model.
class ZstdReader {
public:
    explicit ZstdReader(std::function<size_t(void*, size_t)> source)
        : source_(std::move(source)), decoder_(ZSTD_createDStream(), ZSTD_freeDStream) {
        if (!decoder_) throw std::bad_alloc();
        check(ZSTD_initDStream(decoder_.get()));
        check(ZSTD_DCtx_setParameter(decoder_.get(), ZSTD_d_windowLogMax, 23));
    }

    void read(void* destination, size_t count) {
        ZSTD_outBuffer output{destination, count, 0};
        while (output.pos != output.size) {
            if (ended_) throw std::runtime_error("Truncated compressed speech model");
            step(output);
        }
    }

    void finish() {
        uint8_t extra = 0;
        while (!ended_) {
            ZSTD_outBuffer output{&extra, 1, 0};
            step(output);
            if (output.pos != 0) throw std::runtime_error("Trailing model container data");
        }
        if (input_.pos != input_.size || source_(&extra, 1) != 0)
            throw std::runtime_error("Trailing compressed model data");
    }

private:
    static void check(size_t result) {
        if (ZSTD_isError(result)) throw std::runtime_error("Invalid compressed speech model");
    }

    void step(ZSTD_outBuffer& output) {
        if (input_.pos == input_.size) {
            input_ = {buffer_.data(), source_(buffer_.data(), buffer_.size()), 0};
        }
        const auto previous_input = input_.pos;
        const auto previous_output = output.pos;
        const auto result = ZSTD_decompressStream(decoder_.get(), &output, &input_);
        check(result);
        ended_ = result == 0;
        if (!ended_ && input_.pos == previous_input && output.pos == previous_output)
            throw std::runtime_error("Truncated compressed speech model");
    }

    std::function<size_t(void*, size_t)> source_;
    std::unique_ptr<ZSTD_DStream, decltype(&ZSTD_freeDStream)> decoder_;
    std::array<uint8_t, 65536> buffer_{};
    ZSTD_inBuffer input_{nullptr, 0, 0};
    bool ended_ = false;
};
}  // namespace transcribro
