// Original Medium recognition with the fixed native runtime.
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <whisper.h>

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "medium/stream.h"
#include "zstd_reader.h"

namespace {
void check(int result) {
    if (result != 0) throw std::runtime_error("Streaming recognition failed");
}
struct Engine {
    void* stream = nullptr;
    bool previews = true;
    uint64_t delivered = 0;
    bool has_audio = false;
    size_t samples_since_analysis = 0;
    bool analysis_was_deferred = false;
    size_t analysis_interval_samples = 16000;
    whisper_vad_context* vad = nullptr;
    std::unique_ptr<void, decltype(&std::free)> model{nullptr, std::free};
    std::atomic<bool> cancelled{false};
    std::mutex pause_mutex;
    std::condition_variable resumed;
    bool paused = false;

    void new_stream() {
        check(ts_reset(stream, 1));
        delivered = 0;
        has_audio = false;
        samples_since_analysis = 0;
        analysis_was_deferred = false;
    }
    bool wait() {
        std::unique_lock<std::mutex> lock(pause_mutex);
        resumed.wait(lock, [this] { return !paused || cancelled.load(); });
        return !cancelled.load();
    }
    ~Engine() {
        if (stream) ts_destroy(stream);
        if (vad) whisper_vad_free(vad);
    }
};

void fail(JNIEnv* env, const char* message) {
    if (!env->ExceptionCheck())
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}
size_t read_asset(void* context, void* output, size_t length) {
    const int result = AAsset_read(static_cast<AAsset*>(context), output, length);
    return result > 0 ? static_cast<size_t>(result) : 0;
}
bool asset_eof(void* context) {
    return AAsset_getRemainingLength64(static_cast<AAsset*>(context)) == 0;
}
void keep_asset(void*) {}
using Asset = std::unique_ptr<AAsset, decltype(&AAsset_close)>;

void read_archive(AAsset* asset, void* output, size_t bytes) {
    off64_t start = 0, length = 0;
    const int fd = AAsset_openFileDescriptor64(asset, &start, &length);
    if (fd >= 0) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || start < 0 || length <= 0 || length > 512 * 1024 * 1024) {
            close(fd);
            throw std::runtime_error("Invalid model asset extent");
        }
        const size_t prefix = size_t(start % page), extent = size_t(length) + prefix;
        void* mapped = mmap(nullptr, extent, PROT_READ, MAP_PRIVATE, fd, start - prefix);
        close(fd);
        if (mapped != MAP_FAILED) {
            auto unmap = [extent](void* address) { munmap(address, extent); };
            std::unique_ptr<void, decltype(unmap)> mapping(mapped, unmap);
            transcribro::decompress_model_frame(static_cast<const char*>(mapped) + prefix,
                                               size_t(length), output, bytes);
            return;
        }
    }
    // Asset descriptors are unavailable for ZIP-compressed assets. Retain the
    // bounded streaming path for that case and for mapping failures.
    transcribro::ZstdReader reader([&](void* destination, size_t count) -> size_t {
        const int copied = AAsset_read(asset, destination, count);
        if (copied < 0) throw std::runtime_error("Cannot read model archive");
        return copied;
    });
    reader.read(output, bytes);
    reader.finish();
}

Engine* engine(jlong handle) {
    if (!handle) throw std::runtime_error("Speech engine is closed");
    return reinterpret_cast<Engine*>(handle);
}
void discard_log(ggml_log_level, const char*, void*) {}

jobjectArray strings(JNIEnv* env, const std::vector<std::string>& text) {
    // Byte arrays preserve standard UTF-8, including characters outside Java's modified UTF-8.
    auto output =
        env->NewObjectArray(static_cast<jsize>(text.size()), env->FindClass("[B"), nullptr);
    if (!output) return nullptr;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i].size() > 16000) throw std::runtime_error("Recognition result too long");
        auto item = env->NewByteArray(static_cast<jsize>(text[i].size()));
        if (!item) return nullptr;
        env->SetByteArrayRegion(item, 0, static_cast<jsize>(text[i].size()),
                                reinterpret_cast<const jbyte*>(text[i].data()));
        env->SetObjectArrayElement(output, static_cast<jsize>(i), item);
        env->DeleteLocalRef(item);
        if (env->ExceptionCheck()) return nullptr;
    }
    return output;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeCreate(
    JNIEnv* env, jobject, jobject assets, jboolean previews) {
    try {
        whisper_log_set(discard_log, nullptr);
        auto value = std::make_unique<Engine>();
        value->previews = previews;
        value->analysis_interval_samples = previews ? 16000 : 8000;
        auto* manager = AAssetManager_fromJava(env, assets);
        if (!manager) throw std::runtime_error("Cannot access speech models");
        constexpr size_t archive_size = 249585408;
        void* archive = nullptr;
        if (posix_memalign(&archive, 64, archive_size)) throw std::bad_alloc();
        value->model.reset(archive);
        {
            Asset compressed(
                AAssetManager_open(manager, "models/standalone/medium.trm.zst", AASSET_MODE_STREAMING),
                AAsset_close);
            if (!compressed) throw std::runtime_error("Standalone model is missing");
            read_archive(compressed.get(), archive, archive_size);
        }
        Asset tokenizer(
            AAssetManager_open(manager, "models/standalone/tokenizer.bin", AASSET_MODE_STREAMING),
            AAsset_close);
        if (!tokenizer || AAsset_getLength64(tokenizer.get()) != 249974)
            throw std::runtime_error("Invalid tokenizer");
        std::vector<char> tokens(249974);
        size_t at = 0;
        while (at < tokens.size()) {
            const int copied = AAsset_read(tokenizer.get(), tokens.data() + at, tokens.size() - at);
            if (copied <= 0) throw std::runtime_error("Tokenizer is truncated");
            at += size_t(copied);
        }
        char error[1024]{};
        value->stream = ts_create_memory(archive, archive_size, tokens.data(), tokens.size(), error,
                                         sizeof(error));
        if (!value->stream) throw std::runtime_error(error);
        value->new_stream();
        Asset vad(AAssetManager_open(manager, "models/silero_vad/ggml-silero-v6.2.0.bin",
                                     AASSET_MODE_STREAMING),
                  AAsset_close);
        if (!vad) throw std::runtime_error("Voice detector is missing");
        whisper_model_loader loader{vad.get(), read_asset, asset_eof, keep_asset};
        auto params = whisper_vad_default_context_params();
        params.use_gpu = false;
        params.n_threads = 1;
        value->vad = whisper_vad_init_with_params(&loader, params);
        if (!value->vad) throw std::runtime_error("Cannot load voice detector");
        whisper_vad_reset_state(value->vad);
        return reinterpret_cast<jlong>(value.release());
    } catch (const std::exception&) {
        fail(env, "Could not initialize on-device recognition");
        return 0;
    }
}

extern "C" JNIEXPORT jfloat JNICALL Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeVad(
    JNIEnv* env, jobject, jlong handle, jfloatArray input) {
    try {
        auto* value = engine(handle);
        if (env->GetArrayLength(input) != 512) throw std::runtime_error("Invalid audio frame");
        float samples[512];
        env->GetFloatArrayRegion(input, 0, 512, samples);
        if (env->ExceptionCheck()) return 0;
        if (!whisper_vad_detect_speech_no_reset(value->vad, samples, 512) ||
            whisper_vad_n_probs(value->vad) != 1)
            throw std::runtime_error("Voice detection failed");
        const float probability = whisper_vad_probs(value->vad)[0];
        if (!std::isfinite(probability)) throw std::runtime_error("Invalid voice probability");
        return probability;
    } catch (const std::exception&) {
        fail(env, "Voice detection failed");
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeStream(JNIEnv* env, jobject, jlong handle,
                                                                 jfloatArray input,
                                                                 jboolean finish_phrase,
                                                                 jboolean defer_analysis) {
    try {
        auto* value = engine(handle);
        if (!value->wait()) return strings(env, {""});
        const jsize count = env->GetArrayLength(input);
        if (count < 0 || count > 8192) throw std::runtime_error("Invalid audio chunk");
        std::vector<float> samples(count);
        if (count) {
            env->GetFloatArrayRegion(input, 0, count, samples.data());
            if (env->ExceptionCheck()) return nullptr;
            check(ts_push(value->stream, samples.data(), count));
            value->has_audio = true;
            value->samples_since_analysis += count;
        }
        // When capture is ahead, combine old preview work without dropping any PCM.
        // Final/manual boundaries always decode. Bound deferred audio to 2.048 seconds
        // even if the consumer cannot catch up; the outer queue still owns overload.
        if (defer_analysis) value->analysis_was_deferred = true;
        if (!finish_phrase && defer_analysis && value->samples_since_analysis < 32768)
            return strings(env, {});
        if (finish_phrase) {
            if (value->has_audio) {
                // Moonshine 0.1.5 skips final decoding when the last update consumed all
                // buffered audio. Its frontend also rounds down to 1,280-sample blocks.
                // A 512 ms silent tail forces a final pass over every real sample,
                // releases lookahead, and gives abrupt stops an acoustic ending. A
                // remainder-only drain can make fast counts continue beyond the audio.
                // This never enters our VAD or listening clock, so it cannot manufacture
                // a pause or an Auto-send, and it adds no wall-clock listening delay.
                const std::array<float, 8192> drain{};
                check(ts_push(value->stream, drain.data(), drain.size()));
            }
        }
        if (finish_phrase || value->samples_since_analysis >= value->analysis_interval_samples) {
            check(ts_update_ex(value->stream, finish_phrase,
                               value->previews && !value->analysis_was_deferred));
            value->samples_since_analysis = 0;
            value->analysis_was_deferred = false;
        }
        if (value->cancelled.load()) return strings(env, {""});
        std::vector<std::string> result;
        std::string preview;
        const size_t count_lines = ts_segment_count(value->stream);
        if (count_lines > 2000) throw std::runtime_error("Too many speech segments");
        for (size_t i = value->delivered; i < count_lines; ++i) {
            const char* text = ts_segment_text(value->stream, i);
            if (!text) throw std::runtime_error("Missing speech segment");
            if (ts_segment_complete(value->stream, i)) {
                result.emplace_back(text);
                value->delivered = i + 1;
            } else
                preview = text;
        }
        // The final array entry is always the revisable phrase; earlier entries are newly final.
        result.push_back(preview);
        if (finish_phrase) value->new_stream();
        return strings(env, result);
    } catch (const std::exception&) {
        fail(env, "On-device recognition failed");
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeCancel(JNIEnv*, jobject, jlong handle) {
    if (!handle) return;
    auto* value = reinterpret_cast<Engine*>(handle);
    {
        std::lock_guard<std::mutex> lock(value->pause_mutex);
        value->cancelled.store(true);
    }
    value->resumed.notify_all();
}
extern "C" JNIEXPORT void JNICALL Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativePause(
    JNIEnv*, jobject, jlong handle, jboolean paused) {
    if (!handle) return;
    auto* value = reinterpret_cast<Engine*>(handle);
    {
        std::lock_guard<std::mutex> lock(value->pause_mutex);
        value->paused = paused;
    }
    value->resumed.notify_all();
}
extern "C" JNIEXPORT void JNICALL
Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Engine*>(handle);
}
