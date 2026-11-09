#pragma once
#include <stddef.h>
#include <stdint.h>

// Fixed original Medium component boundary. No graph/runtime types cross it.
// One instance serializes its calls; independent instances run independently.
// Caller verifies the pinned archive digest and keeps its file immutable for
// the model's lifetime. Schema validation does not authenticate weight bytes.
// Buffers use original [batch=1, frames, channels] and cache
// [14, 1, 10, sequence, 64] layouts. Caller owns and sizes all output buffers.
#ifdef __cplusplus
extern "C" {
#endif
void *tm_create(const char *archive, char *error, size_t error_bytes);
// Borrow a verified 64-byte-aligned archive. It remains immutable and alive
// until tm_destroy, avoiding a second copy of decompressed Android assets.
void *tm_create_memory(const void *archive, size_t size, char *error,
                       size_t error_bytes);
void tm_destroy(void *model);
const char *tm_error(void);
// A cache belongs to one stream. Do not share it concurrently or between
// models. Reset invalidates all previous entries without reallocating the
// arena.
void *tm_cache_create(void);
void tm_cache_destroy(void *cache);
void tm_cache_reset(void *cache);
int tm_decoder_cached(void *model, void *cache, const int64_t *tokens,
                      size_t count, const float *cross_keys,
                      const float *cross_values, size_t frames,
                      int64_t *predictions);
typedef struct tm_frontend_state {
  float sample_buffer[79];
  int64_t sample_len;
  float conv1[768 * 4];
  float conv2[1536 * 4];
  int64_t frame_count;
} tm_frontend_state;
// Zero-initialize state at the start of an utterance. Capacity counts frames.
int tm_frontend(void *model, const float *audio, size_t samples,
                tm_frontend_state *state, float *features, size_t capacity,
                size_t *frames);
typedef struct tm_vad_state {
  float context[64];
  float hidden[128];
  float cell[128];
} tm_vad_state;
// Exactly 512 new samples at 16 kHz, with per-stream recurrent state.
int tm_vad(void *model, const float *audio, tm_vad_state *state,
           float *probability);
#ifdef TRANSCRIBRO_TRACE
// Diagnostic builds only. The callback is synchronous and local to the calling
// thread. It must not retain borrowed pointers or re-enter the model.
typedef void (*tm_trace_callback)(void *, const char *, const float *, size_t);
void tm_set_trace(tm_trace_callback callback, void *context);
#endif
int tm_encoder(void *model, const float *features, size_t frames,
               float *encoded);
int tm_adapter(void *model, const float *encoded, size_t frames, size_t offset,
               float *memory);
int tm_cross(void *model, const float *memory, size_t frames, float *keys,
             float *values);
int tm_decoder(void *model, const int64_t *tokens, size_t count,
               const float *keys, const float *values, size_t past,
               const float *cross_keys, const float *cross_values,
               size_t frames, float *logits, float *out_keys,
               float *out_values);
#ifdef __cplusplus
}
#endif
