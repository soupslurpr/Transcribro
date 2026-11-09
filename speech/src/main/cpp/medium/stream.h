#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Research streaming API. PCM is mono float32 at 16 kHz. Calls for a stream
// must be serialized; separate streams have independent model/state ownership.
void *ts_create(const char *archive, const char *tokenizer, char *error,
                size_t size);
// Archive is borrowed, 64-byte aligned and immutable until ts_destroy.
// Tokenizer bytes are copied while constructing the stream.
void *ts_create_memory(const void *archive, size_t archive_size,
                       const void *tokenizer, size_t tokenizer_size,
                       char *error, size_t error_size);
void ts_destroy(void *stream);
int ts_push(void *stream, const float *audio, size_t samples);
int ts_update(void *stream, int final);
int ts_update_ex(void *stream, int final, int decode_previews);
size_t ts_segment_count(void *stream);
const char *ts_segment_text(void *stream, size_t index);
int ts_segment_complete(void *stream, size_t index);
// Returned text is borrowed until the next call on this stream.
const char *ts_text(void *stream);
// Pass zero to start an independent session. Keep the recurrent detector only
// between phrases in one session, or to reproduce the reference benchmark's
// global detector across clips. A newly created stream starts with zero state.
int ts_reset(void *stream, int keep_vad);
const char *ts_error(void);
#ifdef __cplusplus
}
#endif
