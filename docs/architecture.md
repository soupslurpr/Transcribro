# Architecture

The [product direction](product-direction.md) defines the user-facing scope.

## Entry points and process ownership

`MainActivity` hosts setup, temporary keyboard practice, file transcription,
and settings. Practice text stays in memory. File results have a separate,
read-only owner. Neither transcript survives process recreation.
The keyboard is an `InputMethodService`; external speech clients use
`MainRecognitionService` or the exported recognition activity.

Each dictation owner has its own `DictationController` and generation.
Cancellation invalidates late callbacks before they can commit into another
editor. The keyboard and recognition activity do not share composing state.

`MainRecognitionService` checks the caller's microphone permission and
attribution. Live capture uses a caller-attributed `AudioRecord`. Supplied
mono 16 kHz signed PCM16 uses the provided descriptor without opening the
microphone. Invalid supplied-audio requests fail without falling back to live
capture. The older `EXTRA_AUTO_STOP` contract remains supported; the newer
continuous option takes precedence when supplied.

Each session binds a uniquely named `TranscriptionWorker` with
`bindIsolatedService`. Its isolated UID has no microphone, network, or
app-private storage access. It receives packaged assets and a PCM pipe and
returns bounded text, progress and terminal status. Kotlin owns the session
and queue; JNI owns the native contexts. Interactive recognition uses
`BIND_IMPORTANT`; file work uses an ordinary binding.

## Native recognition

All app variants use `speech/src/main/cpp/engine.cpp` and the original-Medium
runtime in `speech/src/main/cpp/medium/`, on ARM64 and x86_64. It implements
the frontend, encoder, adapter, cross-attention, decoder, inner Silero VAD,
segmentation and tokenizer. There is one app engine and one model precision.

The archive reversibly packs the exact original integer weights and retains
all FP32 values, scales, zero points and biases. The tokenizer is unchanged.
JNI decompresses its bundled Zstandard frame into aligned RAM and lends the
allocation to the model until destruction. It creates no extracted model
files. ARM64 reads the packed tied embedding/head directly; x86_64 reconstructs
the same original FP32 head. Baseline x86 projections reconstruct original
unsigned codes for MLAS integer GEMM; CPUs supporting the selected VNNI path
keep its direct packed projection kernel.

The runtime links Microsoft ONNX Runtime's standalone MLAS numerical kernels
and Eigen MPL 2.0 headers. It contains no ONNX graph/session runtime or
Moonshine runtime library. Arithmetic follows the reference on each
architecture, including explicit normalization contraction rules. CPU kernels
use runtime dispatch with a portable baseline, without host-specific tuning.
The outer pause detector remains whisper.cpp/Silero 6.2 with one CPU thread.

Each model serializes component calls and owns its scratch storage. Each
stream owns its detector, segments and persistent decoder cache. Separate
models have independent inference state. The worker serializes stream calls;
pause and cancellation use the existing synchronized JNI boundary.

Streaming policy adapted from Moonshine retains bounded initial look-behind
and prevents a completed segment's audio from entering its successor.
Previews use incremental encoding. Finalization rebuilds the complete retained
feature span before final decoding. Speculative decoding respects the same
budget as greedy decoding: 20 final tokens per second, bounded at 256 per
segment and by model context, with 6.5 tokens per second for previews.
No transcript-text deduplication or repair is applied.

The outer detector consumes 512-sample frames with persistent state,
hysteresis, minimum voiced duration and a configurable pause threshold.
Its pause markers are distinct from internal processing splits. A final or
manual boundary drains every preceding real sample; JNI appends 8,192 silent
samples for frontend lookahead and an abrupt acoustic ending. This tail never
enters the pause detector or listening clock and adds no capture delay.
A fresh phrase resets the stream while retaining the intended detector state.

The stream emits stable segments and one revisable unfinished segment.
Android partial results are cumulative; a confirmed-length extra distinguishes
stable text from previews. Final results contain completed text. File work
and dictation with Live text disabled skip tentative decoding.

Gradle automatically acquires and verifies pinned sources, converter wheels
and original model files. `speech/native-inputs.lock.json` and
`speech/medium-model.json` identify every conversion input and exact artifact.
See [CONTRIBUTING](../CONTRIBUTING.md#build) for requirements and offline caches.
Native libraries and DEX are uncompressed; ELF segments align to 16 KiB.

## Capture, queue and boundaries

Live capture starts before model loading and buffers into a bounded startup
queue. Readiness follows successful capture startup. Capture and inference
have separate threads. At most eight 512 ms chunks wait behind the active
recognition call. With at least two queued chunks, preview analysis can be
coalesced while all PCM remains in FIFO order. Native analyses are separated
by at most 32,768 retained samples, and final/manual boundaries always decode.
Deferred analysis retains the previous preview until a normal update revises it.

If inference cannot keep up, the worker requests capture stop, then drains
captured pipe audio through EOF using bounded backpressure. Completed words
remain available; protective stopping does not trigger Send or switch the
keyboard. Supplied audio and files use backpressure directly.

Finish drains the recorder's captured samples, releases capture, and closes
the pipe. Microphone-end state is distinct from a request to finish. A manual
phrase boundary has a sample watermark and acknowledgement so later audio
cannot be committed into the preceding phrase. Cancellation can discard queued
audio and interrupt blocked I/O; an active native inference call completes
before its context is released.

## Text and file ownership

The dictation controller separates stable text, revisable text and a pending
Send boundary. Editing, cursor movement, focus changes and cancellation can
invalidate composing ownership. Late results cannot overwrite intervening
edits. Sending follows a completed phrase and the configured policy; internal
segment splits and overloaded capture do not manufacture Send actions.

`FileTranscriptionService` owns selected-file work independently of launcher
composition and practice text. Files are decoded to bounded PCM chunks and
passed through the same isolated worker. Dictation has priority over active
file inference. File progress, cancellation and terminal status have one owner;
rotation or stale Stop actions cannot attach to a newer job.

Audio and transcripts are not retained in history or sent to a server.
Export and copy are explicit user actions. Recognition has a five-minute
live limit, file input a thirty-minute limit, and native text results are
bounded to 16,000 bytes per entry. These resource limits do not promise
recognition accuracy for every voice or recording.
