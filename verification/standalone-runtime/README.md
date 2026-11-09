# Native correctness tools

These tools build the production implementation from
`speech/src/main/cpp/medium/`. They do not select another app engine.
Keep detailed local evidence outside the repository.

Prepare the normal checksum-locked inputs, then build the native API tests:

```sh
./gradlew :speech:prepareNativeSources :speech:prepareMediumModel
transcribro_native_build="$(mktemp -d)"
cmake -S verification/standalone-runtime -B "$transcribro_native_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_SOURCE="$PWD/speech/build/generated/native/mlas" \
  -DEIGEN_SOURCE="$PWD/speech/build/generated/native/eigen" \
  -DMOONSHINE_SOURCE="$PWD/speech/build/generated/native/moonshine/core"
cmake --build "$transcribro_native_build" --target \
  test_runtime test_projection run_components run_stream
"$transcribro_native_build/test_runtime" /path/to/medium.trm /path/to/tokenizer.bin
"$transcribro_native_build/test_projection"
```

The uncompressed archive and tokenizer are retained under
`.gradle/native-inputs/models/`. `TRANSCRIBRO_SANITIZERS=ON` enables ASan/UBSan.
For Android, use the same source directory with the NDK toolchain, the target
ABI, API 37, and static libc++. No host-specific CPU flags are needed.

`run_components` consumes captured tensor fixtures. `verify_components.py`
and the optional `run_reference_components` target compare the exact original
ORT models against the selected native implementation. The latter requires
explicit `ORT_TEST_INCLUDE` and `ORT_TEST_LIBRARY` comparison dependencies.
`verify_audio` checks frontend and inner VAD bits; `--stream-blocks` also
compares paired native frontend work with the reference's 1,280-sample blocks.
Compare float bits against the reference on the same architecture.

`run_stream native` processes a list of supplied float PCM files and records
every scheduled preview and final update. For an ORT/Moonshine reference, use
`run_stream reference-independent`: its parent never loads Moonshine and
forks a fresh reference process for each recording. Recreating a transcriber
alone does not reset Moonshine's global Silero state. `compare_streams.py`
checks matching clip sets, schedules, every update, finals and word errors.
Reference libraries are test dependencies and never enter app artifacts.
The five `reference-patches/` patches reproduce the prior Medium policy from
Moonshine `234f60faa0eb388b01cdf7e60aca232af37aefda`, applied in this order:
VAD boundaries, full final encoding, token budgets, skipped previews, allocation.

`device_app.py` runs the existing JNI, worker, file and external supplied-PCM
tests. It backs up and restores the prior APK, permission, settings and
foreground, and removes its helper packages. It excludes microphone recording
and playback tests. Reserve an idle device or use an isolated emulator;
existing verification packages cause it to stop before changing the device.

`verification/corpus/` provides optional EdAcc, Earnings22, and quiet-recording
download helpers and PCM workload preparation. The harness reads prepared local
audio; it does not download every fixture automatically. LibriSpeech fixtures
currently need separate preparation. Keep downloaded inputs, license copies,
and their hashes in an external data directory.
