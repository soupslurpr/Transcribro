# Contributing

Discuss substantial changes in an issue before implementing them. Bug reports
should explain the steps, expected result, actual result, app version, and
Android version. Do not attach private recordings or transcripts.

Translations are not currently accepted.

## Build

Requirements:

- JDK 21 and the Gradle wrapper.
- Android SDK platform 37.1, NDK 30.0.16248370, and SDK CMake 3.22.1.
- Linux x86_64, a Rust toolchain (cargo), and a host C/C++ compiler and CMake
  for the pinned Zstandard compressor the model converter builds.
- No separate native source checkouts are needed; Gradle prepares the
  pinned Zstandard source automatically.
- Network access for the first build's declared Gradle dependencies and
  checksum-locked native/model/converter inputs.

```sh
./gradlew :app:assembleDebug :app:assembleStaging :app:assembleRelease
```

Set `ANDROID_HOME` or `sdk.dir` in local.properties to the installed SDK.
All variants use the original Medium weights and the same native engine for
`arm64-v8a` and `x86_64`. No runtime-selection or model-directory properties
are needed. `speech/native-inputs.lock.json` pins the one native download
(the pinned Zstandard source), the vendored VAD asset and its provenance,
the converter inputs, and the exact archive hashes. Conversion
runs in the Rust prepare tool in `tools/`; it verifies every pinned input by
size and SHA-256 and never resolves an unlocked package. Its current host
support is Linux x86_64.

Persistent inputs and reproducible intermediates live under
`.gradle/native-inputs/`, outside all `build/` outputs. Keep that directory
and the normal Gradle dependency cache for offline reconstruction:

```sh
./gradlew clean
./gradlew :app:assembleStaging --offline
```

`:speech:prepareNativeSources` extracts the pinned Zstandard source
into `speech/build/generated/native/` (which then holds exactly
`inputs.json` and `zstd/`).
`:speech:prepareMediumModel` verifies the
eight originals in `docs/moonshine-model.json`, imports their unchanged values
with the vendored inner Silero asset (`tools/assets/silero-vad.ort`), and
compresses the packed-head archive using
Zstandard 1.5.7, level 9, one worker. Both run automatically through declared
build dependencies; `--offline` also applies to nested acquisition. Deleting
only `build/` outputs needs no further network access. Deleting the persistent
input cache requires reacquisition. No model is published externally.

`speech/medium-model.json` pins the uncompressed archive to SHA-256
`11da1c8706a7b0b386d954ff9d77d805f2f4cd0cbbf09ca8292c0e68d492f896`
and its compressed representation to
`de2e9fd91654e889909c8420069ac24a60a5b088771e93e3204d8fb7b7e38184`.
The APK stores model assets, native libraries and DEX without ZIP compression.
At runtime the engine memory-maps the uncompressed model archive from the
APK and pages tensors in on demand; the app creates no extracted model files
and downloads nothing.

Use the Gradle wrapper. The Android interface is Kotlin and Jetpack Compose;
the isolated Kotlin worker owns session coordination, pause detection, and
its bounded audio queue. The small JNI bridge in the Rust engine (`engine/`) owns native model
contexts. Do not introduce
a second decoder or model runtime without reviewing its provenance, license,
maintenance, memory use, and on-device behavior.

Debug and staging share a development application ID and signing key. Installing
one replaces the other, but neither replaces production. Staging enables R8 and
resource shrinking. Do not use an intermediate APK from an earlier build:
install the artifact under `app/build/outputs/apk/`.

## Interface icons

Use Material Symbols Rounded Android vectors for standard interface icons.
Load them with `painterResource` and let the component supply its theme color.
Keep the upstream 24 dp geometry and automatic mirroring; record each pinned
source and its modifications in `CREDITS` and the bundled Material Symbols
notice. App identity artwork is maintained separately.

## Verify

```sh
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebugAndroidTest
adb -s DEVICE_SERIAL install -r app/build/outputs/apk/debug/app-debug.apk
adb -s DEVICE_SERIAL install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
```

File tests intentionally run without microphone access:

```sh
adb -s DEVICE_SERIAL shell pm revoke dev.soupslurpr.transcribro.reimagined android.permission.RECORD_AUDIO
adb -s DEVICE_SERIAL shell am instrument -w -r -e class dev.soupslurpr.transcribro.AudioFileIntegrationTest dev.soupslurpr.transcribro.reimagined.test/androidx.test.runner.AndroidJUnitRunner
```

Keyboard UI tests change and restore the test device's input-method settings.
The picker regressions start with Gboard or AOSP Keyboard and verify selection
from the launcher and from an unfocused practice field. Stop other UI automation
before running them:

```sh
adb -s DEVICE_SERIAL shell pm grant dev.soupslurpr.transcribro.reimagined android.permission.RECORD_AUDIO
adb -s DEVICE_SERIAL shell am instrument -w -r -e class dev.soupslurpr.transcribro.KeyboardDeviceTest dev.soupslurpr.transcribro.reimagined.test/androidx.test.runner.AndroidJUnitRunner
```

Launcher/file ownership checks use a provider packaged only in the test APK:

```sh
adb -s DEVICE_SERIAL shell am instrument -w -r -e class dev.soupslurpr.transcribro.LauncherFileTest dev.soupslurpr.transcribro.reimagined.test/androidx.test.runner.AndroidJUnitRunner
```

They cover practice/result separation, rotation, replacement and export
ownership, read-only presentation, cancellation, source playback setup, and
the playback slider's native accessibility label, time, and seek action.
The provider exposes only the synthetic fixtures and unlinks temporary copies
after opening their descriptors. Production APKs contain neither this provider
nor its audio fixtures.

Check the instrumentation's explicit test count and `OK` result, not just the
ADB process exit status. Test fixtures are synthetic and appear only in the test
APK. They verify integration, not recognition accuracy across real-world speech.

The `verification` module runs speech-service tests as an external
Android client against the optimized staging APK. It neither links app internals
nor weakens the app's shrinking rules to accommodate the test runner.

These tests temporarily select the candidate as Android's voice-input service
and restore the previous selection afterward. Android 17 only grants the needed
background microphone capability to selected or system recognition services.
They also temporarily revoke and restore the candidate's microphone permission
to distinguish live capture from supplied audio. Close other UI automation
before running the test client.

```sh
./gradlew :app:assembleStaging :verification:assembleStaging
adb -s DEVICE_SERIAL install -r app/build/outputs/apk/staging/app-staging.apk
adb -s DEVICE_SERIAL install -r verification/build/outputs/apk/staging/verification-staging.apk
adb -s DEVICE_SERIAL shell pm grant dev.soupslurpr.transcribro.reimagined android.permission.RECORD_AUDIO
adb -s DEVICE_SERIAL shell pm grant dev.soupslurpr.transcribro.verification android.permission.RECORD_AUDIO
adb -s DEVICE_SERIAL shell am instrument -w -r dev.soupslurpr.transcribro.verification/androidx.test.runner.AndroidJUnitRunner
```

Check portrait, landscape, enlarged text, light/dark appearance, microphone
denial, rotation during recognition, and returning from the file picker.
`EditorDictationTest` and `EditorHostTest` cover composing ownership, rollback,
completed-phrase fallback, and actual EditText/Compose hosts. The debug-only
editor activity records Send actions locally without contacting anyone.
`FileServiceTest` covers background work, dictation priority, screen-off
completion, and stale Stop actions.

Accessibility tests expect TalkBack and Switch Access installed on the test
device. They enable and restore those services to check native focus/actions
and target sizes; they do not replace testing with a physical switch or a
user who relies on spoken navigation.

Run `lintDebug` before a staging build, in a separate Gradle invocation.
Building another variant concurrently can replace the generated AIDL sources
while lint is inspecting them.

Keep local screenshots, build logs, detailed results, and candidate APKs out
of commits. Use temporary directories for test outputs and an external data
directory for downloaded benchmark inputs. The C++/Python component-verification harness that lived under
`verification/standalone-runtime/` was removed at Gate 7 of the Kotlin+Rust
rewrite, once the Rust engine had reached transcript parity with it; the C++
engine sources that were its reference oracle followed at the same gate,
once the speed program closed. The component tests must compare float bits
against a reference on the same architecture; stream checks must include
every scheduled preview and final update.
