# Transcribro

Your voice. Your words.

Transcribro turns English speech into text in the Android apps you already
use. Dictate with the voice keyboard, use Android's recognition integrations,
or transcribe an audio file. No account, model download, or internet
permission is needed.

This branch contains **Transcribro Reimagined**, a local development candidate.
It has not replaced the published app.
The [product direction](docs/product-direction.md) describes its scope.
Bundled streaming recognition, live composing text, and background file
transcription are implemented.

## Speak, then make it yours

- **Ready to speak.** The launcher checks keyboard setup and microphone
  access, links to Android's settings, and offers a temporary field for trying
  the actual voice keyboard. Practice text clears when you leave the test.
- **Bring your audio.** Select an audio file with Android's file picker.
  Read its transcription, play and seek the source, then copy, share, or save
  a UTF-8 text file. Stopping keeps completed text. Editing belongs in
  BeauTyXT or another writing app. WAV and AAC/M4A are tested; other formats
  depend on Android's codecs.
- **Write in other apps.** Enable the voice keyboard for dictation, cursor
  movement, whole-character deletion, and the action requested by your editor.
  Separate Return and editor-action keys, accessible correction controls,
  and optional automatic sending at completion or after each speech phrase.
- **Android voice input.** Compatible apps can use Transcribro's recognition
  service or speech-recognition activity.
- **Material 3 Expressive.** Device colors, system light and dark appearance,
  and layouts for portrait and landscape.

Requires **Android 17 or newer** and a **64-bit ARM or x86 device**. Recognition
supports **English**. Live text appears while speaking and may change until
finalized. Turn Live text off for editors that prefer completed phrases.
Automatic sending is an explicit opt-in.

The voice keyboard starts listening when it opens by default. Turn off
**Start automatically** to begin each session with Start instead. Password
fields always require Start; opening Home or Settings does not start listening.

## What is kept

Microphone audio, practice text, and file results stay in memory. Transcribro
does not keep recordings, copy selected audio into a private library, or save
a transcript history. Practice text and the current file result survive
rotation, but not app-process death. Leaving the practice screen clears its
text and undo history. **Copy or save a transcription to keep your words.**
Only settings are stored automatically.

Recognition uses the original Moonshine English Medium Streaming weights
and Silero VAD through Transcribro's native runtime on ARM64 and x86_64.
The bundled archive is lossless; native libraries and DEX are uncompressed.
Every recognition session runs native inference in its own isolated process.
Selected files are decoded using Android's media services and streamed to the
worker without creating a recording file.

The receiving app gets the text when you dictate through the keyboard or use
its speech-recognition integration. Copy, Share, and Save hand text to Android
or the destination you choose. A cloud document provider may sync a saved file
using its own network access.

Live dictation sessions last up to five minutes. Selected audio can be up to
30 minutes long. File processing continues with a Stop notification when
you leave the app or lock the screen, and pauses while dictating. Listening
also stops after a minute without speech; a protective timeout keeps words
and does not automatically send them. Temporary text holds up to 64,000 UTF-16
code units. See [the architecture notes](docs/architecture.md) for resource bounds.

## Try the candidate

With JDK 21, Python 3.11, CMake, and the Android SDK installed
(see [build requirements](CONTRIBUTING.md#build)):

```sh
./gradlew :app:assembleStaging
adb -s DEVICE_SERIAL install -r app/build/outputs/apk/staging/app-staging.apk
```

The first build downloads checksum-pinned native sources, converter wheels,
and original model files, then prepares the exact bundled archive. Cached
inputs support `--offline` rebuilds after cleaning. Debug, staging, and release
use the same engine and both ABIs. The installed app works offline.

Debug and staging builds use `dev.soupslurpr.transcribro.reimagined`, a
development signing key, and the launcher name **Transcribro Reimagined**.
They install alongside the published app. Staging enables release optimization.

See [CONTRIBUTING.md](CONTRIBUTING.md) for build requirements and verification.
The [app icon artwork](artwork/launcher-icon/README.md) includes its editable
Blender scene, themed vector and export instructions.

## Published app

Published versions are available from
[Accrescent](https://accrescent.app/app/dev.soupslurpr.transcribro) and
[GitHub releases](https://github.com/soupslurpr/Transcribro/releases).
They differ from this development branch.

Production package: `dev.soupslurpr.transcribro`.

Production signing-certificate SHA-256:

```text
7D:BC:FB:FA:A1:35:B4:4E:6E:93:91:02:25:DC:B1:4E:05:82:91:DA:8C:2D:36:22:73:49:49:B7:1A:B3:BE:64
```

You can verify a published APK using `apksigner verify --print-certs FILE.apk`
or [AppVerifier](https://github.com/soupslurpr/AppVerifier). Local candidates
use a different certificate.

## Community and license

Report problems in the [issue tracker](https://github.com/soupslurpr/Transcribro/issues)
or join the [Transcribro Matrix space](https://matrix.to/#/#transcribro:matrix.org).

Transcribro is [MIT licensed](LICENSE.txt). Copied third-party material and its
provenance are recorded in [CREDITS](CREDITS). Required dependency and model
notices are also included in the app.
