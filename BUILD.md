# Building and signing a Transcribro APK

These steps assume the Android SDK is already installed. Transcribro has a native
component (whisper.cpp), so the NDK and CMake are required in addition to the usual
SDK pieces.

## 1. Prerequisites

Install / make sure you have:

- **JDK 17** (the project targets Java 17).
- **Android SDK Platform 36** (`compileSdk = 36`, `targetSdk = 36`).
- **Android SDK Build-Tools 36.0.0** (matches `buildToolsVersion`; provides `zipalign`/`apksigner`).
- **Android NDK 27.2.12479018** (matches `ndkVersion`).
- **CMake** (installed via the SDK manager; used to build whisper.cpp).

Using the SDK command-line tools you can install everything with:

```bash
sdkmanager "platforms;android-36" \
           "build-tools;36.0.0" \
           "ndk;27.2.12479018" \
           "cmake;3.22.1"
```

Point the build at your SDK by creating `local.properties` in the project root:

```properties
sdk.dir=/absolute/path/to/Android/sdk
```

(or export `ANDROID_HOME` / `ANDROID_SDK_ROOT`).

## 2. Get the sources, including whisper.cpp

whisper.cpp is a git submodule (`whisper.cpp/`, from
`https://github.com/ggerganov/whisper.cpp`). The native build will fail if it is empty.
If you cloned the repo, populate it with:

```bash
git submodule update --init --recursive
```

If you are working from an archive instead of a git checkout, clone whisper.cpp into the
`whisper.cpp/` directory manually (use the commit your version expects).

## 3. Provide the multilingual model

The app bundles a single model asset (the "Base" model) and can also use a model imported
from the device's file system.

Bundled model path:

```
app/src/main/assets/models/whisper/ggml-base-q8_0.bin
```

It ships as an empty placeholder — replace it with a real **multilingual** GGML base model
for the bundled option to work. (`noCompress += "bin"` keeps it stored uncompressed so the
build succeeds even with a placeholder and the model can be read directly.)

Alternatively, users can supply a model at runtime without rebuilding: in
*Settings → Model → Model download links* the app generates a Hugging Face URL for any
ggml model, which can be downloaded in a browser and then imported via
*Settings → Model → Choose model file…*. Importing uses the Storage Access Framework, so it
needs no storage permission and the app requires no network permission (it never downloads
models itself). An English-only (`.en`) model will ignore any non-English language request.

## 4. Quick path — debug APK (auto-signed with the debug key)

Good for local testing; signed automatically with the Android debug keystore.

```bash
./gradlew :app:assembleDebug
```

Output: `app/build/outputs/apk/debug/app-debug.apk`
(the debug build uses application id `dev.soupslurpr.transcribro.debug`).

## 5. Release APK signed with your own key

### 5a. Create a release keystore (once)

```bash
keytool -genkeypair -v \
  -keystore transcribro-release.jks \
  -keyalg RSA -keysize 4096 -validity 10000 \
  -alias transcribro
```

Keep this file and its passwords safe — every future update must be signed with the same key.

### 5b. Option A — let Gradle sign the release build

Add a signing config to `app/build.gradle.kts` that reads credentials from environment
variables (so secrets stay out of version control), and attach it to the `release` build type:

```kotlin
android {
    signingConfigs {
        create("release") {
            storeFile = file(System.getenv("TRANSCRIBRO_KEYSTORE") ?: "transcribro-release.jks")
            storePassword = System.getenv("TRANSCRIBRO_KEYSTORE_PASSWORD")
            keyAlias = System.getenv("TRANSCRIBRO_KEY_ALIAS") ?: "transcribro"
            keyPassword = System.getenv("TRANSCRIBRO_KEY_PASSWORD")
        }
    }
    buildTypes {
        release {
            // ...existing release config...
            signingConfig = signingConfigs.getByName("release")
        }
    }
}
```

Then build:

```bash
export TRANSCRIBRO_KEYSTORE=/absolute/path/to/transcribro-release.jks
export TRANSCRIBRO_KEYSTORE_PASSWORD=...   # your store password
export TRANSCRIBRO_KEY_ALIAS=transcribro
export TRANSCRIBRO_KEY_PASSWORD=...        # your key password

./gradlew :app:assembleRelease
```

Output (already signed and zipaligned):
`app/build/outputs/apk/release/app-release.apk`

### 5c. Option B — build unsigned, then sign manually

Without changing the Gradle config, the release build is unsigned. Build it, then align and
sign with the Build-Tools utilities:

```bash
# 1. Build the (unsigned) release APK
./gradlew :app:assembleRelease
# -> app/build/outputs/apk/release/app-release-unsigned.apk

# 2. Align it (use your Build-Tools path)
BT="$ANDROID_HOME/build-tools/36.0.0"
"$BT/zipalign" -v -p 4 \
  app/build/outputs/apk/release/app-release-unsigned.apk \
  app-release-aligned.apk

# 3. Sign it
"$BT/apksigner" sign \
  --ks transcribro-release.jks \
  --ks-key-alias transcribro \
  --out Transcribro-release.apk \
  app-release-aligned.apk
```

## 6. Verify the signature

```bash
"$ANDROID_HOME/build-tools/36.0.0/apksigner" verify --print-certs Transcribro-release.apk
```

This prints the signing certificate's SHA-256, which you can compare against the value
published in the project README to confirm the build is signed with the expected key.

## 7. Install on a device

```bash
adb install -r Transcribro-release.apk
```

After installing, enable the **Transcribro** keyboard in *Settings → System → Languages &
input → On-screen keyboards*, and pick its language with the on-screen language-switch
("globe") key. The transcription language follows that selection unless you override it in
the app under *Settings → Language → Transcription language* (Follow keyboard language /
Auto-detect / a specific language).

> Note: building an Android App Bundle (`./gradlew :app:bundleRelease`) and turning it into
> APKs with `bundletool` is also possible; see `app/release/signing-command.txt` for the
> author's bundletool invocation. The steps above produce a directly installable APK.
