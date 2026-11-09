plugins { id("com.android.library") }

val nativeSources = layout.buildDirectory.dir("generated/native")
val mediumAssets = layout.buildDirectory.dir("generated/medium-assets")
val nativeCache = rootProject.layout.projectDirectory.dir(".gradle/native-inputs")
val prepareScript = layout.projectDirectory.file("tools/prepare.py")
val inputLock = layout.projectDirectory.file("native-inputs.lock.json")

val prepareNativeSources = tasks.register<Exec>("prepareNativeSources") {
    inputs.files(prepareScript, inputLock)
    outputs.dir(nativeSources)
    commandLine(
        "python3", prepareScript.asFile, "sources", "--cache", nativeCache.asFile,
        "--output", nativeSources.get().asFile,
    )
    if (gradle.startParameter.isOffline) args("--offline")
}

val prepareMediumModel = tasks.register<Exec>("prepareMediumModel") {
    dependsOn(prepareNativeSources)
    inputs.files(
        prepareScript, inputLock, layout.projectDirectory.file("tools/import_model.py"),
        layout.projectDirectory.file("medium-model.json"),
        rootProject.layout.projectDirectory.file("docs/moonshine-model.json"),
        rootProject.layout.projectDirectory.file(
            "app/src/main/assets/models/silero_vad/ggml-silero-v6.2.0.bin",
        ),
    )
    outputs.dir(mediumAssets)
    commandLine(
        "python3", prepareScript.asFile, "model", "--cache", nativeCache.asFile,
        "--sources", nativeSources.get().asFile, "--output", mediumAssets.get().asFile,
    )
    if (gradle.startParameter.isOffline) args("--offline")
}

tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake")) {
        dependsOn(prepareNativeSources)
    }
}

android {
    namespace = "dev.soupslurpr.transcribro.speech"
    compileSdk { version = release(37) { minorApiLevel = 1 } }
    ndkVersion = "30.0.16248370"
    defaultConfig {
        minSdk = 37
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
            providers.gradleProperty("transcribroAbis").orNull?.let {
                abiFilters.clear()
                abiFilters += it.split(",")
            }
        }
        externalNativeBuild {
            cmake {
                targets += "transcribro"
                arguments += listOf(
                    "-DANDROID_STL=c++_static", "-DCMAKE_BUILD_TYPE=Release",
                    "-DORT_SOURCE=${nativeSources.get().asFile}/mlas",
                    "-DEIGEN_SOURCE=${nativeSources.get().asFile}/eigen",
                    "-DZSTD_SOURCE=${nativeSources.get().asFile}/zstd",
                    "-DWHISPER_ROOT=${nativeSources.get().asFile}/whisper",
                )
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildTypes { create("staging") { initWith(getByName("release")) } }
}
