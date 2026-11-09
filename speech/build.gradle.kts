plugins { id("com.android.library") }

val nativeSources = layout.buildDirectory.dir("generated/native")
val mediumAssets = layout.buildDirectory.dir("generated/medium-assets")
val nativeCache = rootProject.layout.projectDirectory.dir(".gradle/native-inputs")
val inputLock = layout.projectDirectory.file("native-inputs.lock.json")
val prepareToolProject = rootProject.layout.projectDirectory.dir("tools")
val prepareTool = prepareToolProject.file("target/release/transcribro-prepare")

// Gate 1 of the Kotlin+Rust rewrite: prepare.py + import_model.py are
// replaced by the Rust tool in /tools (byte-identical outputs; the Python
// tools remain in the tree as the reference oracle until Gate 7).
val buildPrepareTool = tasks.register<Exec>("buildPrepareTool") {
    inputs.dir(prepareToolProject.dir("src"))
    inputs.file(prepareToolProject.file("Cargo.toml"))
    inputs.file(prepareToolProject.file("Cargo.lock"))
    outputs.file(prepareTool)
    val cargo = listOf(
        File(System.getProperty("user.home"), ".cargo/bin/cargo"),
        File("/usr/bin/cargo"),
    ).firstOrNull { it.canExecute() } ?: File("cargo")
    commandLine(
        cargo, "build", "--release", "--locked",
        "--manifest-path", prepareToolProject.file("Cargo.toml").asFile,
    )
}

val prepareNativeSources = tasks.register<Exec>("prepareNativeSources") {
    dependsOn(buildPrepareTool)
    inputs.files(prepareTool, inputLock)
    outputs.dir(nativeSources)
    commandLine(
        prepareTool.asFile, "sources", "--cache", nativeCache.asFile,
        "--output", nativeSources.get().asFile,
    )
    if (gradle.startParameter.isOffline) args("--offline")
}

val prepareMediumModel = tasks.register<Exec>("prepareMediumModel") {
    dependsOn(prepareNativeSources)
    inputs.files(
        prepareTool, inputLock,
        layout.projectDirectory.file("medium-model.json"),
        rootProject.layout.projectDirectory.file("docs/moonshine-model.json"),
        rootProject.layout.projectDirectory.file(
            "app/src/main/assets/models/silero_vad/ggml-silero-v6.2.0.bin",
        ),
    )
    outputs.dir(mediumAssets)
    commandLine(
        prepareTool.asFile, "model", "--cache", nativeCache.asFile,
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
