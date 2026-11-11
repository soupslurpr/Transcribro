import java.util.Properties

plugins { id("com.android.library") }

val nativeSources = layout.buildDirectory.dir("generated/native")
val mediumAssets = layout.buildDirectory.dir("generated/medium-assets")
val nativeCache = rootProject.layout.projectDirectory.dir(".gradle/native-inputs")
val inputLock = layout.projectDirectory.file("native-inputs.lock.json")
val prepareToolProject = rootProject.layout.projectDirectory.dir("tools")
val prepareTool = prepareToolProject.file("target/release/transcribro-prepare")
val engineProject = rootProject.layout.projectDirectory.dir("engine")

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
        rootProject.layout.projectDirectory.file("tools/assets/silero-vad.ort"),
    )
    outputs.dir(mediumAssets)
    commandLine(
        prepareTool.asFile, "model", "--cache", nativeCache.asFile,
        "--sources", nativeSources.get().asFile, "--output", mediumAssets.get().asFile,
    )
    if (gradle.startParameter.isOffline) args("--offline")
}

// Gate 6 of the Kotlin+Rust rewrite: the product native library is the
// pure-Rust engine in /engine (JNI cdylib), built per ABI with cargo and
// the NDK clang linkers. The C++/CMake/whisper.cpp build is gone from
// the product; its sources remain in the tree as the reference oracle
// until Gate 7.
val cargoBinary = listOf(
    File(System.getProperty("user.home"), ".cargo/bin/cargo"),
    File("/usr/bin/cargo"),
).firstOrNull { it.canExecute() } ?: File("cargo")

data class RustAbi(val abi: String, val triple: String, val clangPrefix: String)

val rustAbis = listOf(
    RustAbi("arm64-v8a", "aarch64-linux-android", "aarch64"),
    RustAbi("x86_64", "x86_64-linux-android", "x86_64"),
)

val sdkDir = run {
    val props = Properties()
    val lp = rootProject.file("local.properties")
    if (lp.exists()) lp.inputStream().use { props.load(it) }
    File(
        props.getProperty("sdk.dir")
            ?: File(System.getProperty("user.home"), "android-sdk").absolutePath,
    )
}
val ndkDir = sdkDir.resolve("ndk/30.0.16248370")

val rustEngineTasks = rustAbis.map { target ->
    val suffix = target.abi.replace("-", "").replaceFirstChar { it.uppercase() }
    val buildTask = tasks.register<Exec>("buildRustEngine$suffix") {
        inputs.dir(engineProject.dir("src"))
        inputs.file(engineProject.file("Cargo.toml"))
        inputs.file(engineProject.file("Cargo.lock"))
        val lib = engineProject.file("target/${target.triple}/release/libtranscribro_standalone.so")
        outputs.file(lib)
        val linker = ndkDir.resolve(
            "toolchains/llvm/prebuilt/linux-x86_64/bin/${target.clangPrefix}-linux-android37-clang",
        )
        environment(
            "CARGO_TARGET_${target.triple.uppercase().replace('-', '_')}_LINKER",
            linker.absolutePath,
        )
        commandLine(
            cargoBinary, "build", "--release", "--locked",
            "--target", target.triple,
            "--manifest-path", engineProject.file("Cargo.toml").asFile,
        )
    }
    tasks.register<Copy>("stageRustEngine$suffix") {
        dependsOn(buildTask)
        from(engineProject.dir("target/${target.triple}/release")) {
            include("libtranscribro_standalone.so")
            rename { "libtranscribro.so" }
        }
        into(layout.projectDirectory.dir("src/main/jniLibs/${target.abi}"))
    }
}

val buildRustEngines = tasks.register("buildRustEngines") {
    dependsOn(rustEngineTasks)
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
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildTypes { create("staging") { initWith(getByName("release")) } }
}

tasks.named("preBuild") { dependsOn(buildRustEngines) }
