plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

abstract class PrepareMediumAssets : DefaultTask() {
    @get:InputDirectory @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val modelDirectory: DirectoryProperty
    @get:OutputDirectory abstract val outputDirectory: DirectoryProperty
    @get:javax.inject.Inject abstract val files: FileSystemOperations

    @TaskAction fun prepare() {
        files.sync {
            from(modelDirectory)
            into(outputDirectory)
        }
    }
}

androidComponents {
    onVariants { variant ->
        val assets = tasks.register<PrepareMediumAssets>(
            "prepare${variant.name.replaceFirstChar { it.uppercase() }}MediumAssets"
        ) {
            dependsOn(":speech:prepareMediumModel")
            modelDirectory.set(project(":speech").layout.buildDirectory.dir("generated/medium-assets"))
            outputDirectory.set(layout.buildDirectory.dir("generated/mediumAssets/${variant.name}"))
        }
        variant.sources.assets?.addGeneratedSourceDirectory(assets, PrepareMediumAssets::outputDirectory)
    }
}

android {
    namespace = "dev.soupslurpr.transcribro"
    compileSdk { version = release(37) { minorApiLevel = 1 } }
    defaultConfig {
        applicationId = "dev.soupslurpr.transcribro"
        minSdk = 37
        targetSdk = 37
        versionCode = 8
        versionName = "8"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
    buildFeatures {
        compose = true
        buildConfig = true
        aidl = true
        resValues = true
    }
    androidResources {
        noCompress += listOf("bin", "zst")
        localeFilters += "en"
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        debug {
            applicationIdSuffix = ".reimagined"
            versionNameSuffix = "-dev"
            resValue("string", "app_name", "Transcribro Reimagined")
        }
        create("staging") {
            initWith(getByName("release"))
            applicationIdSuffix = ".reimagined"
            versionNameSuffix = "-candidate"
            resValue("string", "app_name", "Transcribro Reimagined")
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }
    packaging.jniLibs.useLegacyPackaging = false
    packaging.dex.useLegacyPackaging = false
}

dependencies {
    implementation(project(":speech"))
    implementation("androidx.datastore:datastore-preferences:1.2.1")
    implementation("androidx.core:core-ktx:1.19.0")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.navigation3:navigation3-ui:1.3.0-alpha01")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    implementation(platform("androidx.compose:compose-bom-alpha:2026.09.01"))
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3:1.5.0-alpha29")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:rules:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}
