plugins { id("com.android.test") }

android {
    namespace = "dev.soupslurpr.transcribro.verification"
    compileSdk { version = release(37) { minorApiLevel = 1 } }
    defaultConfig {
        minSdk = 37
        targetSdk = 37
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    targetProjectPath = ":app"
    // Run as an external Android speech client, without linking app internals.
    experimentalProperties["android.experimental.self-instrumenting"] = true
    buildTypes {
        create("staging") {
            isDebuggable = true
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    sourceSets.getByName("main").assets.srcDir("../app/src/androidTest/assets")
}

androidComponents { beforeVariants(selector().all()) { it.enable = it.buildType == "staging" } }

dependencies {
    implementation("androidx.test:runner:1.7.0")
    implementation("androidx.test.ext:junit:1.3.0")
}
