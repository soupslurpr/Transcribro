package dev.soupslurpr.transcribro

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Parcel
import android.os.ParcelFileDescriptor
import android.os.ResultReceiver
import android.os.SystemClock
import android.provider.Settings
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.preferences.Preference
import dev.soupslurpr.transcribro.preferences.PreferencesRepository
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

internal data class EditorHostState(
    val text: String,
    val sent: List<String>,
    val trace: String = "",
)

/** Observes local test hosts without putting Chromium and the keyboard on the same UI thread. */
internal class EditorTestHost(kind: String) : AutoCloseable {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val browser = kind.startsWith("web-")
    private val local: ActivityScenario<EditorCompatibilityActivity>?

    init {
        if (browser) {
            local = null
            context.startActivity(
                Intent(context, RemoteWebEditorActivity::class.java)
                    .putExtra("host", kind)
                    .putExtra(
                        "editorTrace",
                        InstrumentationRegistry.getArguments().getString("editorTrace") == "true",
                    )
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
            val deadline = SystemClock.elapsedRealtime() + 10_000
            var ready = false
            while (!ready && SystemClock.elapsedRealtime() < deadline) {
                ready =
                    runCatching { query("snapshot", timeoutMillis = 250).getBoolean("ready") }
                        .getOrDefault(false)
                if (!ready) Thread.sleep(25)
            }
            check(ready) { "Browser fixture did not initialize" }
        } else {
            local =
                ActivityScenario.launch(
                    Intent(context, EditorCompatibilityActivity::class.java).putExtra("host", kind)
                )
        }
    }

    fun snapshot(): EditorHostState {
        if (browser) {
            val data = query("snapshot")
            return EditorHostState(
                data.getString("text").orEmpty(),
                data.getStringArrayList("sent").orEmpty(),
                data.getString("trace").orEmpty(),
            )
        }
        lateinit var state: EditorHostState
        local!!.onActivity { state = EditorHostState(it.field().text, it.sent.toList()) }
        return state
    }

    fun setSendDelay(delay: Long) {
        if (browser) query("delay", Bundle().apply { putLong("delay", delay) })
        else local!!.onActivity { it.sendDelay = delay }
    }

    fun setContent(text: String) {
        check(browser)
        query("content", Bundle().apply { putString("text", text) })
    }

    fun readParts(): List<String> = query("read").getStringArrayList("parts")!!

    fun append(text: String): Boolean =
        query("append", Bundle().apply { putString("text", text) }).getBoolean("accepted")

    private fun query(
        command: String,
        arguments: Bundle = Bundle(),
        timeoutMillis: Long = 5_000,
    ): Bundle {
        val done = CountDownLatch(1)
        var code = -1
        var result = Bundle()
        val callback =
            object : ResultReceiver(Handler(Looper.getMainLooper())) {
                override fun onReceiveResult(resultCode: Int, resultData: Bundle?) {
                    code = resultCode
                    result = resultData ?: Bundle()
                    done.countDown()
                }
            }
        // Send the framework parcelable, not the anonymous test-only subclass. The browser
        // process has the app's class loader and must not need the instrumentation APK.
        val parcel = Parcel.obtain()
        val remoteCallback =
            try {
                callback.writeToParcel(parcel, 0)
                parcel.setDataPosition(0)
                ResultReceiver.CREATOR.createFromParcel(parcel)
            } finally {
                parcel.recycle()
            }
        context.sendBroadcast(
            Intent(RemoteWebEditorActivity.ACTION)
                .setPackage(context.packageName)
                .putExtras(arguments)
                .putExtra("command", command)
                .putExtra("reply", remoteCallback)
        )
        check(done.await(timeoutMillis, TimeUnit.MILLISECONDS)) {
            "Browser command timed out: $command"
        }
        check(code == 0) { "Browser command failed: $command; ${result.getString("error")}" }
        return result
    }

    override fun close() {
        if (browser) query("close") else local!!.close()
    }
}

/** Prevents another keyboard from changing compositions inserted by editor component tests. */
internal fun withIdleVoiceKeyboard(test: () -> Unit) = runBlocking {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    val component = "${context.packageName}/dev.soupslurpr.transcribro.ui.voiceinput.VoiceInput"
    val resolver = context.contentResolver
    val preferences = PreferencesRepository(context)
    val before = preferences.values.first()
    fun shell(command: String) =
        ParcelFileDescriptor.AutoCloseInputStream(
                instrumentation.uiAutomation.executeShellCommand(command)
            )
            .bufferedReader()
            .use { it.readText() }
    val selected =
        shell("settings get secure default_input_method").trim().takeUnless { it == "null" }
    val enabled =
        shell("settings get secure enabled_input_methods").trim().takeUnless { it == "null" }
    try {
        preferences.set(Preference.AutoStart, false)
        shell("ime enable $component")
        shell("ime set $component")
        test()
    } finally {
        if (selected != null) shell("ime set $selected")
        instrumentation.uiAutomation.adoptShellPermissionIdentity(
            "android.permission.WRITE_SECURE_SETTINGS"
        )
        try {
            Settings.Secure.putString(resolver, Settings.Secure.ENABLED_INPUT_METHODS, enabled)
        } finally {
            instrumentation.uiAutomation.dropShellPermissionIdentity()
        }
        preferences.set(Preference.AutoStart, before.autoStart)
    }
}
