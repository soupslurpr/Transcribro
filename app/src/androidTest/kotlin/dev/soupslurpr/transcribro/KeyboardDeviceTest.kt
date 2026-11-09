package dev.soupslurpr.transcribro

import android.Manifest
import android.accessibilityservice.AccessibilityServiceInfo
import android.app.UiAutomation
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Rect
import android.media.AudioManager
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.provider.Settings
import android.view.InputDevice
import android.view.MotionEvent
import android.view.ViewConfiguration
import android.view.WindowInsets
import android.view.accessibility.AccessibilityNodeInfo
import android.view.inputmethod.EditorInfo
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.preferences.Preference
import dev.soupslurpr.transcribro.preferences.PreferencesRepository
import dev.soupslurpr.transcribro.preferences.SendTiming
import dev.soupslurpr.transcribro.ui.voiceinput.VoiceInput
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the actual voice keyboard against the launcher's temporary practice field. */
@RunWith(AndroidJUnit4::class)
class KeyboardDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val automation =
        instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)

    @Test
    fun microphoneReplayReachesTenInTheActualPracticeField() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("microphoneReplay") == "true")
        withReplayPreferences {
            withKeyboard {
                val started = SystemClock.elapsedRealtime()
                click("Start dictating")
                awaitNode { it.text?.toString() == "Listening" }
                instrumentation.sendStatus(
                    2,
                    Bundle().apply {
                        putLong(
                            "transcribro_ready_observed_ms",
                            SystemClock.elapsedRealtime() - started,
                        )
                    },
                )
                replayMicrophone()
                click("Finish dictation")
                awaitNode { it.text?.toString() == "Ready" }
                val field = awaitNode { it.className == "android.widget.EditText" }
                assertCounting(field.text.toString())
            }
        }
    }

    @Test
    fun microphoneReplaySendsSuccessivePhrasesThroughTheActualKeyboard() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("microphoneReplay") == "true")
        val automatic = InstrumentationRegistry.getArguments().getString("automatic") == "true"
        val hostKind = InstrumentationRegistry.getArguments().getString("host") ?: "compose"
        val betweenPhrases =
            InstrumentationRegistry.getArguments()
                .getString("betweenPhrasesMs")
                ?.toLongOrNull()
                ?.coerceIn(0, 15_000) ?: 0
        withReplayPreferences(automatic) {
            withKeyboard {
                EditorTestHost(hostKind).use { host ->
                    host.setSendDelay(150)
                    if (hostKind.startsWith("web-")) focusBrowserField()
                    awaitNode { it.contentDescription?.toString() == "Start dictating" }
                    click("Start dictating")
                    awaitNode { it.text?.toString() == "Listening" }
                    repeat(2) { index ->
                        if (index == 1 && betweenPhrases > 0) Thread.sleep(betweenPhrases)
                        replayMicrophone()
                        instrumentation.sendStatus(
                            2,
                            Bundle().apply {
                                putString("transcribro_before_send", host.snapshot().text)
                            },
                        )
                        if (!automatic) click("Send")
                        val deadline = android.os.SystemClock.elapsedRealtime() + 15_000
                        var sent = emptyList<String>()
                        var remaining = "<not cleared>"
                        do {
                            val snapshot = host.snapshot()
                            sent = snapshot.sent
                            remaining = snapshot.text
                            if (sent.size == index + 1 && remaining.isEmpty()) break
                            Thread.sleep(25)
                        } while (android.os.SystemClock.elapsedRealtime() < deadline)
                        assertEquals(
                            "Message action was not completed exactly once; field=$remaining; " +
                                "controls=${visibleLabels(context.packageName)}",
                            index + 1,
                            sent.size,
                        )
                        try {
                            assertCounting(sent.last())
                        } catch (error: AssertionError) {
                            instrumentation.sendStatus(
                                2,
                                Bundle().apply {
                                    putString("transcribro_editor_trace", host.snapshot().trace)
                                },
                            )
                            throw error
                        }
                        assertEquals("The local message host did not clear", "", remaining)
                        awaitNode { it.text?.toString() == "Listening" }
                        if (index == 0 && !automatic && hostKind != "web-input") {
                            // A same-field correction also keeps capture running.
                            click("Insert newline")
                            if (hostKind.startsWith("web-"))
                                awaitHostText(
                                    host,
                                    "\n",
                                    trailingPlaceholder = hostKind == "web-rich",
                                )
                            else awaitText("\n")
                            awaitNode { it.text?.toString() == "Listening" }
                        }
                    }
                    click("Finish dictation")
                    awaitNode { it.text?.toString() == "Ready" }
                    val final = host.snapshot()
                    assertEquals(2, final.sent.size)
                    assertEquals("", final.text)
                }
            }
        }
    }

    @Test
    fun newlineInBrowserMessageFieldDoesNotSend() = withKeyboard {
        for (kind in listOf("web-textarea", "web-rich")) {
            EditorTestHost(kind).use { host ->
                focusBrowserField()
                awaitNode { it.contentDescription?.toString() == "Start dictating" }
                host.setContent("First line")
                awaitHostText(host, "First line")
                click("Insert newline")
                awaitHostText(host, "First line\n", trailingPlaceholder = kind == "web-rich")
                val snapshot = host.snapshot()
                assertTrue("Return must not send: ${snapshot.sent}", snapshot.sent.isEmpty())
                assertTrue(host.append("Second line"))
                awaitHostText(host, "First line\nSecond line")
            }
        }
    }

    @Test
    fun newlineStillWorksWhenAnEditorRejectsComposition() = withKeyboard {
        ActivityScenario.launch<EditorCompatibilityActivity>(
                Intent(context, EditorCompatibilityActivity::class.java)
                    .putExtra("rejectComposing", true)
            )
            .use { host ->
                awaitNode { it.contentDescription?.toString() == "Start dictating" }
                host.onActivity { it.input!!.commitText("First line", 1) }
                awaitText("First line")
                click("Insert newline")
                awaitText("First line\n")
                host.onActivity { assertTrue(it.sent.isEmpty()) }
            }
    }

    @Test
    fun correctionsKeepTheActualMicrophoneListening() = withReplayPreferences {
        withKeyboard {
            setText("Words")
            select(5, 5)
            click("Start dictating")
            awaitNode { it.text?.toString() == "Listening" }
            for ((control, expected) in
                listOf(
                    "Insert newline" to "Words\n",
                    "Space" to "Words\n ",
                    "Backspace" to "Words\n",
                    "Move cursor left" to "Words\n",
                    "Move cursor right" to "Words\n",
                    "Clear field" to "",
                    "Undo" to "Words\n",
                    "Redo" to "",
                )) {
                instrumentation.sendStatus(
                    2,
                    Bundle().apply { putString("transcribro_edit", control) },
                )
                click(control)
                awaitText(expected)
                awaitNode { it.text?.toString() == "Listening" }
            }
            click("Finish dictation")
            awaitNode { it.text?.toString() == "Ready" }
        }
    }

    @Test
    fun automaticStartWaitsForPreferencesAndStaysStoppedUntilReopened() = withReplayPreferences {
        withKeyboard(autoStart = true) {
            awaitNode { it.text?.toString() == "Listening" }
            awaitMicrophone(active = true)
            click("Finish dictation")
            awaitNode { it.text?.toString() == "Ready" }
            awaitMicrophone(active = false)
            Thread.sleep(300)
            awaitNode { it.contentDescription?.toString() == "Start dictating" }
            click("Start dictating")
            awaitNode { it.text?.toString() == "Listening" }
            awaitMicrophone(active = true)
            click("Cancel")
            awaitNode { it.text?.toString() == "Ready" }
            awaitMicrophone(active = false)
            Thread.sleep(300)
            awaitNode { it.contentDescription?.toString() == "Start dictating" }
            shell("input keyevent KEYCODE_BACK")
            awaitNode { it.text?.toString()?.startsWith("Tap the field") == true }
            assertTrue(
                awaitNode { it.className == "android.widget.EditText" }
                    .performAction(AccessibilityNodeInfo.ACTION_CLICK)
            )
            awaitNode { it.text?.toString() == "Listening" }
            awaitMicrophone(active = true)
            click("Cancel")
            awaitMicrophone(active = false)
        }
    }

    @Test
    fun automaticStartKeepsHomeAndSettingsSilent() = withReplayPreferences {
        withKeyboard(autoStart = true, openPractice = false) {
            awaitNode { it.text?.toString() == "Try dictation" }
            awaitMicrophone(active = false)
            click("Settings")
            awaitNode { it.text?.toString() == "Start automatically" }
            awaitMicrophone(active = false)
            captureKeyboard("automatic-start-settings")
            shell("input keyevent KEYCODE_BACK")
            click("Try dictation")
            awaitNode { it.text?.toString()?.contains("Speak when it says Listening") == true }
            awaitMicrophone(active = false)
        }
    }

    @Test
    fun storedAutomaticStartOffWaitsForAnExplicitStart() = withReplayPreferences {
        withKeyboard(autoStart = false) {
            awaitNode { it.text?.toString() == "Ready" }
            awaitMicrophone(active = false)
            click("Start dictating")
            awaitNode { it.text?.toString() == "Listening" }
            awaitMicrophone(active = true)
            click("Cancel")
            awaitMicrophone(active = false)
        }
    }

    private fun awaitMicrophone(active: Boolean) {
        val audio = context.getSystemService(AudioManager::class.java)
        val deadline = SystemClock.elapsedRealtime() + 5_000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (audio.activeRecordingConfigurations.isNotEmpty() == active) return
            Thread.sleep(50)
        }
        assertEquals("Unexpected microphone capture state", active, audio.activeRecordingConfigurations.isNotEmpty())
    }

    @Test
    fun passwordFieldAllowsExplicitDictationAndDoesNotStartAutomatically() = withReplayPreferences {
        withKeyboard(autoStart = true, openPractice = false) {
            ActivityScenario.launch<EditorCompatibilityActivity>(
                    Intent(context, EditorCompatibilityActivity::class.java)
                        .putExtra("password", true)
                )
                .use {
                    awaitNode { it.contentDescription?.toString() == "Start dictating" }
                    awaitNode { it.text?.toString() == "Ready" }
                    awaitNode {
                        it.text?.toString() ==
                            "Tap to dictate in this field. Automatic start is off."
                    }
                    click("Start dictating")
                    awaitNode { it.text?.toString() == "Listening" }
                    click("Finish dictation")
                    awaitNode { it.text?.toString() == "Ready" }
                    val explanation = awaitNode {
                        it.text?.toString() == "No words recognized. Tap Start to try again."
                    }
                    assertEquals(
                        android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE,
                        explanation.liveRegion,
                    )
                    if (
                        InstrumentationRegistry.getArguments().getString("captureScreenshots") ==
                            "true"
                    ) {
                        instrumentation.waitForIdleSync()
                        Thread.sleep(500)
                        val bitmap = checkNotNull(automation.takeScreenshot())
                        try {
                            File(context.cacheDir, "password-no-speech.png").outputStream().use {
                                output ->
                                assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
                            }
                        } finally {
                            bitmap.recycle()
                        }
                    }
                }
        }
    }

    @Test
    fun noSpeechTimeoutExplainsTheStopAndKeepsTheDraftWithoutSending() =
        withReplayPreferences(automatic = true) {
            withKeyboard(openPractice = false) {
                ActivityScenario.launch<EditorCompatibilityActivity>(
                        Intent(context, EditorCompatibilityActivity::class.java)
                    )
                    .use { host ->
                        awaitNode { it.contentDescription?.toString() == "Start dictating" }
                        setText("Keep this draft")
                        click("Start dictating")
                        awaitNode { it.text?.toString() == "Listening" }
                        val message =
                            awaitNode(timeoutMs = 30_000) {
                                it.text?.toString() == "No speech detected. Tap Start to try again."
                            }
                        assertEquals(
                            android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE,
                            message.liveRegion,
                        )
                        awaitNode { it.text?.toString() == "Ready" }
                        host.onActivity {
                            assertEquals("Keep this draft", it.field().text)
                            assertTrue("A no-speech timeout must not send", it.sent.isEmpty())
                        }
                    }
                // Reuse the same keyboard instance: a previous timeout must not
                // hide the explanation for a newly focused password field.
                ActivityScenario.launch<EditorCompatibilityActivity>(
                        Intent(context, EditorCompatibilityActivity::class.java)
                            .putExtra("password", true)
                    )
                    .use {
                        awaitNode {
                            it.text?.toString() ==
                                "Tap to dictate in this field. Automatic start is off."
                        }
                        awaitNode { it.text?.toString() == "Ready" }
                    }
            }
        }

    private fun withReplayPreferences(automatic: Boolean = false, test: () -> Unit) = runBlocking {
        val repository = PreferencesRepository(context)
        val saved = repository.values.first()
        try {
            assertEquals(
                android.content.pm.PackageManager.PERMISSION_GRANTED,
                context.checkSelfPermission(Manifest.permission.RECORD_AUDIO),
            )
            repository.set(Preference.Sounds, false)
            repository.set(Preference.AutoSend, automatic)
            repository.set(Preference.StopAtPause, false)
            repository.set(Preference.SwitchBack, false)
            repository.set(
                Preference.LiveText,
                InstrumentationRegistry.getArguments().getString("liveText") != "false",
            )
            repository.setSendTiming(dev.soupslurpr.transcribro.preferences.SendTiming.Phrase)
            repository.setPauseLength(
                dev.soupslurpr.transcribro.preferences.PauseLength.ThreeSeconds
            )
            test()
        } finally {
            repository.set(Preference.Sounds, saved.sounds)
            repository.set(Preference.AutoSend, saved.autoSend)
            repository.set(Preference.StopAtPause, saved.stopAtPause)
            repository.set(Preference.SwitchBack, saved.switchBack)
            repository.set(Preference.LiveText, saved.liveText)
            repository.setSendTiming(saved.sendTiming)
            repository.setPauseLength(saved.pauseLength)
        }
    }

    /** Only the opt-in host runner injects PCM. This never substitutes a recognizer or editor. */
    private fun replayMicrophone() {
        val token = UUID.randomUUID().toString()
        val complete = CountDownLatch(1)
        val action = "dev.soupslurpr.transcribro.MICROPHONE_REPLAY_COMPLETE"
        val receiver =
            object : BroadcastReceiver() {
                override fun onReceive(context: Context, intent: Intent) {
                    if (intent.getStringExtra("token") == token) complete.countDown()
                }
            }
        context.registerReceiver(receiver, IntentFilter(action), Context.RECEIVER_EXPORTED)
        try {
            instrumentation.sendStatus(2, Bundle().apply { putString("transcribro_replay", token) })
            assertTrue(
                "Host did not complete the microphone replay",
                complete.await(45, TimeUnit.SECONDS),
            )
        } finally {
            context.unregisterReceiver(receiver)
        }
    }

    private fun assertCounting(text: String) {
        val words =
            listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        val count =
            Regex("[a-z]+|[0-9]+")
                .findAll(text.lowercase())
                .map { it.value.toIntOrNull() ?: (words.indexOf(it.value) + 1) }
                .toList()
        assertEquals(text, (1..10).toList(), count)
        instrumentation.sendStatus(2, Bundle().apply { putString("transcribro_result", text) })
    }

    @Test
    fun launcherSwitchListsTranscribroWithTheKeyboardInitiallyHidden() =
        withKeyboard(selectVoice = false, openPractice = false) { scenario ->
            awaitIme(scenario, false)
            click("Switch to Transcribro")
            chooseTranscribroFromSystemPicker()
            awaitNode { it.contentDescription?.toString() == "Backspace" }
            assertEquals(
                "${context.packageName}/dev.soupslurpr.transcribro.ui.voiceinput.VoiceInput",
                secureSetting(Settings.Secure.DEFAULT_INPUT_METHOD),
            )
        }

    @Test
    fun practiceSwitchOpensThePickerOnceAfterShowingTheKeyboard() =
        withKeyboard(selectVoice = false, openPractice = false) { scenario ->
            click("Try dictation")
            awaitNode { it.className == "android.widget.EditText" }
            awaitIme(scenario, false)
            click("Choose Transcribro")
            awaitNode(packageName = "com.android.systemui") {
                it.text?.toString() == context.getString(R.string.app_name)
            }
            shell("input keyevent KEYCODE_BACK")
            awaitIme(scenario, true)
            shell("input keyevent KEYCODE_BACK")
            awaitIme(scenario, false)
            awaitNode { it.className == "android.widget.EditText" }
                .performAction(AccessibilityNodeInfo.ACTION_CLICK)
            awaitIme(scenario, true)
            instrumentation.waitForIdleSync()
            assertEquals(
                null,
                findInWindows(
                    { it.text?.toString() == context.getString(R.string.app_name) },
                    "com.android.systemui",
                ),
            )
        }

    private fun chooseTranscribroFromSystemPicker() {
        val name = context.getString(R.string.app_name)
        val heading =
            awaitNode(packageName = "com.android.systemui") { it.text?.toString() == name }
        // Android 17 groups subtype buttons beneath the input method's heading.
        var afterHeading = false
        fun option(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
            if (node == null || node.packageName?.toString() != "com.android.systemui") return null
            if (node.text?.toString() == name) afterHeading = true
            if (afterHeading && node.isClickable) return node
            repeat(node.childCount) { index ->
                option(node.getChild(index))?.let {
                    return it
                }
            }
            return null
        }
        val choice =
            checkNotNull(option(heading.window.root)) { "Transcribro has no selectable subtype" }
        assertTrue(choice.performAction(AccessibilityNodeInfo.ACTION_CLICK))
    }

    private fun awaitIme(scenario: ActivityScenario<MainActivity>, visible: Boolean) {
        val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            var actual = !visible
            var focused = false
            scenario.onActivity {
                actual =
                    it.window.decorView.rootWindowInsets?.isVisible(WindowInsets.Type.ime()) == true
                focused = it.hasWindowFocus()
            }
            if (actual == visible && focused) return
            Thread.sleep(25)
        }
        error("IME visibility never became $visible")
    }

    @Test
    fun holdingBackspaceRepeatsAndReleaseStops() = withKeyboard {
        setText("abcdefghijklmnopqrstuvwx")
        select(24, 24)
        var key = awaitNode { it.contentDescription?.toString() == "Backspace" }
        while (!key.isClickable && key.parent != null) key = key.parent
        val bounds = Rect().also(key::getBoundsInScreen)
        val down = android.os.SystemClock.uptimeMillis()
        touch(down, down, MotionEvent.ACTION_DOWN, bounds.exactCenterX(), bounds.exactCenterY())
        Thread.sleep(ViewConfiguration.getLongPressTimeout().toLong() + 350)
        touch(
            down,
            android.os.SystemClock.uptimeMillis(),
            MotionEvent.ACTION_UP,
            bounds.exactCenterX(),
            bounds.exactCenterY(),
        )
        Thread.sleep(150)
        val field = awaitNode { it.className == "android.widget.EditText" }
        val remaining = field.text.toString()
        assertTrue("Hold must delete repeatedly", remaining.length in 1..21)
        Thread.sleep(200)
        awaitText(remaining)
    }

    @Test
    fun talkBackCanFocusAndActivateSpace() = withKeyboard {
        withTouchExploration {
            setText("Voice")
            select(5, 5)
            var key = awaitNode { it.contentDescription?.toString() == "Space" }
            while (!key.isClickable && key.parent != null) key = key.parent
            assertTrue(key.performAction(AccessibilityNodeInfo.ACTION_ACCESSIBILITY_FOCUS))
            assertTrue(key.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            awaitText("Voice ")
        }
    }

    private fun withTouchExploration(test: () -> Unit) {
        val previous = secureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
        val master = secureSetting(Settings.Secure.ACCESSIBILITY_ENABLED)
        try {
            val service =
                "com.google.android.marvin.talkback/com.google.android.marvin.talkback.TalkBackService"
            setSecureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES, service)
            setSecureSetting(Settings.Secure.ACCESSIBILITY_ENABLED, "1")
            val manager =
                context.getSystemService(
                    android.view.accessibility.AccessibilityManager::class.java
                )
            val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
            while (
                !manager.isTouchExplorationEnabled &&
                    android.os.SystemClock.elapsedRealtime() < deadline
            ) Thread.sleep(50)
            assertTrue("TalkBack touch exploration didn't start", manager.isTouchExplorationEnabled)
            test()
        } finally {
            setSecureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES, previous)
            setSecureSetting(Settings.Secure.ACCESSIBILITY_ENABLED, master)
        }
    }

    private fun touch(down: Long, time: Long, action: Int, x: Float, y: Float) {
        // Material tooltips distinguish fingers and styli from mouse/unknown pointers.
        // The short obtain() overload leaves the tool type unknown even with a touch source.
        val pointer =
            MotionEvent.PointerProperties().apply {
                id = 0
                toolType = MotionEvent.TOOL_TYPE_FINGER
            }
        val coordinates =
            MotionEvent.PointerCoords().apply {
                this.x = x
                this.y = y
                pressure = 1f
                size = 1f
            }
        val event =
            MotionEvent.obtain(
                down,
                time,
                action,
                1,
                arrayOf(pointer),
                arrayOf(coordinates),
                0,
                0,
                1f,
                1f,
                0,
                0,
                InputDevice.SOURCE_TOUCHSCREEN,
                0,
            )
        try {
            assertTrue(automation.injectInputEvent(event, true))
        } finally {
            event.recycle()
        }
    }

    @Test
    fun switchAccessCanReachAndActivateCorrectionControls() = withKeyboard { scenario ->
        val previous = secureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
        val master = secureSetting(Settings.Secure.ACCESSIBILITY_ENABLED)
        try {
            val service =
                "com.google.android.accessibility.switchaccess/com.android.switchaccess.SwitchAccessService"
            setSecureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES, service)
            setSecureSetting(Settings.Secure.ACCESSIBILITY_ENABLED, "1")
            val manager =
                context.getSystemService(
                    android.view.accessibility.AccessibilityManager::class.java
                )
            val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
            fun bound() =
                manager
                    .getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
                    .any { it.id == service }
            while (!bound() && android.os.SystemClock.elapsedRealtime() < deadline) Thread.sleep(50)
            assertTrue("Switch Access didn't bind", bound())
            // First activation may launch the system's switch-setup activity.
            Thread.sleep(1_000)
            scenario.onActivity {
                it.startActivity(
                    Intent(it, MainActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
                )
            }
            awaitNode { it.className == "android.widget.EditText" }
                .performAction(AccessibilityNodeInfo.ACTION_CLICK)
            awaitNode { it.contentDescription?.toString() == "Backspace" }
            for (label in
                listOf(
                    "Undo",
                    "Redo",
                    "Backspace",
                    "Move cursor left",
                    "Move cursor right",
                    "Space",
                    "Insert newline",
                )) {
                var key = awaitNode { it.contentDescription?.toString() == label }
                while (!key.isClickable && key.parent != null) key = key.parent
                assertTrue(
                    "Missing activation action: $label",
                    key.actionList.contains(AccessibilityNodeInfo.AccessibilityAction.ACTION_CLICK),
                )
                val bounds = Rect().also(key::getBoundsInScreen)
                val minimum = context.resources.displayMetrics.density * 48 - 1
                assertTrue(
                    "Small touch target: $label",
                    bounds.width() >= minimum && bounds.height() >= minimum,
                )
            }
            setText("Correction")
            select(10, 10)
            click("Backspace")
            awaitText("Correctio")
        } finally {
            setSecureSetting(Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES, previous)
            setSecureSetting(Settings.Secure.ACCESSIBILITY_ENABLED, master)
        }
    }

    @Test
    fun deletesWholeGraphemesAndSupportsCursorEditing() = withKeyboard {
        val source = "A👍🏽" + "👩‍👩‍👧‍👦" + "e\u0301"
        setText(source)
        select(source.length, source.length)
        click("Backspace")
        awaitText("A👍🏽👩‍👩‍👧‍👦")
        click("Backspace")
        awaitText("A👍🏽")
        click("Backspace")
        awaitText("A")
        click("Space")
        awaitText("A ")
        click("Move cursor left")
        awaitNode {
            it.className == "android.widget.EditText" &&
                it.textSelectionStart == 1 &&
                it.textSelectionEnd == 1
        }
        click("Backspace")
        awaitText(" ")
        select(0, 1)
        click("Backspace")
        awaitText("")
    }

    @Test
    fun reusedLauncherOpensRequestedSettingsAndDoesNotReplayTheRequestOnRotation() =
        withKeyboard { scenario ->
            setText("Temporary practice text")
            scenario.onActivity {
                it.startActivity(
                    Intent(it, MainActivity::class.java)
                        .setAction(Intent.ACTION_APPLICATION_PREFERENCES)
                        .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
                )
            }
            awaitNode { it.text?.toString() == "Live text" }
            scenario.onActivity {
                assertEquals("", ViewModelProvider(it)[MainViewModel::class.java].testInput.text)
            }
            // Android's still-disappearing IME navigation button also says "Back".
            click("Back", android.view.accessibility.AccessibilityWindowInfo.TYPE_APPLICATION)
            // Home can retain an audio result, changing its file action label.
            awaitNode { it.isHeading && it.text?.toString() == "Transcribro" }
            scenario.recreate()
            awaitNode { it.isHeading && it.text?.toString() == "Transcribro" }
        }

    @Test
    fun practiceSurvivesRecreationAndIsClearedOnLeavingTheTest() = withKeyboard { scenario ->
        val text = "Keep this practice text through editing and rotation."
        setText(text)
        click("Undo")
        awaitText("")
        // Empty/nonempty layout changes must not destroy the input connection.
        awaitNode { it.contentDescription?.toString() == "Backspace" }
        click("Redo")
        awaitText(text)
        scenario.recreate()
        awaitText(text)
        val deadline = android.os.SystemClock.elapsedRealtime() + 5_000
        var focused = false
        while (!focused && android.os.SystemClock.elapsedRealtime() < deadline) {
            scenario.onActivity { focused = it.hasWindowFocus() }
            if (!focused) Thread.sleep(25)
        }
        assertTrue("Recreated editor never gained window focus", focused)
        scenario.onActivity {
            val model = ViewModelProvider(it)[MainViewModel::class.java]
            assertEquals(text, model.testInput.text)
        }
        awaitNode { it.className == "android.widget.EditText" }
            .performAction(AccessibilityNodeInfo.ACTION_CLICK)
        awaitNode { it.contentDescription?.toString() == "Backspace" }
        shell("input keyevent KEYCODE_BACK KEYCODE_BACK")
        // Both rapid events may legitimately be consumed while the IME is hiding.
        // Wait for the actual inset transition before the final Back assertion.
        val hideDeadline = android.os.SystemClock.elapsedRealtime() + 5_000
        var keyboardVisible = true
        while (keyboardVisible && android.os.SystemClock.elapsedRealtime() < hideDeadline) {
            scenario.onActivity {
                keyboardVisible =
                    it.window.decorView.rootWindowInsets?.isVisible(WindowInsets.Type.ime()) == true
            }
            if (keyboardVisible) Thread.sleep(25)
        }
        assertEquals("Keyboard remained visible after Back", false, keyboardVisible)
        // The introduction returns only after Compose's IME animation is complete.
        val destination = awaitNode {
            it.text?.toString() == "A little room to try it." ||
                it.isHeading && it.text?.toString() == "Transcribro"
        }
        if (destination.text?.toString() == "A little room to try it.") {
            shell("input keyevent KEYCODE_BACK")
        }
        awaitNode { it.isHeading && it.text?.toString() == "Transcribro" }
        scenario.onActivity {
            assertEquals("", ViewModelProvider(it)[MainViewModel::class.java].testInput.text)
        }
        click("Try dictation")
        awaitText("")
    }

    @Test
    fun primaryActionRetainsAccessibilityFocusWhileListening() = withReplayPreferences {
        withKeyboard {
            withTouchExploration {
                setText("Keep these words")
                var start = awaitNode { it.contentDescription?.toString() == "Start dictating" }
                while (!start.isClickable && start.parent != null) start = start.parent
                val bounds = Rect().also(start::getBoundsInScreen)
                assertTrue(start.performAction(AccessibilityNodeInfo.ACTION_ACCESSIBILITY_FOCUS))
                captureKeyboard("ready")
                assertTrue(start.performAction(AccessibilityNodeInfo.ACTION_CLICK))
                val status = awaitNode { it.text?.toString() == "Listening" }
                assertEquals(android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE, status.liveRegion)
                var finish = awaitNode { it.contentDescription?.toString() == "Finish dictation" }
                while (!finish.isClickable && finish.parent != null) finish = finish.parent
                assertEquals("The primary action must keep its accessible identity", start, finish)
                assertTrue(
                    "Changing state must preserve accessible focus",
                    finish.isAccessibilityFocused,
                )
                assertEquals(
                    "Start and Finish must keep their position",
                    bounds,
                    Rect().also(finish::getBoundsInScreen),
                )
                captureKeyboard("listening")
                click("Cancel")
                awaitNode { it.text?.toString() == "Ready" }
                awaitText("Keep these words")
            }
        }
    }

    @Test
    fun editingControlsClearSystemNavigationButtons() = withKeyboard {
        for (navigation in listOf("Back", "Switch input method")) {
            val button = awaitNode { it.contentDescription?.toString() == navigation }
            val navigationBounds = Rect().also(button::getBoundsInScreen)
            for (label in listOf("Move cursor left", "Space", "Insert newline")) {
                val key = awaitNode { it.contentDescription?.toString() == label }
                assertTrue(
                    "$label overlaps Android's $navigation button",
                    !Rect.intersects(navigationBounds, Rect().also(key::getBoundsInScreen)),
                )
            }
        }
    }

    @Test
    fun scrollingKeepsActionsReachableOnAShortScreen() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("shortScreen") == "true")
        withReplayPreferences(automatic = true) {
            withKeyboard(openPractice = false) {
                ActivityScenario.launch<EditorCompatibilityActivity>(
                        Intent(context, EditorCompatibilityActivity::class.java)
                    )
                    .use { host ->
                        awaitNode { it.contentDescription?.toString() == "Start dictating" }
                        host.onActivity {
                            it.input!!.commitText("Keep these words", 1)
                            it.input!!.setSelection(5, 10)
                        }
                        awaitNode { it.text?.toString() == "Keep selected text" }
                        captureKeyboard("scroll-top")
                        fun scroll(action: AccessibilityNodeInfo.AccessibilityAction) {
                            val container = awaitNode {
                                it.isScrollable && it.actionList.contains(action)
                            }
                            assertTrue(container.performAction(action.id))
                            instrumentation.waitForIdleSync()
                        }
                        scroll(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_FORWARD)
                        awaitNode {
                            it.contentDescription?.toString() == "Send" && it.isVisibleToUser
                        }
                        captureKeyboard("scroll-bottom")
                        click("Insert newline")
                        awaitText("Keep \n words")
                        click("Send")
                        instrumentation.waitForIdleSync()
                        host.onActivity { assertEquals(listOf("Keep \n words"), it.sent) }
                        scroll(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_BACKWARD)
                        awaitNode {
                            it.contentDescription?.toString() == "Start dictating" &&
                                it.isVisibleToUser
                        }
                    }
            }
        }
    }

    @Test
    fun namedEditorActionsReachTheReceivingField() =
        withKeyboard(openPractice = false) {
            var firstRowBounds: List<Rect>? = null
            for (action in
                listOf(
                    EditorInfo.IME_ACTION_SEND,
                    EditorInfo.IME_ACTION_SEARCH,
                    EditorInfo.IME_ACTION_GO,
                    EditorInfo.IME_ACTION_NEXT,
                    EditorInfo.IME_ACTION_PREVIOUS,
                    EditorInfo.IME_ACTION_DONE,
                    EditorInfo.IME_ACTION_NONE,
                    EditorInfo.IME_ACTION_UNSPECIFIED,
                )) {
                ActivityScenario.launch<EditorCompatibilityActivity>(
                        Intent(context, EditorCompatibilityActivity::class.java)
                            .putExtra("action", action)
                    )
                    .use { host ->
                        awaitNode { it.contentDescription?.toString() == "Start dictating" }
                        val enabled =
                            action != EditorInfo.IME_ACTION_NONE &&
                                action != EditorInfo.IME_ACTION_UNSPECIFIED
                        val label =
                            if (enabled) VoiceInput.actionLabel(action) else "No editor action"
                        val actionKey = awaitNode { it.contentDescription?.toString() == label }
                        instrumentation.waitForIdleSync()
                        Thread.sleep(400) // Wait for Android's keyboard opening animation.
                        val rowBounds =
                            listOf(
                                    "Move cursor left",
                                    "Move cursor right",
                                    "Space",
                                    "Insert newline",
                                    label,
                                )
                                .map { description ->
                                    Rect().also(
                                        awaitNode { it.contentDescription?.toString() == description }
                                            ::getBoundsInScreen
                                    )
                                }
                        val actionBounds = rowBounds.last()
                        val target = context.resources.displayMetrics.density * 48
                        assertTrue(
                            "Editor action $action must keep its 48dp square target: $actionBounds",
                            kotlin.math.abs(actionBounds.width() - target) <= 1 &&
                                kotlin.math.abs(actionBounds.height() - target) <= 1,
                        )
                        assertTrue(
                            "Editor action $action must stay on the same row as Space and Newline",
                            rowBounds.all {
                                it.top == actionBounds.top && it.bottom == actionBounds.bottom
                            },
                        )
                        firstRowBounds?.let {
                            assertEquals(
                                "Changing the editor action must not move any key",
                                it,
                                rowBounds,
                            )
                        }
                        firstRowBounds = rowBounds
                        val space = awaitNode { it.contentDescription?.toString() == "Space" }
                        val bounds = Rect().also(space::getBoundsInScreen)
                        val minimum = context.resources.displayMetrics.density * 48 - 1
                        assertTrue(
                            "Editor action $action must leave a full Space touch target",
                            bounds.width() >= minimum && bounds.height() >= minimum,
                        )
                        if (!enabled) {
                            assertTrue(!actionKey.isEnabled)
                            host.onActivity { assertTrue(it.performedActions.isEmpty()) }
                        } else {
                            host.onActivity { it.input!!.commitText("Words for this field", 1) }
                            awaitText("Words for this field")
                            captureKeyboard("action-${VoiceInput.actionLabel(action).lowercase()}")
                            click(VoiceInput.actionLabel(action))
                            instrumentation.waitForIdleSync()
                            host.onActivity { assertEquals(listOf(action), it.performedActions) }
                        }
                    }
            }
        }

    @Test
    fun holdingEditorActionShowsItsNameWithoutSending() =
        withKeyboard(openPractice = false) {
            ActivityScenario.launch<EditorCompatibilityActivity>(
                    Intent(context, EditorCompatibilityActivity::class.java)
                        .putExtra("action", EditorInfo.IME_ACTION_SEND)
                )
                .use { host ->
                    awaitNode { it.contentDescription?.toString() == "Start dictating" }
                    host.onActivity { it.input!!.commitText("Keep until tapped", 1) }
                    awaitText("Keep until tapped")
                    Thread.sleep(400) // Pointer coordinates must describe the settled keyboard.
                    val key = awaitNode { it.contentDescription?.toString() == "Send" }
                    val bounds = Rect().also(key::getBoundsInScreen)
                    val down = SystemClock.uptimeMillis()
                    touch(
                        down,
                        down,
                        MotionEvent.ACTION_DOWN,
                        bounds.exactCenterX(),
                        bounds.exactCenterY(),
                    )
                    try {
                        awaitNode { it.text?.toString() == "Send" && it.isVisibleToUser }
                        captureKeyboard("send-tooltip")
                        host.onActivity { assertTrue(it.performedActions.isEmpty()) }
                    } finally {
                        touch(
                            down,
                            SystemClock.uptimeMillis(),
                            MotionEvent.ACTION_UP,
                            bounds.exactCenterX(),
                            bounds.exactCenterY(),
                        )
                    }
                    instrumentation.waitForIdleSync()
                    host.onActivity { assertTrue(it.performedActions.isEmpty()) }
                    awaitText("Keep until tapped")
                    Thread.sleep(1_600) // Let the transient tooltip dismiss before a normal tap.
                    val tap = SystemClock.uptimeMillis()
                    touch(
                        tap,
                        tap,
                        MotionEvent.ACTION_DOWN,
                        bounds.exactCenterX(),
                        bounds.exactCenterY(),
                    )
                    touch(
                        tap,
                        SystemClock.uptimeMillis(),
                        MotionEvent.ACTION_UP,
                        bounds.exactCenterX(),
                        bounds.exactCenterY(),
                    )
                    instrumentation.waitForIdleSync()
                    host.onActivity {
                        assertEquals(listOf(EditorInfo.IME_ACTION_SEND), it.performedActions)
                        assertEquals(listOf("Keep until tapped"), it.sent)
                    }
                }
        }

    @Test
    fun selectionClearAndNewlineRemainDirectlyAccessible() = withKeyboard {
        setText("Before selected after")
        select(7, 15)
        captureKeyboard("selection")
        click("Keep selected text")
        awaitText("selected")
        select(8, 8)
        click("Insert newline")
        awaitText("selected\n")
        click("Clear field")
        awaitText("")
    }

    @Test
    fun autoSendTimingRemainsVisibleAndOpensSettings() = runBlocking {
        val repository = PreferencesRepository(context)
        val saved = repository.values.first()
        try {
            repository.set(Preference.AutoSend, true)
            repository.setSendTiming(SendTiming.Phrase)
            withKeyboard(openPractice = false) {
                EditorTestHost("compose").use {
                    val phrase = awaitNode {
                        it.text?.toString()?.endsWith("After each phrase") == true
                    }
                    assertTrue(
                        "Auto-send timing must remain visible at large text sizes",
                        phrase.isVisibleToUser,
                    )
                    click("Start dictating")
                    val listening = awaitNode { it.text?.toString() == "Listening" }
                    assertEquals(
                        android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE,
                        listening.liveRegion,
                    )
                    captureKeyboard("auto-listening")
                    click("Cancel")
                    awaitNode { it.text?.toString() == "Ready" }
                    val args = InstrumentationRegistry.getArguments()
                    if (args.getString("captureScreenshots") == "true") {
                        instrumentation.waitForIdleSync()
                        Thread.sleep(500)
                        val bitmap = checkNotNull(automation.takeScreenshot())
                        try {
                            val prefix = args.getString("capturePrefix").orEmpty()
                            File(context.cacheDir, "${prefix}keyboard-auto-send.png")
                                .outputStream()
                                .use { output ->
                                    assertTrue(
                                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
                                    )
                                }
                        } finally {
                            bitmap.recycle()
                        }
                    }
                    runBlocking { repository.setSendTiming(SendTiming.Session) }
                    assertTrue(
                        awaitNode { it.text?.toString()?.endsWith("When finished") == true }
                            .isVisibleToUser
                    )
                    click(
                        awaitNode { it.text?.toString()?.endsWith("When finished") == true }
                            .text
                            .toString()
                    )
                    awaitNode { it.text?.toString() == "Settings" }
                }
            }
        } finally {
            repository.set(Preference.AutoSend, saved.autoSend)
            repository.setSendTiming(saved.sendTiming)
        }
    }

    private fun captureKeyboard(name: String) {
        val args = InstrumentationRegistry.getArguments()
        if (args.getString("captureScreenshots") != "true") return
        instrumentation.waitForIdleSync()
        Thread.sleep(300)
        val windows = org.json.JSONArray()
        fun describe(node: AccessibilityNodeInfo): org.json.JSONObject {
            val bounds = Rect().also(node::getBoundsInScreen)
            val children = org.json.JSONArray()
            repeat(node.childCount) { index ->
                node.getChild(index)?.let { children.put(describe(it)) }
            }
            return org.json.JSONObject().apply {
                put("text", node.text?.toString())
                put("description", node.contentDescription?.toString())
                put(
                    "bounds",
                    org.json.JSONArray(
                        listOf(bounds.left, bounds.top, bounds.right, bounds.bottom)
                    ),
                )
                put("clickable", node.isClickable)
                put("enabled", node.isEnabled)
                put("focused", node.isAccessibilityFocused)
                put("children", children)
            }
        }
        automation.windows.forEach { it.root?.let { root -> windows.put(describe(root)) } }
        File(context.cacheDir, "${args.getString("capturePrefix").orEmpty()}keyboard-$name.json")
            .writeText(windows.toString(2))
        val bitmap = checkNotNull(automation.takeScreenshot())
        try {
            File(context.cacheDir, "${args.getString("capturePrefix").orEmpty()}keyboard-$name.png")
                .outputStream()
                .use {
                    assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it))
                }
        } finally {
            bitmap.recycle()
        }
    }

    private fun withKeyboard(
        selectVoice: Boolean = true,
        autoStart: Boolean = false,
        openPractice: Boolean = true,
        test: (ActivityScenario<MainActivity>) -> Unit,
    ) = runBlocking {
        val component = "${context.packageName}/dev.soupslurpr.transcribro.ui.voiceinput.VoiceInput"
        val previousInput = secureSetting(Settings.Secure.DEFAULT_INPUT_METHOD)
        val previousEnabled = secureSetting(Settings.Secure.ENABLED_INPUT_METHODS)
        val previousHardwareKeyboard = secureSetting("show_ime_with_hard_keyboard")
        val repository = PreferencesRepository(context)
        val preferences = repository.values.first()
        val oldServiceInfo = automation.serviceInfo
        try {
            repository.set(Preference.AutoStart, autoStart)
            automation.serviceInfo =
                automation.serviceInfo.apply {
                    flags = flags or AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
                }
            shell("input keyevent KEYCODE_WAKEUP")
            shell("wm dismiss-keyguard")
            shell("ime enable $component")
            val selected =
                if (selectVoice) component
                else
                    shell("ime list -a -s").lineSequence().map(String::trim).first {
                        ComponentName.unflattenFromString(it)?.className ==
                            "com.android.inputmethod.latin.LatinIME"
                    }
            shell("ime enable $selected")
            shell("ime set $selected")
            setSecureSetting("show_ime_with_hard_keyboard", "1")
            ActivityScenario.launch<MainActivity>(
                    Intent(context, MainActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                    }
                )
                .use { scenario ->
                    if (openPractice) {
                        click("Try dictation")
                        val field = awaitNode { it.className == "android.widget.EditText" }
                        field.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        awaitNode { it.contentDescription?.toString() == "Backspace" }
                    }
                    test(scenario)
                }
        } finally {
            shell("input keyevent KEYCODE_BACK")
            if (previousInput != null) shell("ime set $previousInput")
            setSecureSetting(Settings.Secure.ENABLED_INPUT_METHODS, previousEnabled)
            setSecureSetting("show_ime_with_hard_keyboard", previousHardwareKeyboard)
            repository.set(Preference.AutoStart, preferences.autoStart)
            automation.serviceInfo = oldServiceInfo
        }
    }

    private fun setText(text: String) {
        check(
            awaitNode { it.className == "android.widget.EditText" }
                .performAction(
                    AccessibilityNodeInfo.ACTION_SET_TEXT,
                    Bundle().apply {
                        putCharSequence(
                            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                            text,
                        )
                    },
                )
        )
        awaitText(text)
    }

    // Check Chromium's input connection as well as the DOM used for message assertions.
    private fun awaitHostText(
        host: EditorTestHost,
        expected: String,
        trailingPlaceholder: Boolean = false,
    ) {
        var actual = "<missing>"
        val deadline = SystemClock.elapsedRealtime() + 5_000
        while (SystemClock.elapsedRealtime() < deadline) {
            val parts = host.readParts()
            actual = parts.joinToString("")
            if (actual == expected) return
            // A trailing contenteditable <br> keeps its empty last line visible. It is
            // after the caret and disappears when the next line acquires real text.
            if (trailingPlaceholder && parts == listOf(expected, "", "\n")) return
            Thread.sleep(25)
        }
        assertEquals(host.snapshot().toString(), expected, actual)
    }

    private fun focusBrowserField() {
        val field = awaitNode { it.className == "android.widget.EditText" }
        val bounds = Rect().also(field::getBoundsInScreen)
        val now = SystemClock.uptimeMillis()
        for (action in listOf(MotionEvent.ACTION_DOWN, MotionEvent.ACTION_UP)) {
            val event =
                MotionEvent.obtain(
                        now,
                        SystemClock.uptimeMillis(),
                        action,
                        bounds.exactCenterX(),
                        bounds.exactCenterY(),
                        0,
                    )
                    .apply { source = InputDevice.SOURCE_TOUCHSCREEN }
            try {
                check(automation.injectInputEvent(event, true))
            } finally {
                event.recycle()
            }
        }
    }

    /** Waits for the editor to reflect the preceding asynchronous input operation. */
    private fun awaitText(text: String) {
        var observed = "<missing>"
        val deadline = android.os.SystemClock.elapsedRealtime() + 5_000
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            for (window in automation.windows) {
                val field = findNode(window.root) { it.className == "android.widget.EditText" }
                if (field != null) observed = field.text?.toString().orEmpty()
                if (observed == text) return
            }
            Thread.sleep(50)
        }
        assertEquals(text, observed)
    }

    /** Selects a UTF-16 range through the editor's accessibility contract. */
    private fun select(start: Int, end: Int) {
        val field = awaitNode { it.className == "android.widget.EditText" }
        if (field.textSelectionStart != start || field.textSelectionEnd != end) {
            assertEquals(
                true,
                field.performAction(
                    AccessibilityNodeInfo.ACTION_SET_SELECTION,
                    Bundle().apply {
                        putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_START_INT, start)
                        putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_END_INT, end)
                    },
                ),
            )
        }
        awaitNode {
            it.className == "android.widget.EditText" &&
                it.textSelectionStart == start &&
                it.textSelectionEnd == end
        }
        instrumentation.waitForIdleSync()
    }

    /** Clicks a labelled keyboard control through its accessible parent. */
    private fun click(
        label: String,
        windowType: Int? = null,
        packageName: String = context.packageName,
    ) {
        var node =
            awaitNode(packageName) {
                (it.contentDescription?.toString() == label || it.text?.toString() == label) &&
                    (windowType == null || it.window?.type == windowType)
            }
        while (!node.isClickable && node.parent != null) node = node.parent
        check(node.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
            "control is not clickable: $label"
        }
    }

    /** Searches the app by default, or Android's picker when explicitly requested. */
    private fun awaitNode(
        packageName: String = context.packageName,
        timeoutMs: Long = 10_000,
        predicate: (AccessibilityNodeInfo) -> Boolean,
    ): AccessibilityNodeInfo {
        val deadline = android.os.SystemClock.elapsedRealtime() + timeoutMs
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            findInWindows(predicate, packageName)?.let {
                return it
            }
            Thread.sleep(50)
        }
        error(
            "expected control did not appear in $packageName; controls: ${visibleLabels(packageName)}"
        )
    }

    private fun visibleLabels(packageName: String): List<String> {
        val visible = mutableListOf<String>()
        fun collect(node: AccessibilityNodeInfo?) {
            if (node == null || node.packageName?.toString() != packageName || !node.refresh())
                return
            node.text?.let { visible.add(it.toString()) }
            node.contentDescription?.let { visible.add(it.toString()) }
            repeat(node.childCount) { collect(node.getChild(it)) }
        }
        automation.windows.forEach { collect(it.root) }
        return visible
    }

    private fun findInWindows(
        predicate: (AccessibilityNodeInfo) -> Boolean,
        packageName: String = context.packageName,
    ): AccessibilityNodeInfo? {
        for (window in automation.windows) {
            findNode(window.root, packageName, predicate)?.let {
                return it
            }
        }
        return null
    }

    /** Traverses only the explicitly selected package's semantic tree. */
    private fun findNode(
        node: AccessibilityNodeInfo?,
        packageName: String = context.packageName,
        predicate: (AccessibilityNodeInfo) -> Boolean,
    ): AccessibilityNodeInfo? {
        if (node == null || node.packageName?.toString() != packageName) return null
        if (!node.refresh()) return null
        if (node.isVisibleToUser && predicate(node)) return node
        repeat(node.childCount) { child ->
            findNode(node.getChild(child), packageName, predicate)?.let {
                return it
            }
        }
        return null
    }

    private fun secureSetting(key: String): String? =
        shell("settings get secure $key").trim().takeUnless { it == "null" }

    private fun setSecureSetting(key: String, value: String?) {
        // executeShellCommand splits arguments itself; quoting would become literal data.
        automation.adoptShellPermissionIdentity(Manifest.permission.WRITE_SECURE_SETTINGS)
        try {
            check(Settings.Secure.putString(context.contentResolver, key, value))
        } finally {
            automation.dropShellPermissionIdentity()
        }
    }

    /** Runs test-device setup commands and closes the returned descriptor. */
    private fun shell(command: String): String =
        ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command)).use {
            it.bufferedReader().readText()
        }
}
