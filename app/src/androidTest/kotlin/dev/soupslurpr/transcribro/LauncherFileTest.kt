package dev.soupslurpr.transcribro

import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import android.view.KeyEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.activity.BackEventCompat
import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.audio.FileTranscriptions
import java.io.File
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LauncherFileTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val fixture =
        Uri.parse("content://dev.soupslurpr.transcribro.test.audio/dictation-stereo.wav")

    @Test
    fun fileResultIsSeparateFromPracticeAndExportAcknowledgementBelongsToOneFile() {
        scenario().use { scenario ->
            scenario.onActivity {
                model(it).updateTestInput(TextFieldValue("Practice stays separate."))
                model(it).importAudio(fixture)
            }
            val first = completedFile(scenario)
            assertTrue(first.text.contains("voice", ignoreCase = true))
            assertFalse(first.text.contains("Practice"))
            assertTrue(first.unpreserved)
            var open = screenNode { it.text?.toString() == "Open transcription" }
            while (!open.isClickable && open.parent != null) open = open.parent
            assertTrue(open.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            screenNode { it.text?.toString() == first.text }
            assertNull(
                "The result screen must not contain an editor",
                findNode(instrumentation.uiAutomation.rootInActiveWindow) { it.isEditable },
            )
            await(scenario) { it.playback.state.value.ready }
            if (InstrumentationRegistry.getArguments().getString("captureScreenshots") == "true") {
                instrumentation.waitForIdleSync()
                // Semantics are published before the emulator presents the frame.
                Thread.sleep(500)
                val bitmap = checkNotNull(instrumentation.uiAutomation.takeScreenshot())
                try {
                    File(context.cacheDir, "file-transcription-screen.png").outputStream().use {
                        assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it))
                    }
                } finally {
                    bitmap.recycle()
                }
            }
            // Returning Home keeps the result. Leaving the app still requires confirmation,
            // and cancelling a predictive gesture must not discard or open that confirmation.
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            screenNode { it.text?.toString() == "Open transcription" }
            instrumentation.waitForIdleSync()
            Thread.sleep(500)
            scenario.onActivity {
                it.onBackPressedDispatcher.dispatchOnBackStarted(
                    BackEventCompat(0f, 800f, 0f, BackEventCompat.EDGE_LEFT)
                )
                it.onBackPressedDispatcher.dispatchOnBackProgressed(
                    BackEventCompat(300f, 800f, 0.6f, BackEventCompat.EDGE_LEFT)
                )
                it.onBackPressedDispatcher.dispatchOnBackCancelled()
            }
            instrumentation.waitForIdleSync()
            assertNull(
                findNode(instrumentation.uiAutomation.rootInActiveWindow) {
                    it.text?.toString() == "Discard this transcription?"
                }
            )
            scenario.onActivity {
                assertEquals(first.id, model(it).file?.id)
                it.onBackPressedDispatcher.onBackPressed()
            }
            screenNode { it.text?.toString() == "Discard this transcription?" }
            var keep = screenNode { it.text?.toString() == "Keep transcription" }
            while (!keep.isClickable && keep.parent != null) keep = keep.parent
            assertTrue(keep.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            open = screenNode { it.text?.toString() == "Open transcription" }
            while (!open.isClickable && open.parent != null) open = open.parent
            assertTrue(open.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            screenNode { it.text?.toString() == first.text }

            val export = TextExport(first.id, first.text)
            scenario.recreate()
            scenario.onActivity {
                val model = model(it)
                assertEquals("Practice stays separate.", model.testInput.text)
                assertEquals(first.id, model.file?.id)
                model.markPreserved(export)
                assertFalse(model.file!!.unpreserved)
                model.importAudio(fixture)
            }
            val replacement = completedFile(scenario)
            assertNotEquals(first.id, replacement.id)
            assertEquals(first.text, replacement.text)
            scenario.onActivity {
                val model = model(it)
                model.markPreserved(export)
                assertTrue(
                    "An older export must not mark a replacement file preserved",
                    model.file!!.unpreserved,
                )
                model.clearFile()
                assertNull(model.file)
                assertEquals("Practice stays separate.", model.testInput.text)
            }
        }
        scenario().use { scenario ->
            scenario.onActivity {
                assertNull(model(it).file)
                assertEquals("", model(it).testInput.text)
            }
        }
    }

    @Test
    fun playbackPositionIsOneNamedAccessibleSliderAndCanSeek() {
        scenario().use { scenario ->
            scenario.onActivity { model(it).importAudio(fixture) }
            completedFile(scenario)
            var open = screenNode { it.text?.toString() == "Open transcription" }
            while (!open.isClickable && open.parent != null) open = open.parent
            assertTrue(open.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            await(scenario) { it.playback.state.value.ready }

            val slider = screenNode { it.className == "android.widget.SeekBar" }
            assertEquals("Audio position", slider.contentDescription?.toString())
            assertTrue(slider.stateDescription.toString().startsWith("0:00 of "))
            repeat(slider.childCount) { index ->
                assertNull(
                    "The slider label must not be a separate accessibility stop",
                    findNode(slider.getChild(index)) {
                        it.contentDescription?.toString() == "Audio position"
                    },
                )
            }
            assertTrue(
                slider.performAction(
                    AccessibilityNodeInfo.AccessibilityAction.ACTION_SET_PROGRESS.id,
                    Bundle().apply {
                        putFloat(AccessibilityNodeInfo.ACTION_ARGUMENT_PROGRESS_VALUE, 1_500f)
                    },
                )
            )
            await(scenario) { it.playback.state.value.positionMs == 1_500L }
            screenNode {
                it.className == "android.widget.SeekBar" &&
                    it.stateDescription.toString().startsWith("0:01 of ")
            }
            scenario.onActivity { assertFalse(model(it).playback.state.value.playing) }
        }
    }

    @Test
    fun discardedImportCannotReappearAndSourcePlaybackDoesNotAutostart() {
        scenario().use { scenario ->
            scenario.onActivity {
                val model = model(it)
                model.importAudio(fixture)
                model.clearFile()
            }
            instrumentation.waitForIdleSync()
            scenario.onActivity { model(it).playback.load(fixture) }
            await(scenario) { model ->
                model.playback.state.value.ready || model.playback.state.value.error != null
            }
            scenario.onActivity {
                val model = model(it)
                val state = model.playback.state.value
                assertNull(state.error)
                assertTrue(state.durationMs > 2_000)
                assertFalse(state.playing)
                model.playback.seek(1_500)
                assertEquals(1_500L, model.playback.state.value.positionMs)
                assertNull(model.file)
                model.playback.close()
                assertFalse(model.playback.state.value.ready)
            }
        }
    }

    @Test
    fun longTranscriptKeepsAllExportActionsVisibleAndCopiesTheWholeResult() {
        val text =
            (1..24).joinToString("\n\n") { paragraph ->
                "Part $paragraph. We will meet at the library on Saturday morning, then walk to the market. " +
                    "Bring a notebook for the ideas we want to discuss. We can review the plan together " +
                    "and decide what to do next."
            }
        scenario().use { scenario ->
            scenario.onActivity {
                val model = model(it)
                // A presentation fixture: complete the selected job on the main thread
                // before the service handles its queued start. Recognition is tested separately.
                model.importAudio(fixture)
                FileTranscriptions.updateActive(checkNotNull(model.file).id) { selected ->
                    selected.copy(
                        name = "Sample voice note.wav",
                        text = text,
                        active = false,
                        message = "Transcription complete.",
                    )
                }
            }
            clickText("Open transcription")
            await(scenario) { it.playback.state.value.ready }
            val before = exportBounds()
            capture("transcript-top")
            val scroll = screenNode { it.isScrollable }
            assertTrue(scroll.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD))
            instrumentation.waitForIdleSync()
            Thread.sleep(400)
            assertEquals("Export actions moved with the transcript", before, exportBounds())
            capture("transcript-scrolled")
            clickText("Copy")
            scenario.onActivity {
                val copied = it.getSystemService(ClipboardManager::class.java).primaryClip
                assertEquals(text, copied?.getItemAt(0)?.text?.toString())
                assertEquals(text, model(it).file?.preservedText)
            }
            // Recreation preserves the result and its directly available export controls.
            scenario.recreate()
            exportBounds()
            scenario.onActivity {
                assertEquals(text, model(it).file?.text)
                model(it).clearFile()
            }
        }
    }

    @Test
    fun cancelledRetryAndRecreationKeepFailedSaveRecoverable() {
        val firstWords = "We will meet at the library."
        val completedWords = "$firstWords Bring a notebook on Saturday."
        val output = File(context.cacheDir, "retried-transcription.txt")
        output.delete()
        try {
            scenario().use { scenario ->
                scenario.onActivity {
                    val model = model(it)
                    model.importAudio(fixture)
                    val id = checkNotNull(model.file).id
                    FileTranscriptions.updateActive(id) { selected ->
                        selected.copy(text = firstWords)
                    }
                    assertTrue(model.prepareSave())
                    assertFalse(
                        "A second picker must not overwrite the pending save",
                        model.prepareSave(),
                    )
                    // Complete more words after the initial snapshot, before the queued
                    // service start. Retrying must export these newly completed words too.
                    FileTranscriptions.updateActive(id) { selected ->
                        selected.copy(
                            name = "Sample voice note.wav",
                            text = completedWords,
                            active = false,
                            message = "Transcription complete.",
                        )
                    }
                    // The fixture provider is read-only: exercise a real output failure.
                    model.saveDestination(fixture)
                }
                await(scenario) { it.saveFailed && !it.saving }
                clickText("Open transcription")
                exportBounds()
                capture("save-recovery")
                revealText("Try saving again", lastScrollArea = true)
                clickText("Try saving again")
                val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
                while (
                    instrumentation.uiAutomation.rootInActiveWindow
                        ?.packageName
                        ?.toString()
                        ?.endsWith(".documentsui") != true &&
                        android.os.SystemClock.elapsedRealtime() < deadline
                ) {
                    Thread.sleep(50)
                }
                assertTrue(
                    "Retry must open Android's save picker",
                    instrumentation.uiAutomation.rootInActiveWindow
                        ?.packageName
                        ?.toString()
                        ?.endsWith(".documentsui") == true,
                )
                instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
                await(scenario) { !it.saving }
                scenario.recreate()
                screenNode {
                    it.text?.toString() ==
                        "Couldn't save the file. Your transcription is still here."
                }
                exportBounds()
                revealText("Choose another file")
                clickText("Choose another file")
                screenNode { it.text?.toString() == "Replace this transcription?" }
                assertTrue(screenNode { it.text?.toString() == "Choose file" }.isVisibleToUser)
                assertTrue(
                    screenNode { it.text?.toString() == "Keep transcription" }.isVisibleToUser
                )
                capture("replace-confirmation")
                clickText("Keep transcription")
                scenario.onActivity {
                    val model = model(it)
                    assertTrue(model.saveFailed)
                    assertEquals(completedWords, model.file?.text)
                    assertTrue(checkNotNull(model.file).unpreserved)
                    assertTrue(model.prepareSave())
                    model.saveDestination(Uri.fromFile(output))
                }
                await(scenario) { !it.saving }
                assertEquals(completedWords, output.readText())
                scenario.onActivity {
                    val model = model(it)
                    assertFalse(model.saveFailed)
                    assertFalse(checkNotNull(model.file).unpreserved)
                    assertEquals(completedWords, model.file?.text)
                    model.clearFile()
                }
            }
        } finally {
            output.delete()
        }
    }

    @Test
    fun dismissedFailureAndOlderWritesDoNotAffectAReplacement() {
        val output = File(context.cacheDir, "older-transcription.txt")
        try {
            scenario().use { scenario ->
                scenario.onActivity {
                    val model = model(it)
                    completePresentationFile(model, "Keep these words.")
                    assertTrue(model.prepareSave())
                    model.savePickerUnavailable()
                }
                clickText("Open transcription")
                revealText("Dismiss", lastScrollArea = true)
                clickText("Dismiss")
                scenario.onActivity {
                    val model = model(it)
                    assertFalse(model.saveFailed)
                    assertFalse(model.saving)
                    assertEquals("Keep these words.", model.file?.text)
                    assertTrue(checkNotNull(model.file).unpreserved)
                }
                // Both a failed and a successful older write can finish after a file
                // is replaced. Neither may change the replacement's recovery state.
                listOf(fixture, Uri.fromFile(output)).forEach { destination ->
                    scenario.onActivity {
                        val model = model(it)
                        assertTrue(model.prepareSave())
                        model.saveDestination(destination)
                        completePresentationFile(model, "This replacement is still unsaved.")
                    }
                    await(scenario) { !it.saving }
                    scenario.onActivity {
                        val model = model(it)
                        assertFalse(model.saveFailed)
                        assertTrue(checkNotNull(model.file).unpreserved)
                        assertEquals("This replacement is still unsaved.", model.file?.text)
                    }
                }
                scenario.onActivity { model(it).clearFile() }
            }
        } finally {
            output.delete()
        }
    }

    private fun completePresentationFile(model: MainViewModel, text: String) {
        model.importAudio(fixture)
        FileTranscriptions.updateActive(checkNotNull(model.file).id) { selected ->
            selected.copy(text = text, active = false, message = "Transcription complete.")
        }
    }

    private fun revealText(label: String, lastScrollArea: Boolean = false) {
        repeat(16) {
            val root = instrumentation.uiAutomation.rootInActiveWindow
            val target = findNode(root) { it.text?.toString() == label }
            if (target != null) {
                target.performAction(
                    AccessibilityNodeInfo.AccessibilityAction.ACTION_SHOW_ON_SCREEN.id
                )
                instrumentation.waitForIdleSync()
                Thread.sleep(400)
                return
            }
            val areas = mutableListOf<AccessibilityNodeInfo>()
            fun collect(node: AccessibilityNodeInfo?) {
                if (node == null) return
                if (node.isScrollable) areas.add(node)
                repeat(node.childCount) { collect(node.getChild(it)) }
            }
            collect(root)
            val area = if (lastScrollArea) areas.lastOrNull() else areas.firstOrNull()
            checkNotNull(area) { "No scroll area could reveal $label" }
                .performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
            instrumentation.waitForIdleSync()
            Thread.sleep(400)
        }
        error("Could not scroll to $label")
    }

    private fun exportBounds(): List<Rect> =
        listOf("Copy", "Share", "Save").map { label ->
            var button = screenNode { it.isVisibleToUser && it.text?.toString() == label }
            while (!button.isClickable && button.parent != null) button = button.parent
            assertTrue(
                "$label is not available",
                button.isVisibleToUser && button.isEnabled && button.isClickable,
            )
            val bounds = Rect().also(button::getBoundsInScreen)
            val window =
                Rect().also {
                    instrumentation.uiAutomation.rootInActiveWindow.getBoundsInScreen(it)
                }
            assertTrue("$label is outside the window", window.contains(bounds) && !bounds.isEmpty)
            bounds
        }

    private fun clickText(label: String) {
        var node = screenNode { it.isVisibleToUser && it.text?.toString() == label }
        while (!node.isClickable && node.parent != null) node = node.parent
        assertTrue(node.performAction(AccessibilityNodeInfo.ACTION_CLICK))
        instrumentation.waitForIdleSync()
        Thread.sleep(400)
    }

    private fun capture(name: String) {
        val args = InstrumentationRegistry.getArguments()
        if (args.getString("captureScreenshots") != "true") return
        instrumentation.waitForIdleSync()
        Thread.sleep(500)
        val bitmap = checkNotNull(instrumentation.uiAutomation.takeScreenshot())
        try {
            val prefix = args.getString("capturePrefix").orEmpty()
            File(context.cacheDir, "$prefix$name.png").outputStream().use { output ->
                assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun scenario() =
        ActivityScenario.launch<MainActivity>(
            Intent(context, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
        )

    private fun model(activity: MainActivity) =
        ViewModelProvider(activity)[MainViewModel::class.java]

    private fun findNode(
        node: AccessibilityNodeInfo?,
        predicate: (AccessibilityNodeInfo) -> Boolean,
    ): AccessibilityNodeInfo? {
        if (node == null || node.packageName?.toString() != context.packageName) return null
        if (predicate(node)) return node
        repeat(node.childCount) {
            findNode(node.getChild(it), predicate)?.let {
                return it
            }
        }
        return null
    }

    private fun screenNode(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
        val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            findNode(instrumentation.uiAutomation.rootInActiveWindow, predicate)?.let {
                return it
            }
            Thread.sleep(50)
        }
        error("The expected launcher control did not appear")
    }

    private fun completedFile(scenario: ActivityScenario<MainActivity>): FileTranscription {
        await(scenario) { it.file?.active == false }
        var result: FileTranscription? = null
        scenario.onActivity { result = model(it).file }
        return checkNotNull(result).also { assertFalse(it.message, it.failed) }
    }

    private fun await(
        scenario: ActivityScenario<MainActivity>,
        condition: (MainViewModel) -> Boolean,
    ) {
        val deadline = android.os.SystemClock.elapsedRealtime() + 60_000
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = condition(model(it)) }
            if (ready) return
            Thread.sleep(50)
        }
        fail("The launcher operation did not complete")
    }
}
