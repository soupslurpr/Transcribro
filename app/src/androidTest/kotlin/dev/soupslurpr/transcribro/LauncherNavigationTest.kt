package dev.soupslurpr.transcribro

import android.content.Intent
import android.os.SystemClock
import android.view.KeyEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.activity.BackEventCompat
import androidx.compose.ui.text.input.TextFieldValue
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LauncherNavigationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun practiceWordsStayVisibleDuringExitButDoNotReturn() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Try dictation" }
            scenario.onActivity { model(it).updateTestInput(TextFieldValue("Visible until the page leaves.")) }
            settle()
            withNavigationAnimationScale(6f) {
                // Leave enough time to inspect a rendered exit frame after model clearing.
                settle()
                startBack(scenario)
                scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
                scenario.onActivity { assertEquals("", model(it).testInput.text) }
                Thread.sleep(120)
                assertNotNull(
                    "Practice words disappeared before their page finished exiting",
                    findNode(currentRoot()) {
                        it.isEditable && it.text?.toString() == "Visible until the page leaves."
                    },
                )
                Thread.sleep(3000)
            }
            click { it.text?.toString() == "Try dictation" }
            screenNode { it.isEditable && it.text.isNullOrEmpty() }
            scenario.onActivity {
                model(it).undoTestInput()
                assertEquals("", model(it).testInput.text)
            }
        }
    }

    @Test
    fun keyboardHidingDuringBackDoesNotInventADiscardPrompt() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Try dictation" }
            click { it.isEditable }
            await(scenario) { imeVisible(it) }
            scenario.onActivity { model(it).updateTestInput(TextFieldValue("Keep this practice.")) }
            settle()
            startBack(scenario)
            scenario.onActivity {
                WindowInsetsControllerCompat(it.window, it.window.decorView)
                    .hide(WindowInsetsCompat.Type.ime())
            }
            await(scenario) { !imeVisible(it) }
            settle()
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            settle()
            assertNull(findNode(currentRoot()) {
                it.text?.toString() == "Discard this transcription?"
            })
            screenNode { it.isEditable && it.text?.toString() == "Keep this practice." }
            scenario.onActivity { assertEquals("Keep this practice.", model(it).testInput.text) }
        }
    }

    @Test
    fun successiveBackDuringExitRetiresBothPages() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Settings" }
            click { it.contentDescription?.toString() == "About" }
            click { it.text?.toString() == "Open-source licenses" }
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            Thread.sleep(80)
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            settle()
            screenNode { it.contentDescription?.toString() == "About" }
            click { it.contentDescription?.toString() == "Back" }
            screenNode { it.text?.toString() == "Try dictation" }
            click { it.text?.toString() == "About" }
            screenNode { it.text?.toString() == "About Transcribro" }
        }
    }

    @Test
    fun cancelledPredictiveBackKeepsPracticeAndCommittedBackClearsIt() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Try dictation" }
            screenNode { it.isEditable }
            scenario.onActivity { model(it).updateTestInput(TextFieldValue("Keep these words.")) }
            settle()
            startBack(scenario)
            scenario.onActivity { assertEquals("Keep these words.", model(it).testInput.text) }
            scenario.onActivity { it.onBackPressedDispatcher.dispatchOnBackCancelled() }
            settle()
            screenNode { it.isEditable && it.text?.toString() == "Keep these words." }
            scenario.recreate()
            screenNode { it.isEditable && it.text?.toString() == "Keep these words." }
            settle()
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            screenNode { it.text?.toString() == "Settings" }
            scenario.onActivity { assertEquals("", model(it).testInput.text) }
            click { it.text?.toString() == "Try dictation" }
            screenNode { it.isEditable && it.text.isNullOrEmpty() }
        }
    }

    @Test
    fun backDismissesKeyboardBeforeLeavingPractice() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Try dictation" }
            click { it.isEditable }
            await(scenario) { imeVisible(it) }
            scenario.onActivity { model(it).updateTestInput(TextFieldValue("Still practicing.")) }
            settle()
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.dispatchOnBackCancelled() }
            settle()
            scenario.onActivity {
                assertTrue(imeVisible(it))
                assertEquals("Still practicing.", model(it).testInput.text)
            }
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            await(scenario) { !imeVisible(it) }
            screenNode { it.isEditable }
            scenario.onActivity { assertEquals("Still practicing.", model(it).testInput.text) }
            settle()
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            screenNode { it.text?.toString() == "Settings" }
            scenario.onActivity { assertEquals("", model(it).testInput.text) }
        }
    }

    @Test
    fun licenseIndexHasReadableHeadingsAndRestoresPositionAndBackStack() {
        scenario().use { scenario ->
            click { it.text?.toString() == "Settings" }
            click { it.contentDescription?.toString() == "About" }
            click { it.text?.toString() == "Open-source licenses" }
            screenNode { it.isHeading && it.text?.toString() == "Transcribro and core libraries" }
            click { it.text?.toString() == "Jump to component" }
            scenario.recreate()
            screenNode { it.isHeading && it.text?.toString() == "Components" }
            // The modal's own Back handling must close the index, not leave the reader.
            settle()
            instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
            settle()
            screenNode { it.text?.toString() == "Jump to component" }
            click { it.text?.toString() == "Jump to component" }
            scrollTo { it.text?.toString() == "Protocol Buffers" }
            click { it.text?.toString() == "Protocol Buffers" }
            screenNode { it.isHeading && it.text?.toString() == "Protocol Buffers" }
            scenario.recreate()
            screenNode { it.isHeading && it.text?.toString() == "Protocol Buffers" }
            // A cancelled Back preview must preserve the selected component as well.
            settle()
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.dispatchOnBackCancelled() }
            settle()
            screenNode { it.isHeading && it.text?.toString() == "Protocol Buffers" }
            click { it.contentDescription?.toString() == "Back" }
            screenNode { it.text?.toString() == "About Transcribro" }
            click { it.contentDescription?.toString() == "Back" }
            screenNode { it.contentDescription?.toString() == "About" }
            click { it.contentDescription?.toString() == "Back" }
            screenNode { it.text?.toString() == "Settings" }
        }
    }

    @Test
    fun homeAboutReturnsHomeAfterOpeningLicensesAndRecreating() {
        scenario().use { scenario ->
            click { it.text?.toString() == "About" }
            click { it.text?.toString() == "Open-source licenses" }
            screenNode { it.text?.toString() == "Jump to component" }
            scenario.recreate()
            click { it.contentDescription?.toString() == "Back" }
            screenNode { it.text?.toString() == "About Transcribro" }
            settle()
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.dispatchOnBackCancelled() }
            settle()
            screenNode { it.text?.toString() == "About Transcribro" }
            startBack(scenario)
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            screenNode { it.text?.toString() == "Try dictation" }
            screenNode { it.text?.toString() == "Settings" }
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

    private fun imeVisible(activity: MainActivity) =
        ViewCompat.getRootWindowInsets(activity.window.decorView)
            ?.isVisible(WindowInsetsCompat.Type.ime()) == true

    private fun startBack(scenario: ActivityScenario<MainActivity>) {
        scenario.onActivity {
            it.onBackPressedDispatcher.dispatchOnBackStarted(
                BackEventCompat(0f, 800f, 0f, BackEventCompat.EDGE_LEFT)
            )
        }
        settle()
        scenario.onActivity {
            it.onBackPressedDispatcher.dispatchOnBackProgressed(
                BackEventCompat(300f, 800f, 0.6f, BackEventCompat.EDGE_LEFT)
            )
        }
        settle()
    }

    private fun settle() {
        instrumentation.waitForIdleSync()
        Thread.sleep(400)
    }

    private fun await(
        scenario: ActivityScenario<MainActivity>,
        condition: (MainActivity) -> Boolean,
    ) {
        val deadline = SystemClock.elapsedRealtime() + 10_000
        while (SystemClock.elapsedRealtime() < deadline) {
            var ready = false
            scenario.onActivity { ready = condition(it) }
            if (ready) return
            Thread.sleep(50)
        }
        fail("The expected navigation state did not appear")
    }

    private fun click(predicate: (AccessibilityNodeInfo) -> Boolean) {
        // During a transition, the outgoing toolbar can share a label with an incoming button.
        settle()
        var node = screenNode(predicate)
        while (!node.isClickable && node.parent != null) node = node.parent
        assertTrue(node.performAction(AccessibilityNodeInfo.ACTION_CLICK))
        settle()
    }

    private fun scrollTo(predicate: (AccessibilityNodeInfo) -> Boolean) {
        // The sheet exposes its controls before its opening motion has finished.
        settle()
        repeat(12) {
            if (findNode(currentRoot(), predicate) != null) return
            val scrollable = screenNode { it.isScrollable }
            assertTrue(
                "Could not scroll the component index: $scrollable",
                scrollable.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD),
            )
            settle()
        }
        fail("The requested component was not in the license index")
    }

    private fun screenNode(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
        val deadline = SystemClock.elapsedRealtime() + 10_000
        while (SystemClock.elapsedRealtime() < deadline) {
            findNode(currentRoot(), predicate)?.let {
                return it
            }
            Thread.sleep(50)
        }
        val labels = mutableListOf<String>()
        findNode(currentRoot()) {
            (it.text ?: it.contentDescription)?.let { label -> labels += label.toString() }
            false
        }
        error("The expected navigation control did not appear. Visible labels: $labels")
    }

    private fun currentRoot(): AccessibilityNodeInfo? {
        val automation = instrumentation.uiAutomation
        // Recreated windows and scrolled lists can leave cached nodes behind.
        // Read the current hierarchy before deciding whether to scroll again.
        automation.clearCache()
        return automation.rootInActiveWindow
    }

    private fun findNode(
        node: AccessibilityNodeInfo?,
        predicate: (AccessibilityNodeInfo) -> Boolean,
    ): AccessibilityNodeInfo? {
        if (node == null || node.packageName?.toString() != context.packageName) return null
        if (node.isVisibleToUser && predicate(node)) return node
        repeat(node.childCount) {
            findNode(node.getChild(it), predicate)?.let {
                return it
            }
        }
        return null
    }
}
