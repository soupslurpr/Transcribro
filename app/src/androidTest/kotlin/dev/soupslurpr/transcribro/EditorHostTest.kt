package dev.soupslurpr.transcribro

import android.content.Intent
import android.os.SystemClock
import android.view.inputmethod.EditorInfo
import androidx.compose.ui.text.input.TextFieldValue
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.dictation.AutomaticSending
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.SendTiming
import dev.soupslurpr.transcribro.ui.voiceinput.EditorDictation
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EditorHostTest {
    @Test fun nativeEditorRevisesCancelsAndSendsSuccessiveCompletedPhrases() = exercise(false)

    @Test fun composeEditorRevisesCancelsAndSendsSuccessiveCompletedPhrases() = exercise(true)

    private fun exercise(compose: Boolean) = withIdleVoiceKeyboard {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        ActivityScenario.launch<EditorCompatibilityActivity>(
                Intent(context, EditorCompatibilityActivity::class.java)
                    .putExtra("compose", compose)
            )
            .use { scenario ->
                lateinit var editor: EditorDictation
                val sending = AutomaticSending()
                val preferences = Preferences(autoSend = true, sendTiming = SendTiming.Phrase)
                waitFor(scenario) { it.input != null }
                fun connect() {
                    scenario.onActivity { activity ->
                        editor =
                            EditorDictation(
                                activity.input!!,
                                true,
                                selection = {
                                    activity.field().selection.let { it.start to it.end }
                                },
                            ) { _, _ ->
                            }
                    }
                }
                connect()
                scenario.onActivity { assertTrue(editor.preview("A note")) }
                awaitText(scenario, "A note")
                acknowledge(scenario, editor)
                scenario.onActivity { assertTrue(editor.preview("A notebook")) }
                awaitText(scenario, "A notebook")
                assertFalse(sending.consume(preferences, phraseEnd = true, canSend = true))
                scenario.onActivity {
                    assertTrue(editor.commit("A notebook."))
                    sending.committedSpeech()
                }
                awaitText(scenario, "A notebook.")
                scenario.onActivity {
                    it.sendDelay = 100
                    assertTrue(sending.consume(preferences, phraseEnd = true, canSend = true))
                    assertTrue(it.input!!.performEditorAction(EditorInfo.IME_ACTION_SEND))
                }
                awaitText(scenario, "")
                scenario.onActivity { assertEquals(listOf("A notebook."), it.sent) }
                connect()
                scenario.onActivity { assertTrue(editor.preview("Second")) }
                awaitText(scenario, "Second")
                acknowledge(scenario, editor)
                scenario.onActivity {
                    assertTrue(editor.commit("Second message."))
                    sending.committedSpeech()
                }
                awaitText(scenario, "Second message.")
                scenario.onActivity {
                    assertTrue(sending.consume(preferences, phraseEnd = true, canSend = true))
                    assertTrue(it.input!!.performEditorAction(EditorInfo.IME_ACTION_SEND))
                }
                awaitText(scenario, "")
                scenario.onActivity {
                    assertEquals(listOf("A notebook.", "Second message."), it.sent)
                }
                connect()
                scenario.onActivity { assertTrue(editor.commit("Before selected after")) }
                awaitText(scenario, "Before selected after")
                scenario.onActivity { it.input!!.setSelection(7, 15) }
                waitFor(scenario) {
                    it.field().selection.min == 7 && it.field().selection.max == 15
                }
                scenario.onActivity { assertTrue(editor.preview("replacement")) }
                awaitText(scenario, "Before replacement after")
                scenario.onActivity { assertTrue(editor.cancel()) }
                awaitText(scenario, "Before selected after")
                waitFor(scenario) {
                    it.field().selection.min == 7 && it.field().selection.max == 15
                }
                scenario.onActivity {
                    it.input!!.setSelection(0, it.field().text.length)
                    it.input!!.commitText("put the red notebook here", 1)
                }
                awaitText(scenario, "put the red notebook here")
                scenario.onActivity { it.input!!.setSelection(8, 11) }
                waitFor(scenario) {
                    it.field().selection.min == 8 && it.field().selection.max == 11
                }
                scenario.onActivity { assertTrue(editor.preview("Blue.")) }
                awaitText(scenario, "put the blue notebook here")
                acknowledge(scenario, editor)
                scenario.onActivity { assertTrue(editor.commit("Blue.")) }
                awaitText(scenario, "put the blue notebook here")
                scenario.onActivity { it.input!!.setSelection(0, it.field().text.length) }
                waitFor(scenario) {
                    it.field().selection.min == 0 &&
                        it.field().selection.max == it.field().text.length
                }
                scenario.onActivity { assertTrue(editor.commit("New sentence.")) }
                awaitText(scenario, "New sentence.")
            }
    }

    private fun acknowledge(
        scenario: ActivityScenario<EditorCompatibilityActivity>,
        editor: EditorDictation,
    ) {
        scenario.onActivity {
            val field = it.field()
            assertNotNull("Host must preserve composing spans", field.composition)
            editor.selectionChanged(
                field.selection.start,
                field.selection.end,
                field.composition!!.start,
                field.composition!!.end,
            )
        }
    }

    private fun awaitText(scenario: ActivityScenario<EditorCompatibilityActivity>, text: String) =
        waitFor(scenario) {
            val input = it.input
            it.field().text == text &&
                input != null &&
                (input.getTextBeforeCursor(1000, 0)?.toString().orEmpty() +
                    input.getSelectedText(0)?.toString().orEmpty() +
                    input.getTextAfterCursor(1000, 0)?.toString().orEmpty()) == text
        }

    private fun waitFor(
        scenario: ActivityScenario<EditorCompatibilityActivity>,
        ready: (EditorCompatibilityActivity) -> Boolean,
    ) {
        val deadline = SystemClock.elapsedRealtime() + 5_000
        var latest: TextFieldValue? = null
        while (SystemClock.elapsedRealtime() < deadline) {
            var done = false
            scenario.onActivity {
                done = ready(it)
                latest = it.field()
            }
            if (done) return
            Thread.sleep(25)
        }
        error("Editor didn't reach expected state; last state: $latest")
    }
}
