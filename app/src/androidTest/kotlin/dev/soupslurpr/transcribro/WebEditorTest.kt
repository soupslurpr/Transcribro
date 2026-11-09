package dev.soupslurpr.transcribro

import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.inputmethod.InputConnection
import androidx.compose.ui.text.input.TextFieldValue
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.ui.voiceinput.EditorDictation
import java.util.concurrent.CountDownLatch
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Chromium input connections are asynchronous and run on their own IME handler thread. */
@RunWith(AndroidJUnit4::class)
class WebEditorTest {
    @Test fun textInputRevisesFinalizesAndCancels() = exercise("input")

    @Test fun textAreaRevisesFinalizesAndCancels() = exercise("textarea")

    @Test fun richTextKeepsSurroundingMarkupWhenReplacingAndCancelling() = exercise("rich")

    private fun exercise(kind: String) = withIdleVoiceKeyboard {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        ActivityScenario.launch<EditorCompatibilityActivity>(
                Intent(context, EditorCompatibilityActivity::class.java)
                    .putExtra("host", "web-$kind")
            )
            .use { scenario ->
                lateinit var connection: InputConnection
                await {
                    var ready = false
                    scenario.onActivity {
                        it.input?.let { input ->
                            connection = input
                            ready = true
                        }
                    }
                    ready
                }
                val selected = AtomicReference(0 to 0)
                val editor =
                    EditorDictation(connection, true, selection = { selected.get() }) { _, _ -> }
                fun field(): TextFieldValue {
                    lateinit var value: TextFieldValue
                    scenario.onActivity { value = it.field() }
                    selected.set(value.selection.start to value.selection.end)
                    return value
                }
                fun <T> input(action: () -> T): T {
                    field()
                    val task = FutureTask(action)
                    check(Handler(connection.handler?.looper ?: Looper.getMainLooper()).post(task))
                    return task.get(5, TimeUnit.SECONDS)
                }
                fun text(expected: String) {
                    var observation = ""
                    await({ "Expected <$expected>; $observation" }) {
                        val actual = field().text
                        val surrounding = input {
                            connection.getTextBeforeCursor(1000, 0)?.toString().orEmpty() +
                                connection.getSelectedText(0)?.toString().orEmpty() +
                                connection.getTextAfterCursor(1000, 0)?.toString().orEmpty()
                        }
                        observation = "DOM=<$actual>, InputConnection=<$surrounding>"
                        actual == expected && surrounding == expected
                    }
                }
                fun acknowledge() {
                    val value = field()
                    assertNotNull(
                        "Chromium should expose an active DOM composition",
                        value.composition,
                    )
                    input {
                        assertTrue(
                            editor.selectionChanged(
                                value.selection.start,
                                value.selection.end,
                                value.composition!!.start,
                                value.composition!!.end,
                            )
                        )
                    }
                }

                input { assertTrue(editor.preview("A note")) }
                text("A note")
                acknowledge()
                input { assertTrue(editor.preview("A notebook")) }
                text("A notebook")
                // Final delivery must also work before acknowledging the latest preview.
                input { assertTrue(editor.commit("A notebook.")) }
                text("A notebook.")
                input { assertTrue(editor.commit("Second phrase.")) }
                text("A notebook. Second phrase.")

                // Recognition can change number formatting and length while the browser is
                // still applying an earlier preview. The final must replace that entire preview.
                val numbers = "One, two, three, four, five, six, seven, eight, nine, ten."
                repeat(5) {
                    scenario.onActivity { it.setWebContent("") }
                    text("")
                    input { assertTrue(editor.preview("1, 2, 3, 4, 5, 6.")) }
                    text("1, 2, 3, 4, 5, 6.")
                    acknowledge()
                    input {
                        assertTrue(editor.preview("1, 2, 3, 4, 5, 6, 7, 8, 9, 10."))
                        assertTrue(editor.commit(numbers))
                    }
                    text(numbers)
                }

                scenario.onActivity {
                    it.setWebContent(
                        if (kind == "rich") "<b>Before</b> selected <i>after</i>"
                        else "Before selected after"
                    )
                }
                text("Before selected after")
                input { assertTrue(connection.setSelection(7, 15)) }
                await { field().selection.start == 7 && field().selection.end == 15 }
                input { assertTrue(editor.preview("replacement")) }
                text("Before replacement after")
                input { assertTrue(editor.cancel()) }
                text("Before selected after")
                await { field().selection.start == 7 && field().selection.end == 15 }
                input { assertTrue(editor.preview("new words")) }
                text("Before new words after")
                input { assertTrue(editor.commit("new words")) }
                text("Before new words after")
                if (kind == "rich") {
                    val observed = AtomicReference("")
                    val done = CountDownLatch(1)
                    scenario.onActivity {
                        it.webMarkup { markup ->
                            observed.set(markup)
                            done.countDown()
                        }
                    }
                    assertTrue(done.await(5, TimeUnit.SECONDS))
                    assertTrue(observed.get(), observed.get().contains("<b>Before</b>"))
                    assertTrue(observed.get(), observed.get().contains("<i>after</i>"))
                }

                input { assertTrue(connection.setSelection(0, 0)) }
                await { field().selection.start == 0 && field().selection.end == 0 }
                input { assertTrue(editor.preview("Tentative")) }
                text("Tentative Before new words after")
                input { assertTrue(connection.setSelection(0, 0)) }
                await { field().selection.start == 0 && field().selection.end == 0 }
                input { assertFalse(editor.cancel()) }
                text("Tentative Before new words after")
            }
    }

    private fun await(
        message: () -> String = { "Browser editor did not reach the expected state" },
        ready: () -> Boolean,
    ) {
        val deadline = SystemClock.elapsedRealtime() + 8_000
        while (SystemClock.elapsedRealtime() < deadline) {
            if (ready()) return
            Thread.sleep(25)
        }
        error(message())
    }
}
