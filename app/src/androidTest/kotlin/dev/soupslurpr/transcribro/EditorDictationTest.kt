package dev.soupslurpr.transcribro

import android.text.Selection
import android.text.SpannableStringBuilder
import android.view.View
import android.view.inputmethod.BaseInputConnection
import android.view.inputmethod.InputConnectionWrapper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.ui.voiceinput.EditorDictation
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Real Android composing spans and replacement semantics, including a noncompliant host. */
@RunWith(AndroidJUnit4::class)
class EditorDictationTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private class Field(context: android.content.Context, text: String) :
        BaseInputConnection(View(context), true) {
        val content = SpannableStringBuilder(text).apply { Selection.setSelection(this, length) }

        override fun getEditable() = content
    }

    private fun acknowledge(editor: EditorDictation, field: Field) {
        editor.selectionChanged(
            Selection.getSelectionStart(field.content),
            Selection.getSelectionEnd(field.content),
            BaseInputConnection.getComposingSpanStart(field.content),
            BaseInputConnection.getComposingSpanEnd(field.content),
        )
    }

    @Test
    fun revisesOnlyThePreviewAndCommitsOnceBeforeTheNextPhrase() {
        val field = Field(context, "Before")
        val editor = EditorDictation(field, true) { _, _ -> }
        assertTrue(editor.preview("a note"))
        assertEquals("Before a note", field.content.toString())
        // Another update arriving before acknowledgement waits instead of appending.
        assertTrue(editor.preview("a notebook"))
        assertEquals("Before a note", field.content.toString())
        acknowledge(editor, field)
        assertEquals("Before a notebook", field.content.toString())
        assertTrue(editor.commit("a notebook."))
        assertEquals("Before a notebook.", field.content.toString())
        assertEquals(-1, BaseInputConnection.getComposingSpanStart(field.content))
        assertTrue(editor.preview("Next"))
        assertTrue(editor.cancel())
        assertEquals("Before a notebook.", field.content.toString())
    }

    @Test
    fun cancelRestoresTheOriginalSelectionAndReplacementSpacing() {
        val field = Field(context, "Before selected after")
        Selection.setSelection(field.content, 15, 7)
        val editor = EditorDictation(field, true) { _, _ -> }
        assertTrue(editor.preview("new words"))
        assertEquals("Before new words after", field.content.toString())
        assertTrue(editor.cancel())
        assertEquals("Before selected after", field.content.toString())
        // Android SurroundingText normalizes the selection direction.
        assertEquals(7, Selection.getSelectionStart(field.content))
        assertEquals(15, Selection.getSelectionEnd(field.content))
        assertTrue(editor.preview("replacement"))
        assertTrue(editor.commit("replacement"))
        assertEquals("Before replacement after", field.content.toString())
    }

    @Test
    fun movedCursorAndHostEditsAreNeverOverwritten() {
        val field = Field(context, "Before")
        val editor = EditorDictation(field, true) { _, _ -> }
        editor.preview("tentative")
        Selection.setSelection(field.content, 0)
        assertFalse(editor.cancel())
        assertEquals("Before tentative", field.content.toString())
        assertEquals(0, Selection.getSelectionStart(field.content))
        Selection.setSelection(field.content, field.content.length)
        editor.preview("new")
        field.content.replace(field.content.length - 3, field.content.length, "own")
        assertFalse(editor.commit("different"))
        assertEquals("Before tentative own", field.content.toString())
    }

    @Test
    fun editorThatDropsComposingSpansFallsBackWithoutDuplicatingWords() {
        val field = Field(context, "")
        val unsupported =
            object : InputConnectionWrapper(field, false) {
                override fun setComposingText(
                    text: CharSequence?,
                    newCursorPosition: Int,
                ): Boolean = field.commitText(text, newCursorPosition)
            }
        val editor = EditorDictation(unsupported, true) { _, _ -> }
        editor.preview("unfinished")
        acknowledge(editor, field)
        assertEquals("", field.content.toString())
        editor.preview("ignored")
        assertEquals("", field.content.toString())
        editor.commit("Final words.")
        assertEquals("Final words.", field.content.toString())
    }

    @Test
    fun completedPhraseSettingNeverInsertsTentativeWords() {
        val field = Field(context, "Existing")
        val editor = EditorDictation(field, false) { _, _ -> }
        editor.preview("temporary")
        assertEquals("Existing", field.content.toString())
        editor.commit("completed.")
        assertEquals("Existing completed.", field.content.toString())
    }

    @Test
    fun anEditorThatRefusesCompositionStillReceivesCompletedWords() {
        val field = Field(context, "Before selected after")
        Selection.setSelection(field.content, 7, 15)
        val unsupported =
            object : InputConnectionWrapper(field, false) {
                override fun setComposingText(text: CharSequence?, newCursorPosition: Int) = false
            }
        val editor = EditorDictation(unsupported, true) { _, _ -> }
        assertTrue(editor.preview("temporary"))
        assertEquals("Before selected after", field.content.toString())
        assertTrue(editor.commit("completed"))
        assertEquals("Before completed after", field.content.toString())
    }
}
