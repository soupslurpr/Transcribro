package dev.soupslurpr.transcribro.dictation

import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import org.junit.Assert.*
import org.junit.Test

class DraftHistoryTest {
    @Test
    fun restoresTextAndSelectionWithoutRestoringAnImeComposition() {
        val history = DraftHistory()
        history.checkpoint(TextFieldValue("before", TextRange(2), TextRange(1, 3)))
        val restored = history.undo(TextFieldValue("after"))!!
        assertEquals("before", restored.text)
        assertEquals(TextRange(2), restored.selection)
        assertNull(restored.composition)
        assertEquals("after", history.redo(restored)!!.text)
    }

    @Test
    fun newEditAfterUndoInvalidatesRedo() {
        val history = DraftHistory()
        history.checkpoint(TextFieldValue("one"))
        val restored = history.undo(TextFieldValue("two"))!!
        assertTrue(history.canRedo)
        history.checkpoint(restored)
        assertFalse(history.canRedo)
        assertNull(history.redo(TextFieldValue("three")))
    }

    @Test
    fun limitsHistoryMemoryForLongDrafts() {
        val history = DraftHistory()
        repeat(100) { history.checkpoint(TextFieldValue("$it".padEnd(64_000, 'x'))) }
        var current = TextFieldValue("latest")
        var steps = 0
        while (true) {
            current = history.undo(current) ?: break
            steps++
        }
        assertEquals(15, steps)
    }

    @Test
    fun unchangedCheckpointsDoNotCreateAnUndoLoop() {
        val history = DraftHistory()
        val value = TextFieldValue("same")
        repeat(4) { history.checkpoint(value) }
        assertNull(history.undo(value))
        assertFalse(history.canUndo(value))
    }
}
