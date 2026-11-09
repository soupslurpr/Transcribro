package dev.soupslurpr.transcribro.dictation

import androidx.compose.ui.text.input.TextFieldValue

/** Undo is transient and bounded by characters as well as entry count. */
internal class DraftHistory {
    private val past = ArrayDeque<TextFieldValue>()
    private val future = ArrayDeque<TextFieldValue>()

    fun canUndo(current: TextFieldValue) = past.any { it.text != current.text }

    val canRedo
        get() = future.isNotEmpty()

    fun checkpoint(value: TextFieldValue) {
        if (past.lastOrNull()?.text != value.text) past.addLast(value.copy(composition = null))
        while (past.size > 50 || past.sumOf { it.text.length } > 1_000_000) past.removeFirst()
        future.clear()
    }

    fun undo(current: TextFieldValue): TextFieldValue? {
        while (past.lastOrNull()?.text == current.text) past.removeLast()
        if (past.isEmpty()) return null
        future.addLast(current.copy(composition = null))
        return past.removeLast()
    }

    fun redo(current: TextFieldValue): TextFieldValue? {
        if (future.isEmpty()) return null
        past.addLast(current.copy(composition = null))
        return future.removeLast()
    }
}
