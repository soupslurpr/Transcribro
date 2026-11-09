package dev.soupslurpr.transcribro.ui.voiceinput

import android.view.inputmethod.InputConnection
import dev.soupslurpr.transcribro.engine.Transcript

/** Owns only the unfinished phrase in one input connection, never a field-sized snapshot. */
internal class EditorDictation(
    private val input: InputConnection,
    private var liveText: Boolean,
    private val selection: () -> Pair<Int, Int>? = { null },
    private val expectSelection: (Int, Int) -> Unit,
) {
    private data class Composition(
        val start: Int,
        val original: String,
        val originalStart: Int,
        val originalEnd: Int,
        val before: String,
        val after: String,
        var rendered: String,
        var acknowledged: Boolean = false,
    ) {
        val end: Int
            get() = start + rendered.length
    }

    private var composition: Composition? = null
    private var pending: String? = null

    fun preview(text: String): Boolean {
        if (text.isEmpty()) return cancel()
        if (!liveText || text.length > MAX_PREVIEW) return true
        var current = composition
        val firstPreview = current == null
        if (current == null) {
            val surrounding = input.getSurroundingText(1, 1, 0)
            if (surrounding == null) {
                liveText = false
                return true
            }
            val start = minOf(surrounding.selectionStart, surrounding.selectionEnd)
            val end = maxOf(surrounding.selectionStart, surrounding.selectionEnd)
            val reported = selection()?.takeIf { it.first >= 0 && it.second >= 0 }
            val offset =
                if (surrounding.offset >= 0) surrounding.offset
                else reported?.let { minOf(it.first, it.second) - start } ?: -1
            if (offset < 0) {
                liveText = false
                return true
            }
            if (start !in 0..end || end > surrounding.text.length || end - start > MAX_PREVIEW) {
                liveText = false
                return true
            }
            // Finish another keyboard's composition before taking ownership of our phrase.
            input.finishComposingText()
            current =
                Composition(
                    start = offset + start,
                    original = surrounding.text.subSequence(start, end).toString(),
                    originalStart = reported?.first ?: (offset + surrounding.selectionStart),
                    originalEnd = reported?.second ?: (offset + surrounding.selectionEnd),
                    before = surrounding.text.subSequence(0, start).toString(),
                    after = surrounding.text.subSequence(end, surrounding.text.length).toString(),
                    rendered = "",
                )
            composition = current
        } else {
            if (!owns(current)) return abandon()
            // InputConnection's boolean means queued, not that the host retained composition.
            // Wait for the editor's composing-range acknowledgement before revising in place.
            if (!current.acknowledged) {
                pending = text
                return true
            }
        }
        val insertion = Transcript.insertion(current.before, text, current.after, current.original)
        if (current.rendered == insertion) return true
        current.rendered = insertion
        current.acknowledged = false
        pending = null
        expectSelection(current.end, current.end)
        if (!input.setComposingText(insertion, 1)) {
            liveText = false
            if (firstPreview) {
                val unchanged = input.getSurroundingText(1, 1, 0)
                if (
                    unchanged != null &&
                        unchanged.text.toString() ==
                            current.before + current.original + current.after &&
                        minOf(unchanged.selectionStart, unchanged.selectionEnd) ==
                            current.before.length &&
                        maxOf(unchanged.selectionStart, unchanged.selectionEnd) ==
                            current.before.length + current.original.length &&
                        (unchanged.offset < 0 ||
                            unchanged.offset + current.before.length == current.start)
                ) {
                    composition = null
                    expectSelection(current.originalStart, current.originalEnd)
                    return true
                }
            }
            return cancel()
        }
        return true
    }

    fun selectionChanged(start: Int, end: Int, composingStart: Int, composingEnd: Int): Boolean {
        val current = composition ?: return true
        if (start != current.end || end != current.end) return true
        if (composingStart == current.start && composingEnd == current.end) {
            current.acknowledged = true
            return pending?.let { preview(it) } ?: true
        } else {
            // An editor that strips composing spans receives completed phrases instead.
            liveText = false
            return cancel()
        }
    }

    fun commit(text: String): Boolean {
        val current = composition
        pending = null
        if (current != null) {
            if (!owns(current)) return abandon()
            val insertion =
                Transcript.insertion(current.before, text, current.after, current.original)
            composition = null
            expectSelection(current.start + insertion.length, current.start + insertion.length)
            // This also handles final text arriving before the composing acknowledgement.
            return input.replaceText(current.start, current.end, insertion, 1, null)
        }
        val surrounding = input.getSurroundingText(1, 1, 0)
        val insertion =
            Transcript.insertion(
                input.getTextBeforeCursor(1, 0),
                text,
                input.getTextAfterCursor(1, 0),
                input.getSelectedText(0),
            )
        val cursor =
            if (surrounding != null && surrounding.offset >= 0)
                surrounding.offset + minOf(surrounding.selectionStart, surrounding.selectionEnd)
            else selection()?.let { minOf(it.first, it.second) }
        if (cursor != null) {
            expectSelection(cursor + insertion.length, cursor + insertion.length)
        }
        return input.commitText(insertion, 1)
    }

    /** Roll back only an unchanged, still-owned preview, including any replaced selection. */
    fun cancel(): Boolean {
        pending = null
        val current = composition ?: return true
        if (!owns(current)) return abandon()
        composition = null
        input.beginBatchEdit()
        try {
            expectSelection(
                current.start + current.original.length,
                current.start + current.original.length,
            )
            val replaced = input.replaceText(current.start, current.end, current.original, 1, null)
            if (!replaced) return false
            expectSelection(current.originalStart, current.originalEnd)
            return input.setSelection(current.originalStart, current.originalEnd)
        } finally {
            input.endBatchEdit()
        }
    }

    /** The user/host owns changed text. Never move its cursor or overwrite its edits. */
    fun abandon(): Boolean {
        composition = null
        pending = null
        input.finishComposingText()
        return false
    }

    private fun owns(current: Composition): Boolean {
        val surrounding = input.getSurroundingText(current.rendered.length, 0, 0) ?: return false
        val cursor =
            if (surrounding.offset >= 0) surrounding.offset + surrounding.selectionStart
            else selection()?.takeIf { it.first == it.second }?.first
        return surrounding.selectionStart == surrounding.selectionEnd &&
            cursor == current.end &&
            surrounding.text.take(surrounding.selectionStart).toString().endsWith(current.rendered)
    }

    private companion object {
        const val MAX_PREVIEW = 16_000
    }
}
