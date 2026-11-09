package dev.soupslurpr.transcribro.engine

/** Joins recognition results and fits deliberate replacements into their existing text. */
object Transcript {
    const val MAX_LENGTH = 64_000

    fun join(previous: String, next: String): String {
        val clean = next.trim()
        if (clean.isEmpty()) return previous
        val separator = if (previous.isNotEmpty() && !previous.last().isWhitespace()) " " else ""
        require(previous.length + separator.length + clean.length <= MAX_LENGTH)
        return previous + separator + clean
    }

    fun insertion(
        before: CharSequence?,
        text: String,
        after: CharSequence? = null,
        replacedText: CharSequence? = null,
    ): String {
        val clean = replacement(before, text.trim(), after, replacedText)
        if (clean.isEmpty()) return ""
        val leading =
            before?.lastOrNull()?.let {
                !it.isWhitespace() && it !in "([{\"“/" && clean.first() !in ",.;:!?)]}"
            } == true
        val trailing = after?.firstOrNull()?.isLetterOrDigit() == true && clean.last() !in "([{\"“/"
        return (if (leading) " " else "") + clean + if (trailing) " " else ""
    }

    private fun replacement(
        before: CharSequence?,
        text: String,
        after: CharSequence?,
        replacedText: CharSequence?,
    ): String {
        val original = replacedText?.toString()?.trim().orEmpty()
        // A whole-field replacement is a new utterance with its own sentence boundaries.
        // Unknown surrounding text is not enough evidence to change those boundaries.
        if (
            before == null ||
                after == null ||
                (before.isEmpty() && after.isEmpty()) ||
                original.none(Char::isLetterOrDigit)
        )
            return text
        var words = text.trim { it.isWhitespace() || it in EDGE_PUNCTUATION }
        if (words.none(Char::isLetterOrDigit)) return text

        // Match only the initial letter. Recognition retains acronym and internal casing;
        // deciding whether a newly dictated title-cased word is a name needs human judgment.
        val first = words.indexOfFirst(Char::isLetter)
        val originalFirst = original.firstOrNull(Char::isLetter)
        if (first >= 0 && originalFirst != null) {
            val word = words.substring(first).takeWhile(Char::isLetter)
            val hasDistinctCase = word.all(Char::isUpperCase) || word.drop(1).any(Char::isUpperCase)
            if (!hasDistinctCase) {
                val letter =
                    if (originalFirst.isUpperCase()) words[first].uppercaseChar()
                    else words[first].lowercaseChar()
                words = words.replaceRange(first, first + 1, letter.toString())
            }
        }
        return original.takeWhile { it in EDGE_PUNCTUATION } +
            words +
            original.takeLastWhile { it in EDGE_PUNCTUATION }
    }

    private const val EDGE_PUNCTUATION = ".,;:!?()[]{}\"'“”‘’"
}
