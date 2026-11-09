package dev.soupslurpr.transcribro.engine

import org.junit.Assert.*
import org.junit.Test

class TranscriptTest {
    @Test
    fun joiningDoesNotInventCommasOrRewriteWords() {
        assertEquals("Hello. This is fine.", Transcript.join("Hello.", " This is fine. "))
        assertEquals("a\nb", Transcript.join("a\n", "b"))
        assertEquals("Hello.", Transcript.join("Hello.", "  "))
    }

    @Test
    fun insertionHandlesEmptyAndOneCharacterFields() {
        assertEquals("Hello", Transcript.insertion(null, "Hello"))
        assertEquals("Hello", Transcript.insertion("", "Hello"))
        assertEquals(" Hello", Transcript.insertion("A", "Hello"))
        assertEquals("Hello", Transcript.insertion("\n", "Hello"))
    }

    @Test
    fun unicodeIsNotRewritten() {
        assertEquals("café 日本語", Transcript.join("café", "日本語"))
        assertEquals(" Hello", Transcript.insertion("🙂", "Hello"))
    }

    @Test
    fun insertionSeparatesExistingWordsWithoutAddingSpacesInsidePunctuation() {
        assertEquals(" hello ", Transcript.insertion("one", "hello", "two"))
        assertEquals("hello", Transcript.insertion("(", "hello", ")"))
        assertEquals(", yes", Transcript.insertion("one", ", yes", "."))
        assertEquals("hello", Transcript.insertion(" ", "hello", " "))
    }

    @Test
    fun selectedWordsAndPhrasesFitTheExistingSentence() {
        assertEquals(
            "put the blue notebook here",
            replace("put the ", "red", " notebook here", "Blue."),
        )
        assertEquals(
            "put the bright blue folder here",
            replace("put the ", "old notebook", " here", "Bright blue folder."),
        )
    }

    @Test
    fun selectedWordsKeepSentenceStartAndEndBoundaries() {
        assertEquals(
            "Blue notebooks are here.",
            replace("", "Red", " notebooks are here.", "blue."),
        )
        assertEquals(
            "put the notebook there.",
            replace("put the notebook ", "here", ".", "There."),
        )
        assertEquals(
            "put the notebook there.",
            replace("put the notebook ", "here.", "", "There."),
        )
    }

    @Test
    fun punctuationIncludedInTheSelectionIsPreserved() {
        assertEquals(
            "Use “blue”, then continue.",
            replace("Use ", "“red”,", " then continue.", "Blue."),
        )
        assertEquals("Is it blue?", replace("Is it ", "red?", "", "Blue."))
    }

    @Test
    fun replacingTheEntireFieldKeepsTheNewSentencesOwnPunctuation() {
        assertEquals("Are we ready?", replace("", "The old answer.", "", "Are we ready?"))
        assertEquals("New sentence.", replace("", "old phrase", "", "New sentence."))
        assertEquals("Blue.", replace("", "red", "", "Blue."))
    }

    @Test
    fun replacementsKeepAcronymsInternalCasingAndMeaningfulSymbols() {
        assertEquals("use NASA here", replace("use ", "this", " here", "NASA."))
        assertEquals("use iPhone here", replace("use ", "this", " here", "iPhone."))
        assertEquals("pay $20 today", replace("pay ", "this", " today", "$20."))
        assertEquals("set -20 here", replace("set ", "this", " here", "-20."))
        assertEquals("set 20% here", replace("set ", "this", " here", "20%."))
        assertEquals("we don't agree", replace("we ", "always", " agree", "Don't."))
    }

    @Test
    fun ordinaryDictationAndUnknownContextKeepRecognitionUnchanged() {
        assertEquals("Hello?", Transcript.insertion("one ", "Hello?", " after"))
        assertEquals("Hello?", Transcript.insertion("one ", "Hello?", " after", replacedText = ""))
        assertEquals("Hello?", Transcript.insertion(null, "Hello?", null, replacedText = "old"))
    }

    @Test
    fun replacingOnlyPunctuationDoesNotRestoreThePunctuationBeingReplaced() {
        assertEquals("Hello brave world", replace("Hello", ",", " world", "brave"))
    }

    private fun replace(before: String, selected: String, after: String, spoken: String): String =
        before + Transcript.insertion(before, spoken, after, replacedText = selected) + after

    @Test(expected = IllegalArgumentException::class)
    fun transcriptLimitIncludesJoiningSpace() {
        Transcript.join("a".repeat(Transcript.MAX_LENGTH), "b")
    }
}
