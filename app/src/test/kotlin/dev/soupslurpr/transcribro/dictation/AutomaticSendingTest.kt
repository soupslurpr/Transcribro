package dev.soupslurpr.transcribro.dictation

import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.SendTiming
import org.junit.Assert.*
import org.junit.Test

class AutomaticSendingTest {
    private val session = Preferences(autoSend = true)
    private val phrase = session.copy(sendTiming = SendTiming.Phrase)

    @Test
    fun sessionTimingWaitsForCompletionAndCannotSendTwice() {
        val sending = AutomaticSending()
        sending.committedSpeech()
        assertFalse(sending.consume(session, phraseEnd = true, canSend = true))
        sending.committedSpeech()
        assertTrue(sending.consume(session, phraseEnd = false, canSend = true))
        assertFalse(sending.consume(session, phraseEnd = false, canSend = true))
    }

    @Test
    fun successivePhrasesSendIndependentlyWithNoExtraSendAtSessionEnd() {
        val sending = AutomaticSending()
        repeat(3) {
            sending.committedSpeech()
            assertTrue(sending.consume(phrase, phraseEnd = true, canSend = true))
            assertFalse(sending.consume(phrase, phraseEnd = true, canSend = true))
        }
        assertFalse(sending.consume(phrase, phraseEnd = false, canSend = true))
    }

    @Test
    fun finishingAPhraseEarlyStillSendsItsCommittedTailOnce() {
        val sending = AutomaticSending()
        sending.committedSpeech()
        assertTrue(sending.consume(phrase, phraseEnd = false, canSend = true))
        assertFalse(sending.consume(phrase, phraseEnd = true, canSend = true))
    }

    @Test
    fun cancellationAndEmptyResultsNeverSend() {
        val sending = AutomaticSending()
        assertFalse(sending.consume(phrase, phraseEnd = true, canSend = true))
        sending.committedSpeech()
        sending.cancel()
        assertFalse(sending.consume(session, phraseEnd = false, canSend = true))
    }

    @Test
    fun disabledAndNonSendEditorsDoNotConsumePendingSpeech() {
        val sending = AutomaticSending()
        sending.committedSpeech()
        assertFalse(sending.consume(Preferences(), phraseEnd = false, canSend = true))
        assertFalse(sending.consume(session, phraseEnd = false, canSend = false))
        assertTrue(sending.consume(session, phraseEnd = false, canSend = true))
    }
}
