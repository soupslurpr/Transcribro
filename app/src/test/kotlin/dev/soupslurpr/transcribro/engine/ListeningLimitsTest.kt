package dev.soupslurpr.transcribro.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class ListeningLimitsTest {
    @Test
    fun thinkingPausesContinueAndSpeechResetsTheInactivityClock() {
        val limit = ListeningLimits(true)
        assertEquals(ListeningLimits.COMPLETE, limit.accept(16_000, true))
        assertEquals(ListeningLimits.COMPLETE, limit.accept(30 * 16_000, false))
        assertEquals(ListeningLimits.COMPLETE, limit.accept(16_000, true))
        assertEquals(ListeningLimits.COMPLETE, limit.accept(59 * 16_000, false))
        assertEquals(ListeningLimits.INACTIVITY, limit.accept(16_000, false))
    }

    @Test
    fun initialSilenceAndSuppliedAudioUseDifferentLimits() {
        assertEquals(ListeningLimits.NO_SPEECH, ListeningLimits(true).accept(12 * 16_000, false))
        val supplied = ListeningLimits(false)
        assertEquals(ListeningLimits.COMPLETE, supplied.accept(60 * 16_000, false))
        assertEquals(ListeningLimits.DURATION, supplied.accept(4 * 60 * 16_000, true))
    }
}
