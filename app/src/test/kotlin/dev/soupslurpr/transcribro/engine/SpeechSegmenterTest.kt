package dev.soupslurpr.transcribro.engine

import org.junit.Assert.*
import org.junit.Test

class SpeechSegmenterTest {
    @Test
    fun pausesEndOnceAfterSpeechAndHysteresisKeepsQuietSpeech() {
        val detector = SpeechSegmenter()
        detector.accept(0.59f)
        assertFalse(detector.speaking)
        detector.accept(0.61f)
        repeat(30) {
            detector.accept(0.4f)
            assertTrue(detector.speaking)
        }
        repeat(29) {
            detector.accept(0.1f)
            assertFalse(detector.completedPhrase)
        }
        detector.accept(0.1f)
        assertTrue(detector.completedPhrase)
        assertFalse(detector.speaking)
        repeat(100) {
            detector.accept(0f)
            assertFalse(detector.completedPhrase)
        }
    }

    @Test
    fun aShortNoiseSpikeIsNotASendBoundary() {
        val detector = SpeechSegmenter()
        repeat(3) { detector.accept(1f) }
        repeat(100) {
            detector.accept(0f)
            assertFalse(detector.completedPhrase)
        }
    }

    @Test
    fun longSpeechAndThinkingPausesDoNotCreateInternalSendBoundaries() {
        val detector = SpeechSegmenter()
        repeat(10_000) {
            detector.accept(0.8f)
            assertFalse(detector.completedPhrase)
        }
        repeat(20) {
            detector.accept(0.1f)
            assertFalse(detector.completedPhrase)
        }
        repeat(4) {
            detector.accept(0.8f)
            assertFalse(detector.completedPhrase)
        }
        repeat(29) {
            detector.accept(0.1f)
            assertFalse(detector.completedPhrase)
        }
        detector.accept(0f)
        assertTrue(detector.completedPhrase)
        repeat(4) {
            detector.accept(0.8f)
            assertFalse(detector.completedPhrase)
        }
        repeat(30) { detector.accept(0f) }
        assertTrue(detector.completedPhrase)
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidProbabilitiesFailExplicitly() {
        SpeechSegmenter().accept(Float.NaN)
    }
}
