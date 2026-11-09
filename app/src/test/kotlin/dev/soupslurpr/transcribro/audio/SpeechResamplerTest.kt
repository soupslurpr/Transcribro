package dev.soupslurpr.transcribro.audio

import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.*
import org.junit.Test

class SpeechResamplerTest {
    @Test
    fun preservesDurationAcrossRatesAndIrregularChunks() {
        for (rate in listOf(8000, 16000, 22050, 44100, 48000, 96000, 192000)) {
            val resampler = SpeechResampler(rate)
            var count = 0
            var offset = 0
            while (offset < rate) {
                val size = minOf(137, rate - offset)
                count += resampler.process(FloatArray(size) { 0.25f }).size
                offset += size
            }
            count += resampler.finish().size
            assertEquals("rate=$rate", 16_000, count)
        }
    }

    @Test
    fun chunkBoundariesDoNotChangeAudio() {
        val input = FloatArray(44100) { (sin(2 * PI * 440 * it / 44100) * 0.6).toFloat() }
        val whole = SpeechResampler(44100).let { it.process(input) + it.finish() }
        val split = SpeechResampler(44100)
        var result = shortArrayOf()
        for (offset in input.indices step 391) {
            result += split.process(input.copyOfRange(offset, minOf(offset + 391, input.size)))
        }
        result += split.finish()
        assertArrayEquals(whole, result)
    }

    @Test
    fun rejectsFrequenciesAboveTheSpeechNyquistLimit() {
        fun energy(frequency: Int): Double {
            val input = FloatArray(48000) { (sin(2 * PI * frequency * it / 48000) * 0.5).toFloat() }
            val converter = SpeechResampler(48000)
            val output = converter.process(input) + converter.finish()
            return sqrt(
                output
                    .sliceArray(100 until output.size - 100)
                    .map {
                        it.toDouble() * it
                    }
                    .average()
            )
        }
        val voice = energy(1000)
        val alias = energy(12000)
        assertTrue("The voice band must be retained: $voice", voice > 10000)
        assertTrue("Aliasing must be suppressed: $alias", alias < voice * 0.01)
    }

    @Test
    fun nativeRateClipsSafelyAndRejectsInvalidSamples() {
        val converter = SpeechResampler(16000)
        assertArrayEquals(
            shortArrayOf(-32768, -16384, 0, 16384, 32767, 0),
            converter.process(floatArrayOf(-2f, -0.5f, 0f, 0.5f, 2f, Float.NaN)),
        )
        assertEquals(0, converter.finish().size)
    }
}
