package dev.soupslurpr.transcribro

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.speech.NativeEngine
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Stop must decode retained audio, even when the preceding update consumed the last chunk. */
@RunWith(AndroidJUnit4::class)
class NativeFinalizationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Test fun finishAfterAnExactChunkIncludesTenWithLiveText() = count(previews = true)

    @Test fun finishAfterAnExactChunkIncludesTenWithoutLiveText() = count(previews = false)

    @Test
    fun slowerCountingKeepsShortGapsInsideACompleteRecognitionPhrase() {
        for (previews in listOf(true, false)) count(
            previews,
            "count-to-ten-slower.pcm",
            listOf(0, 1, 257, 511, 777, 1600),
        )
    }

    @Test
    fun fastNumbersAreNotCutOffByTheirFormattingTokenCount() {
        for (previews in listOf(true, false)) {
            count(previews, "count-to-ten-faster.pcm", listOf(0, 257, 511))
        }
    }

    @Test
    fun deferredPreviewsPreserveSpeechAcrossSuccessiveFinalBoundaries() =
        count(true, "count-to-ten-slower.pcm", listOf(0, 511), deferAnalysis = true)

    @Test
    fun backlogUpdatesKeepTheExistingPreviewWhileAdvancingAudio() {
        val pcm =
            instrumentation.context.assets.open("count-to-ten-slower.pcm").use { it.readBytes() }
        val audio =
            FloatArray(pcm.size / 2) { index ->
                ((pcm[index * 2].toInt() and 255) or (pcm[index * 2 + 1].toInt() shl 8)).toShort() /
                    32768f
            }
        pcm.fill(0)
        NativeEngine(instrumentation.targetContext.assets, previews = true).use { engine ->
            engine.stream(audio.copyOfRange(0, 8192), false)
            val preview = engine.stream(audio.copyOfRange(8192, 16384), false).last()
            assertTrue("The initial live preview must exist", preview.isNotBlank())
            // The queue can empty before the combined analysis. Its first deferred
            // chunk must still suppress that analysis's obsolete tentative decoding.
            assertTrue(engine.stream(audio.copyOfRange(16384, 24576), false, true).isEmpty())
            val caughtUp = engine.stream(audio.copyOfRange(24576, 32768), false, false)
            assertEquals("Catching up must preserve the visible preview", listOf(preview), caughtUp)
            val completed = mutableListOf<String>()
            for (offset in 32768 until audio.size step 8192) {
                completed +=
                    engine
                        .stream(
                            audio.copyOfRange(offset, minOf(offset + 8192, audio.size)),
                            false,
                        )
                        .dropLast(1)
            }
            completed += engine.stream(FloatArray(0), true).dropLast(1)
            assertCount(completed.joinToString(" "), "after a suppressed preview")
        }
        audio.fill(0f)
    }

    private fun count(
        previews: Boolean,
        fixture: String = "count-to-ten.pcm",
        prefixes: List<Int> = listOf(0, 1, 257, 511, 512, 1023, 1279),
        deferAnalysis: Boolean = false,
    ) {
        val pcm = instrumentation.context.assets.open(fixture).use { it.readBytes() }
        val audio =
            FloatArray(pcm.size / 2) { index ->
                ((pcm[index * 2].toInt() and 255) or (pcm[index * 2 + 1].toInt() shl 8)).toShort() /
                    32768f
            }
        pcm.fill(0)
        NativeEngine(instrumentation.targetContext.assets, previews).use { engine ->
            // Exercise empty Stop and several alignments within the 1,280-sample frontend block.
            for (prefix in prefixes) {
                val shifted = FloatArray(prefix) + audio
                val completed = mutableListOf<String>()
                for (offset in shifted.indices step 8192) {
                    completed +=
                        engine
                            .stream(
                                shifted.copyOfRange(offset, minOf(offset + 8192, shifted.size)),
                                finishPhrase = false,
                                deferAnalysis = deferAnalysis,
                            )
                            .dropLast(1)
                }
                if (deferAnalysis) {
                    assertTrue(
                        "Backlog must not suppress completed native lines before Finish",
                        completed.any(String::isNotBlank),
                    )
                }
                completed += engine.stream(FloatArray(0), finishPhrase = true).dropLast(1)
                assertCount(completed.joinToString(" "), "previews=$previews, prefix=$prefix")
                assertTrue(
                    "Finishing the reset stream duplicated text",
                    engine.stream(FloatArray(0), finishPhrase = true).all(String::isBlank),
                )
                shifted.fill(0f)
            }
        }
        audio.fill(0f)
    }

    @Test
    fun successiveCapturedCountsDoNotCompleteEarlyAtAShortGap() {
        // Actual AudioRecord PCM from the emulator's injected synthetic voice. The second
        // count used to split after six, complete as 1–10, then append the real 7–10.
        val pcm =
            instrumentation.context.assets.open("count-to-ten-successive-microphone.pcm").use {
                it.readBytes()
            }
        val audio =
            FloatArray(pcm.size / 2) { index ->
                ((pcm[index * 2].toInt() and 255) or (pcm[index * 2 + 1].toInt() shl 8)).toShort() /
                    32768f
            }
        pcm.fill(0)
        for (previews in listOf(true, false)) {
            NativeEngine(instrumentation.targetContext.assets, previews).use { engine ->
                var offset = 0
                // Preserve the captured 512-sample framing and actual outer pause boundaries.
                for ((index, boundary) in listOf(152064, 307712, 313344).withIndex()) {
                    val completed = mutableListOf<String>()
                    while (offset < boundary) {
                        val end = minOf(offset + 8192, boundary)
                        completed +=
                            engine
                                .stream(audio.copyOfRange(offset, end), end == boundary)
                                .dropLast(1)
                        offset = end
                    }
                    if (index < 2)
                        assertCount(
                            completed.joinToString(" "),
                            "captured count ${index + 1}, previews=$previews",
                        )
                    else
                        assertTrue(
                            "The remaining silence must not repeat text",
                            completed.all(String::isBlank),
                        )
                }
            }
        }
        audio.fill(0f)
    }

    @Test
    fun silenceAndRepeatedFinishesDoNotInventWords() {
        NativeEngine(instrumentation.targetContext.assets).use { engine ->
            repeat(3) {
                assertTrue(engine.stream(FloatArray(8192), false).all(String::isBlank))
            }
            repeat(2) {
                assertTrue(engine.stream(FloatArray(0), true).all(String::isBlank))
            }
        }
    }

    private fun assertCount(text: String, case: String) {
        val names =
            listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        val numbers =
            Regex("[a-z]+|[0-9]+")
                .findAll(text.lowercase())
                .map {
                    it.value.toIntOrNull() ?: (names.indexOf(it.value) + 1)
                }
                .toList()
        assertEquals("$case: $text", (1..10).toList(), numbers)
    }
}
