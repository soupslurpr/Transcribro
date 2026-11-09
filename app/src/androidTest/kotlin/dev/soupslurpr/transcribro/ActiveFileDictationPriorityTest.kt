package dev.soupslurpr.transcribro

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.media.AudioFormat
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.audio.AudioFileTranscriber
import dev.soupslurpr.transcribro.engine.DictationPriority
import dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Both isolated workers must own loaded models; pausing before file startup cannot test this. */
@RunWith(AndroidJUnit4::class)
class ActiveFileDictationPriorityTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun activeFilePausesForRealDictationAndResumesWithoutLosingOrRepeatingText() = runBlocking {
        instrumentation.uiAutomation.grantRuntimePermission(
            context.packageName,
            Manifest.permission.RECORD_AUDIO,
        )
        val file = repeatedFileFixture()
        try {
            ActivityScenario.launch<MainActivity>(
                    Intent(context, MainActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                )
                .use {
                    assertFalse(
                        "Another dictation session is active",
                        DictationPriority.active.value,
                    )
                    // An uninterrupted run supplies an exact phrase-by-phrase control on this CPU.
                    val baseline = ConcurrentLinkedQueue<String>()
                    withTimeout(120_000) { transcribe(file, { _, _ -> }, baseline::add) }
                    assertTrue(
                        "The control must contain several completed phrases",
                        baseline.size >= 3,
                    )

                    val phrases = ConcurrentLinkedQueue<String>()
                    val progress = AtomicLong()
                    val duration = AtomicLong(-1)
                    val started = AtomicBoolean()
                    val atStart = AtomicReference<Checkpoint>()
                    val dictation = SuppliedDictation()
                    try {
                        val processing =
                            async(Dispatchers.IO) {
                                withTimeout(120_000) {
                                    transcribe(
                                        file,
                                        { elapsed, total ->
                                            progress.set(elapsed)
                                            duration.set(total)
                                        },
                                        { phrase ->
                                            phrases.add(phrase)
                                            // This callback runs on main after file
                                            // inference has committed actual speech.
                                            if (started.compareAndSet(false, true)) {
                                                atStart.set(
                                                    Checkpoint(progress.get(), phrases.toList())
                                                )
                                                dictation.start()
                                            }
                                        },
                                    )
                                }
                            }
                        await("The second worker never became ready") {
                            dictation.ready.count == 0L || dictation.terminal.count == 0L
                        }
                        assertEquals("Dictation failed during startup", 0, dictation.error.get())
                        assertEquals(0L, dictation.ready.count)
                        assertTrue("File had not started reading audio", atStart.get().progress > 0)
                        assertTrue(
                            "File finished before the overlap was exercised",
                            processing.isActive,
                        )
                        assertTrue(
                            "The real broker did not acquire priority",
                            DictationPriority.active.value,
                        )

                        // One already-running native call may complete after priority changes.
                        // Require progress and completed text to settle, then stay fixed while
                        // the second loaded worker actually transcribes supplied speech.
                        val paused = awaitStable { Checkpoint(progress.get(), phrases.toList()) }
                        assertTrue("File finished instead of pausing", processing.isActive)
                        assertTrue(
                            "Dictation ended before audio was supplied",
                            DictationPriority.active.value,
                        )
                        assertTrue(paused.progress < duration.get())
                        assertEquals(
                            atStart.get().phrases,
                            paused.phrases.take(atStart.get().phrases.size),
                        )
                        dictation.allowAudio.countDown()
                        await("Dictation did not complete", timeoutMillis = 90_000) {
                            onMain {
                                if (DictationPriority.active.value) {
                                    assertEquals(
                                        "Paused file consumed more audio",
                                        paused.progress,
                                        progress.get(),
                                    )
                                    assertEquals(
                                        "Paused file changed completed text",
                                        paused.phrases,
                                        phrases.toList(),
                                    )
                                }
                            }
                            dictation.terminal.count == 0L
                        }
                        assertEquals("Dictation returned an error", 0, dictation.error.get())
                        assertEquals((1..10).toList(), numbers(dictation.text.get()))
                        withTimeout(120_000) { processing.await() }
                        assertFalse(
                            "Priority lease survived recognition",
                            DictationPriority.active.value,
                        )
                        assertTrue(
                            "File did not advance after dictation",
                            progress.get() > paused.progress,
                        )
                        assertEquals(duration.get(), progress.get())
                        assertEquals(
                            "Paused/resumed file lost, duplicated, or changed phrases",
                            baseline.toList(),
                            phrases.toList(),
                        )
                    } finally {
                        onMain { dictation.close() }
                        dictation.joinWriter()
                    }
                }
        } finally {
            assertTrue("Could not remove the generated audio fixture", file.delete())
        }
    }

    private data class Checkpoint(val progress: Long, val phrases: List<String>)

    private suspend fun awaitStable(snapshot: () -> Checkpoint): Checkpoint {
        var previous = snapshot()
        var unchangedSince = SystemClock.elapsedRealtime()
        val deadline = unchangedSince + 15_000
        while (SystemClock.elapsedRealtime() < deadline) {
            delay(50)
            val current = snapshot()
            if (current != previous) {
                previous = current
                unchangedSince = SystemClock.elapsedRealtime()
            } else if (SystemClock.elapsedRealtime() - unchangedSince >= 1500) {
                return current
            }
        }
        throw AssertionError("The active file did not pause after its bounded native call")
    }

    private suspend fun await(
        message: String,
        timeoutMillis: Long = 45_000,
        condition: () -> Boolean,
    ) {
        val deadline = SystemClock.elapsedRealtime() + timeoutMillis
        while (SystemClock.elapsedRealtime() < deadline) {
            if (condition()) return
            delay(50)
        }
        throw AssertionError(message)
    }

    private suspend fun transcribe(
        file: File,
        progress: (Long, Long) -> Unit,
        phrase: (String) -> Unit,
    ) {
        AssetFileDescriptor(
                ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY),
                0,
                file.length(),
            )
            .use { AudioFileTranscriber(context).transcribe(it, progress, phrase) }
    }

    private fun repeatedFileFixture(): File {
        val pcm =
            instrumentation.context.assets.open("dictation-fixture.pcm").use { it.readBytes() }
        // Three copies leave substantial file work after the first completed phrase. The
        // existing synthetic recording and generated silence are never microphone recordings.
        val silence = ByteArray(4 * 16_000 * 2)
        val bytes = 3 * (pcm.size + silence.size)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(36 + bytes)
        header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)).putInt(16)
        header.putShort(1).putShort(1).putInt(16_000).putInt(32_000).putShort(2).putShort(16)
        header.put("data".toByteArray(Charsets.US_ASCII)).putInt(bytes)
        return File.createTempFile("active-file-priority-", ".wav", context.cacheDir).also { file ->
            try {
                file.outputStream().use { output ->
                    output.write(header.array())
                    repeat(3) {
                        output.write(pcm)
                        output.write(silence)
                    }
                }
            } catch (failure: Exception) {
                file.delete()
                throw failure
            } finally {
                pcm.fill(0)
            }
        }
    }

    private inner class SuppliedDictation : AutoCloseable {
        val ready = CountDownLatch(1)
        val terminal = CountDownLatch(1)
        val allowAudio = CountDownLatch(1)
        val error = AtomicInteger()
        val text = AtomicReference("")
        private val pipe = ParcelFileDescriptor.createPipe()
        private var recognizer: SpeechRecognizer? = null
        private val writer = Thread {
            val pcm = instrumentation.context.assets.open("count-to-ten.pcm").use { it.readBytes() }
            try {
                check(allowAudio.await(45, TimeUnit.SECONDS))
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                    for (offset in pcm.indices step 1024) {
                        output.write(pcm, offset, minOf(1024, pcm.size - offset))
                        Thread.sleep(32)
                    }
                }
            } catch (_: Exception) {
                error.compareAndSet(0, -1)
                terminal.countDown()
            } finally {
                pcm.fill(0)
            }
        }

        fun start() {
            val speech =
                SpeechRecognizer.createSpeechRecognizer(
                    context,
                    ComponentName(context, MainRecognitionService::class.java),
                )
            recognizer = speech
            speech.setRecognitionListener(
                object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        pipe[0].close()
                        writer.start()
                        ready.countDown()
                    }

                    override fun onBeginningOfSpeech() = Unit

                    override fun onRmsChanged(rmsdB: Float) = Unit

                    override fun onBufferReceived(buffer: ByteArray?) = Unit

                    override fun onEndOfSpeech() = Unit

                    override fun onPartialResults(partialResults: Bundle?) = Unit

                    override fun onEvent(eventType: Int, params: Bundle?) = Unit

                    override fun onError(reason: Int) {
                        error.set(reason)
                        terminal.countDown()
                    }

                    override fun onResults(results: Bundle?) {
                        text.set(
                            results
                                ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                ?.firstOrNull()
                                .orEmpty()
                        )
                        terminal.countDown()
                    }
                }
            )
            speech.startListening(
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                    .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                    .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    .putExtra(MainRecognitionService.EXTRA_CONTINUOUS, true)
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe[0])
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16_000)
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                    .putExtra(
                        RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                        AudioFormat.ENCODING_PCM_16BIT,
                    )
            )
        }

        override fun close() {
            recognizer?.cancel()
            recognizer?.destroy()
            recognizer = null
            pipe.forEach { runCatching { it.close() } }
            writer.interrupt()
        }

        fun joinWriter() = writer.join(2_000)
    }

    private fun numbers(text: String): List<Int> {
        val names =
            listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        return Regex("[a-z]+|[0-9]+")
            .findAll(text.lowercase())
            .map { it.value.toIntOrNull() ?: (names.indexOf(it.value) + 1) }
            .toList()
    }

    private fun onMain(action: () -> Unit) = instrumentation.runOnMainSync(action)
}
