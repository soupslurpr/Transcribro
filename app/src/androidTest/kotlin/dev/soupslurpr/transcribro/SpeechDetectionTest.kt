package dev.soupslurpr.transcribro

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.ParcelFileDescriptor
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.engine.ITranscriptionEvents
import dev.soupslurpr.transcribro.engine.ITranscriptionWorker
import dev.soupslurpr.transcribro.engine.ListeningLimits
import dev.soupslurpr.transcribro.engine.TranscriptionWorker
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Real isolated worker, supplied PCM, and live timeout rules; no physical audio. */
@RunWith(AndroidJUnit4::class)
class SpeechDetectionTest {
    @Test
    fun noSpeechTimeoutCompletesRecognitionInsteadOfCancellingIt() {
        val result = recognize(ByteArray(12 * 16_000 * 2))
        assertEquals(ListeningLimits.NO_SPEECH, result.endReason.get())
        assertEquals(0, result.speechStarts.get())
        assertEquals(0, result.phraseEnds.get())
        assertTrue(result.words.toString(), result.words.isEmpty())
    }

    @Test
    fun manualFinishStillCompletesWhenQuietAudioNeverTriggersTheDetector() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val original =
            instrumentation.context.assets.open("count-to-ten-slower.pcm").use { it.readBytes() }
        val quiet = ByteArray(original.size)
        for (offset in original.indices step 2) {
            val sample =
                ((original[offset].toInt() and 255) or (original[offset + 1].toInt() shl 8))
                    .toShort()
            val scaled = kotlin.math.round(sample * 0.003).toInt()
            quiet[offset] = scaled.toByte()
            quiet[offset + 1] = (scaled shr 8).toByte()
        }
        original.fill(0)
        val result = recognize(quiet)
        assertEquals(ListeningLimits.COMPLETE, result.endReason.get())
        assertEquals(0, result.speechStarts.get())
        // The speech detector must not prevent the recognizer's final attempt.
        // Any recovered words are welcome; an empty result is also legitimate.
        assertEquals(0, result.phraseEnds.get())
    }

    @Test
    fun noSpeechTimeoutDrainsAlreadyCapturedWordsWithoutSending() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val count = instrumentation.context.assets.open("count-to-ten.pcm").use { it.readBytes() }
        // Simulate a capture buffer ahead of processing: twelve silent seconds
        // reach the detector while the following spoken count is already queued.
        val captured = ByteArray(12 * 16_000 * 2) + count
        count.fill(0)
        val result = recognize(captured)
        assertEquals(ListeningLimits.NO_SPEECH, result.endReason.get())
        assertTrue("The queued speech was never processed", result.speechStarts.get() > 0)
        assertEquals("Timeout recovery must not signal Send", 0, result.phraseEnds.get())
        val names =
            listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        val text = result.words.joinToString(" ").lowercase()
        val numbers =
            Regex("[a-z]+|[0-9]+")
                .findAll(text)
                .map {
                    it.value.toIntOrNull() ?: (names.indexOf(it.value) + 1)
                }
                .toList()
        assertEquals(text, (1..10).toList(), numbers)
    }

    private class Result {
        val endReason = AtomicInteger(-1)
        val speechStarts = AtomicInteger()
        val phraseEnds = AtomicInteger()
        val words = ConcurrentLinkedQueue<String>()
    }

    private fun recognize(pcm: ByteArray): Result {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return ActivityScenario.launch(MainActivity::class.java).use {
            val result = Result()
            val pipe = ParcelFileDescriptor.createPipe()
            val terminal = CountDownLatch(1)
            val error = AtomicInteger()
            val finished = AtomicInteger()
            val writerError = ConcurrentLinkedQueue<Throwable>()
            var worker: ITranscriptionWorker? = null
            val writer = Thread {
                try {
                    ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                        output.write(pcm)
                    }
                } catch (failure: Exception) {
                    writerError.add(failure)
                }
            }
            val events =
                object : ITranscriptionEvents.Stub() {
                    override fun ready() = writer.start()

                    override fun speech(speaking: Boolean) {
                        if (speaking) result.speechStarts.incrementAndGet()
                    }

                    override fun level(rms: Float) = Unit

                    override fun progress(completedMs: Long, durationMs: Long) = Unit

                    override fun inputEnded(reason: Int) {
                        result.endReason.set(reason)
                    }

                    override fun hypothesis(text: String) = Unit

                    override fun phraseFinished(request: Long) = Unit

                    override fun segment(text: String, phraseEnd: Boolean) {
                        if (text.isNotBlank()) result.words.add(text)
                        if (phraseEnd) result.phraseEnds.incrementAndGet()
                    }

                    override fun finished() {
                        finished.incrementAndGet()
                        terminal.countDown()
                    }

                    override fun failed(reason: Int) {
                        error.set(reason)
                        terminal.countDown()
                    }
                }
            val connection =
                object : ServiceConnection {
                    override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                        worker = ITranscriptionWorker.Stub.asInterface(binder)
                        // Buffered capture can wait for inference, while retaining live timeouts.
                        worker!!.start(pipe[0], false, true, true, true, 3000, events)
                        pipe[0].close()
                    }

                    override fun onServiceDisconnected(name: ComponentName) {
                        error.set(-1)
                        terminal.countDown()
                    }
                }
            assertTrue(
                context.bindIsolatedService(
                    Intent(context, TranscriptionWorker::class.java),
                    Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT,
                    "speech_detection_${UUID.randomUUID().toString().replace("-", "")}",
                    context.mainExecutor,
                    connection,
                )
            )
            try {
                assertTrue("Worker did not finish", terminal.await(60, TimeUnit.SECONDS))
                assertEquals("Recognition was cancelled instead of finalized", 0, error.get())
                assertEquals(1, finished.get())
                writer.join(2000)
                assertFalse("PCM writer did not finish", writer.isAlive)
                assertTrue(writerError.toString(), writerError.isEmpty())
                result
            } finally {
                runCatching { worker?.cancel() }
                context.unbindService(connection)
                pipe.forEach { runCatching { it.close() } }
                writer.interrupt()
                writer.join(2000)
                pcm.fill(0)
            }
        }
    }
}
