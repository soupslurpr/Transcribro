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
import dev.soupslurpr.transcribro.audio.CaptureBuffer
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

/** Startup PCM reaches the real live inference queue, including Finish before model loading. */
@RunWith(AndroidJUnit4::class)
class BufferedStartupTest {
    @Test fun finishAndManualBoundaryBeforeModelLoadingRetainBothCompleteCounts() = exercise(false)

    @Test fun protectiveCaptureStopKeepsBufferedSpeechWithoutAutoSend() = exercise(true)

    @Test
    fun speechStartingAtTheFirstSampleKeepsItsOpeningNumberThroughStartup() =
        exercise(overload = false, trimLeadingSilence = true)

    private fun exercise(overload: Boolean, trimLeadingSilence: Boolean = false) {
        ActivityScenario.launch(MainActivity::class.java).use {
            exerciseVisible(overload, trimLeadingSilence)
        }
    }

    private fun exerciseVisible(overload: Boolean, trimLeadingSilence: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val fixture = instrumentation.context.assets.open("count-to-ten.pcm").use { it.readBytes() }
        val first =
            if (trimLeadingSilence)
                (fixture.indices step 2).first { index ->
                    val sample =
                        ((fixture[index].toInt() and 255) or (fixture[index + 1].toInt() shl 8))
                            .toShort()
                            .toInt()
                    kotlin.math.abs(sample) >= 100
                }
            else 0
        val count = fixture.copyOfRange(first, fixture.size)
        fixture.fill(0)
        val capture =
            if (overload) CaptureBuffer(true, 2048, count.size * 2, count.size)
            else CaptureBuffer(true, 2048)
        val pipe = ParcelFileDescriptor.createPipe()
        val connected = CountDownLatch(1)
        val ready = CountDownLatch(1)
        val terminal = CountDownLatch(1)
        val error = AtomicInteger()
        val endReason = AtomicInteger(-1)
        val unsafeSends = AtomicInteger()
        val words = ConcurrentLinkedQueue<String>()
        val boundaryWords = ConcurrentLinkedQueue<String>()
        var worker: ITranscriptionWorker? = null
        val feeder = Thread {
            val bytes = ByteArray(2048)
            try {
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                    while (true) {
                        when (val next = capture.read(bytes) ?: break) {
                            is CaptureBuffer.Read.Audio -> output.write(bytes, 0, next.size)
                            is CaptureBuffer.Read.Boundary ->
                                worker!!.finishPhrase(next.request, next.afterSamples)
                        }
                        bytes.fill(0)
                    }
                }
            } catch (_: Exception) {
                error.set(-2)
                terminal.countDown()
            } finally {
                bytes.fill(0)
            }
        }
        val events =
            object : ITranscriptionEvents.Stub() {
                override fun ready() {
                    capture.ready()
                    ready.countDown()
                    feeder.start()
                }

                override fun speech(speaking: Boolean) = Unit

                override fun level(rms: Float) = Unit

                override fun progress(completedMs: Long, durationMs: Long) = Unit

                override fun hypothesis(text: String) = Unit

                override fun inputEnded(reason: Int) = endReason.set(reason)

                override fun segment(text: String, phraseEnd: Boolean) {
                    if (text.isNotBlank()) words.add(text)
                    if (phraseEnd && overload) unsafeSends.incrementAndGet()
                }

                override fun phraseFinished(request: Long) {
                    if (request != 7L) error.set(-3)
                    boundaryWords.addAll(words)
                }

                override fun finished() = terminal.countDown()

                override fun failed(reason: Int) {
                    error.set(reason)
                    terminal.countDown()
                }
            }
        val connection =
            object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                    worker = ITranscriptionWorker.Stub.asInterface(binder)
                    // Hold initialization itself, not just the later inference tasks.
                    worker!!.setPaused(true)
                    worker!!.start(pipe[0], false, true, true, true, 3000, events)
                    pipe[0].close()
                    connected.countDown()
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
                "startup_${UUID.randomUUID().toString().replace("-", "")}",
                context.mainExecutor,
                connection,
            )
        )
        try {
            assertTrue(connected.await(10, TimeUnit.SECONDS))
            if (overload) capture.ready()
            assertEquals(overload, capture.append(count, count.size))
            if (overload) {
                worker!!.captureOverloaded()
            } else {
                capture.finishPhrase(7)
                assertFalse(capture.append(count, count.size))
            }
            capture.finish()
            assertEquals("Capture must finish before model readiness", 1L, ready.count)
            worker!!.setPaused(false)
            assertTrue("Buffered startup did not finish", terminal.await(60, TimeUnit.SECONDS))
            assertEquals(0, error.get())
            assertEquals(
                if (overload) ListeningLimits.OVERLOAD else ListeningLimits.COMPLETE,
                endReason.get(),
            )
            assertEquals(0, unsafeSends.get())
            val expected = (1..10).toList()
            assertEquals(if (overload) expected else expected + expected, numbers(words))
            if (!overload) assertEquals(expected, numbers(boundaryWords))
        } finally {
            runCatching { worker?.cancel() }
            capture.cancel()
            context.unbindService(connection)
            pipe.forEach { runCatching { it.close() } }
            feeder.interrupt()
            feeder.join(2000)
            count.fill(0)
        }
    }

    private fun numbers(text: Collection<String>): List<Int> {
        val names =
            listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        return Regex("[a-z]+|[0-9]+")
            .findAll(text.joinToString(" ").lowercase())
            .map { it.value.toIntOrNull() ?: (names.indexOf(it.value) + 1) }
            .toList()
    }
}
