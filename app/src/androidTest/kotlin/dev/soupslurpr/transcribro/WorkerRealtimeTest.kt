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
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the live capture/inference concurrency with synthetic PCM, without a microphone. */
@RunWith(AndroidJUnit4::class)
class WorkerRealtimeTest {
    @Test fun pacedAudioCompletesWithFinalWordsAndLivePreviews() = exercise(paced = true)

    @Test
    fun inputThatOutrunsLiveInferenceDrainsCapturedWordsWithoutSending() = exercise(paced = false)

    @Test
    fun successivePacedCountsKeepListeningAndRetainEveryNumber() =
        exercise(paced = true, fixtureName = "count-to-ten-slower.pcm", repetitions = 6)

    @Test
    fun sustainedPacedCountsBenchmarkKeepsEveryNumber() {
        assumeTrue(
            "Enable explicitly with workerBenchmark=true",
            InstrumentationRegistry.getArguments().getString("workerBenchmark") == "true",
        )
        exercise(paced = true, fixtureName = "count-to-ten-slower.pcm", repetitions = 24)
    }

    private fun exercise(
        paced: Boolean,
        fixtureName: String = "dictation-fixture.pcm",
        repetitions: Int = 1,
    ) {
        // Keep the caller visibly active, matching a dictation session's lifecycle.
        ActivityScenario.launch(MainActivity::class.java).use { activity ->
            val deadline = android.os.SystemClock.elapsedRealtime() + 5_000
            var focused = false
            while (!focused && android.os.SystemClock.elapsedRealtime() < deadline) {
                activity.onActivity { focused = it.hasWindowFocus() }
                if (!focused) Thread.sleep(25)
            }
            assertTrue("The dictation test's client did not become visible", focused)
            exerciseVisible(paced, fixtureName, repetitions)
        }
    }

    private fun exerciseVisible(paced: Boolean, fixtureName: String, repetitions: Int) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val fixture = instrumentation.context.assets.open(fixtureName).use { it.readBytes() }
        val pipe = ParcelFileDescriptor.createPipe()
        val terminal = CountDownLatch(1)
        val error = AtomicInteger(0)
        val previews = AtomicInteger(0)
        val endReason = AtomicInteger(0)
        val unsafePhrases = AtomicInteger(0)
        val words = ConcurrentLinkedQueue<String>()
        var worker: ITranscriptionWorker? = null
        val writer = Thread {
            runCatching {
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                    repeat(repetitions) {
                        for (offset in fixture.indices step 1024) {
                            output.write(fixture, offset, minOf(1024, fixture.size - offset))
                            if (paced) Thread.sleep(32)
                        }
                    }
                }
            }
        }
        val events =
            object : ITranscriptionEvents.Stub() {
                override fun phraseFinished(request: Long) = Unit

                override fun ready() {
                    // Hold inference until the live queue fills. Capture/VAD still runs,
                    // making the overload case deterministic on fast and slow devices.
                    if (!paced) worker!!.setPaused(true)
                    writer.start()
                }

                override fun speech(speaking: Boolean) = Unit

                override fun level(rms: Float) = Unit

                override fun progress(completedMs: Long, durationMs: Long) = Unit

                override fun inputEnded(reason: Int) {
                    endReason.set(reason)
                    if (reason == ListeningLimits.OVERLOAD) worker!!.setPaused(false)
                }

                override fun segment(text: String, phraseEnd: Boolean) {
                    if (text.isNotBlank()) words.add(text)
                    if (phraseEnd && endReason.get() == ListeningLimits.OVERLOAD)
                        unsafePhrases.incrementAndGet()
                }

                override fun hypothesis(text: String) {
                    if (text.isNotBlank()) previews.incrementAndGet()
                }

                override fun finished() {
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
                    worker!!.start(pipe[0], false, true, false, true, 3000, events)
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
                "verification_${UUID.randomUUID().toString().replace("-", "")}",
                context.mainExecutor,
                connection,
            )
        )
        try {
            val timeoutSeconds = maxOf(45L, fixture.size.toLong() * repetitions / 32_000 + 30)
            assertTrue(
                "Live worker did not terminate",
                terminal.await(timeoutSeconds, TimeUnit.SECONDS),
            )
            assertEquals(0, error.get())
            val text = words.joinToString(" ").lowercase()
            if (fixtureName == "dictation-fixture.pcm") {
                assertTrue(text, text.contains("voice") && text.contains("on the table"))
            } else {
                val names =
                    listOf(
                        "one",
                        "two",
                        "three",
                        "four",
                        "five",
                        "six",
                        "seven",
                        "eight",
                        "nine",
                        "ten",
                    )
                val numbers =
                    text.replace(Regex("[^a-z0-9]+"), " ").trim().split(Regex("\\s+")).map {
                        it.toIntOrNull() ?: (names.indexOf(it) + 1)
                    }
                assertEquals(text, List(repetitions) { (1..10).toList() }.flatten(), numbers)
            }
            // A deliberately flooded queue can skip every obsolete preview after
            // its protective stop. Paced speech must still provide live text.
            if (paced) assertTrue("No tentative words", previews.get() > 0)
            assertEquals(
                if (paced) ListeningLimits.COMPLETE else ListeningLimits.OVERLOAD,
                endReason.get(),
            )
            assertEquals("Protective stop must not signal Send", 0, unsafePhrases.get())
        } finally {
            runCatching { worker?.cancel() }
            context.unbindService(connection)
            writer.interrupt()
            pipe.forEach { runCatching { it.close() } }
            writer.join(2_000)
            fixture.fill(0)
        }
    }
}
