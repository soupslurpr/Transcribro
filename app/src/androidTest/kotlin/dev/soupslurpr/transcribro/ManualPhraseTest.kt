package dev.soupslurpr.transcribro

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.ParcelFileDescriptor
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.engine.ITranscriptionEvents
import dev.soupslurpr.transcribro.engine.ITranscriptionWorker
import dev.soupslurpr.transcribro.engine.TranscriptionWorker
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** A control request must not overtake the audio pipe or consume the following phrase. */
@RunWith(AndroidJUnit4::class)
class ManualPhraseTest {
    @Test
    fun manualBoundaryDrainsItsWatermarkAndKeepsTheFollowingPhraseSeparate() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val count = instrumentation.context.assets.open("count-to-ten.pcm").use { it.readBytes() }
        // The marker ends inside a VAD frame, testing its partial-frame finalization too.
        val first = ByteArray(514) + count
        val pipe = ParcelFileDescriptor.createPipe()
        val boundary = CountDownLatch(1)
        val terminal = CountDownLatch(1)
        val error = AtomicInteger()
        val words = ConcurrentLinkedQueue<String>()
        val atBoundary = ConcurrentLinkedQueue<String>()
        var worker: ITranscriptionWorker? = null
        val writer = Thread {
            runCatching {
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                    fun write(bytes: ByteArray) {
                        for (offset in bytes.indices step 1024) {
                            output.write(bytes, offset, minOf(1024, bytes.size - offset))
                            Thread.sleep(32)
                        }
                    }
                    // Register before these bytes are consumed: a bare Binder flag would
                    // incorrectly finalize at the first frame instead of the watermark.
                    worker!!.finishPhrase(1, first.size / 2L)
                    write(first)
                    write(ByteArray(1024))
                    check(boundary.await(15, TimeUnit.SECONDS))
                    write(count)
                }
            }
                .onFailure {
                    error.set(-2)
                    terminal.countDown()
                }
        }
        val events =
            object : ITranscriptionEvents.Stub() {
                override fun ready() = writer.start()

                override fun speech(speaking: Boolean) = Unit

                override fun level(rms: Float) = Unit

                override fun progress(completedMs: Long, durationMs: Long) = Unit

                override fun inputEnded(reason: Int) = Unit

                override fun hypothesis(text: String) = Unit

                override fun segment(text: String, phraseEnd: Boolean) {
                    if (text.isNotBlank()) words.add(text)
                }

                override fun phraseFinished(request: Long) {
                    if (request != 1L) error.set(-3)
                    atBoundary.addAll(words)
                    boundary.countDown()
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
                Context.BIND_AUTO_CREATE,
                "manual_${UUID.randomUUID().toString().replace("-", "")}",
                context.mainExecutor,
                connection,
            )
        )
        try {
            assertTrue("Worker never finished", terminal.await(45, TimeUnit.SECONDS))
            assertEquals(0, error.get())
            assertEquals("Missing boundary acknowledgement", 0L, boundary.count)
            assertEquals((1..10).toList(), numbers(atBoundary))
            assertEquals((1..10).toList() + (1..10).toList(), numbers(words))
        } finally {
            runCatching { worker?.cancel() }
            context.unbindService(connection)
            pipe.forEach { runCatching { it.close() } }
            writer.interrupt()
            writer.join(2000)
            first.fill(0)
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
