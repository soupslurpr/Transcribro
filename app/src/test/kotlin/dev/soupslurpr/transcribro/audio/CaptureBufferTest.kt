package dev.soupslurpr.transcribro.audio

import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.*
import org.junit.Test

class CaptureBufferTest {
    @Test
    fun finishBeforeModelReadinessKeepsAudioAndManualBoundariesInOrder() {
        val buffer = CaptureBuffer(true, 4, startupLimit = 32, listeningLimit = 8)
        val first = ByteArray(12) { it.toByte() }
        val second = ByteArray(12) { (it + 12).toByte() }
        assertFalse(buffer.append(first, first.size))
        buffer.finishPhrase(7)
        assertFalse(buffer.append(second, second.size))
        buffer.finish()
        buffer.ready()
        val read = ByteArray(8)
        val output = ByteArrayOutputStream()
        var boundarySeen = false
        while (true) {
            when (val next = buffer.read(read) ?: break) {
                is CaptureBuffer.Read.Audio -> output.write(read, 0, next.size)
                is CaptureBuffer.Read.Boundary -> {
                    assertArrayEquals(first, output.toByteArray())
                    assertEquals(CaptureBuffer.Read.Boundary(7, 6), next)
                    boundarySeen = true
                }
            }
        }
        assertTrue(boundarySeen)
        assertArrayEquals(first + second, output.toByteArray())
    }

    @Test
    fun startupBacklogMayCatchUpBeforeApplyingTheLiveBound() {
        val buffer = CaptureBuffer(true, 4, startupLimit = 32, listeningLimit = 8)
        val audio = ByteArray(20) { it.toByte() }
        assertFalse(buffer.append(audio, audio.size))
        buffer.ready()
        // Readiness must not truncate a startup backlog larger than the ordinary live bound.
        assertFalse(buffer.append(byteArrayOf(20, 21), 2))
        val read = ByteArray(16)
        assertEquals(CaptureBuffer.Read.Audio(16), buffer.read(read))
        assertArrayEquals(audio.take(16).toByteArray(), read)
        // The remaining six bytes now fit the live bound. Reaching it requests a stop.
        assertTrue(buffer.append(byteArrayOf(22, 23), 2))
        // The final hardware-buffer drain is retained too, even after the protective stop.
        assertTrue(buffer.append(byteArrayOf(24, 25, 26, 27), 4))
        buffer.finish()
        assertEquals(CaptureBuffer.Read.Audio(12), buffer.read(read))
        assertArrayEquals(ByteArray(12) { (it + 16).toByte() }, read.copyOf(12))
        assertNull(buffer.read(read))
    }

    @Test
    fun ringWrapRetainsEveryByteWithoutRepeatingPreviousSpeech() {
        val buffer = CaptureBuffer(true, 4, startupLimit = 16, listeningLimit = 8)
        buffer.ready()
        val read = ByteArray(6)
        repeat(100) { step ->
            val audio = ByteArray(6) { (it + step * 6).toByte() }
            assertFalse(buffer.append(audio, audio.size))
            assertEquals(CaptureBuffer.Read.Audio(6), buffer.read(read))
            assertArrayEquals(audio, read)
        }
        buffer.finish()
        assertNull(buffer.read(read))
    }

    @Test
    fun cancellationWakesAWaitingFeederWithoutDeliveringMoreAudio() {
        val buffer = CaptureBuffer(true, 4, startupLimit = 16, listeningLimit = 8)
        val started = CountDownLatch(1)
        val ended = CountDownLatch(1)
        var result: CaptureBuffer.Read? = CaptureBuffer.Read.Audio(1)
        val thread = Thread {
            started.countDown()
            result = buffer.read(ByteArray(4))
            ended.countDown()
        }
        thread.start()
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS))
            assertFalse(ended.await(50, TimeUnit.MILLISECONDS))
            buffer.cancel()
            assertTrue(ended.await(2, TimeUnit.SECONDS))
            assertNull(result)
            buffer.append(byteArrayOf(1, 2), 2)
            assertNull(buffer.read(ByteArray(4)))
        } finally {
            buffer.cancel()
            thread.join(2000)
        }
    }

    @Test
    fun suppliedAudioWaitsForSpaceAndCancellationReleasesItsProducer() {
        val buffer = CaptureBuffer(false, 4, startupLimit = 16, listeningLimit = 8)
        buffer.ready()
        val audio = ByteArray(8) { it.toByte() }
        assertFalse(buffer.append(audio, audio.size))
        val started = CountDownLatch(1)
        val ended = CountDownLatch(1)
        val thread = Thread {
            started.countDown()
            buffer.append(audio, audio.size)
            ended.countDown()
        }
        thread.start()
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS))
            assertFalse(ended.await(50, TimeUnit.MILLISECONDS))
            buffer.cancel()
            assertTrue(ended.await(2, TimeUnit.SECONDS))
            assertNull(buffer.read(ByteArray(8)))
        } finally {
            buffer.cancel()
            thread.join(2000)
        }
    }
}
