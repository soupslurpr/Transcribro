package dev.soupslurpr.transcribro.engine

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import org.junit.Assert.*
import org.junit.Test

class ProcessingGateTest {
    @Test
    fun pauseBlocksUntilResumedWithoutPolling() {
        val gate = ProcessingGate()
        gate.setPaused(true)
        val waiting = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val permitted = AtomicBoolean()
        val reader = Thread {
            waiting.countDown()
            permitted.set(gate.await())
            finished.countDown()
        }
        reader.start()
        assertTrue(waiting.await(1, TimeUnit.SECONDS))
        assertFalse(finished.await(100, TimeUnit.MILLISECONDS))
        gate.setPaused(false)
        assertTrue(finished.await(1, TimeUnit.SECONDS))
        assertTrue(permitted.get())
        reader.join()
    }

    @Test
    fun cancellationWakesPausedReaderAndCannotBeUndoneByResume() {
        val gate = ProcessingGate()
        gate.setPaused(true)
        val finished = CountDownLatch(1)
        val permitted = AtomicBoolean(true)
        val reader = Thread {
            permitted.set(gate.await())
            finished.countDown()
        }
        reader.start()
        gate.cancel()
        assertTrue(finished.await(1, TimeUnit.SECONDS))
        assertFalse(permitted.get())
        gate.setPaused(false)
        assertFalse(gate.await())
        reader.join()
    }

    @Test
    fun filesYieldUntilTheLastDictationLeaseEnds() {
        val first = DictationPriority.acquire()
        val second = DictationPriority.acquire()
        try {
            assertTrue(DictationPriority.active.value)
            first.close()
            first.close()
            assertTrue(DictationPriority.active.value)
            second.close()
            assertFalse(DictationPriority.active.value)
        } finally {
            first.close()
            second.close()
        }
    }
}
