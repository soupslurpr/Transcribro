package dev.soupslurpr.transcribro.engine

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** Pauses without polling or discarding buffered audio. Cancellation always wakes the reader. */
internal class ProcessingGate {
    private val lock = ReentrantLock()
    private val changed = lock.newCondition()
    private var stopped = false
    var paused = false
        private set

    fun setPaused(value: Boolean) = lock.withLock {
        paused = value
        changed.signalAll()
    }

    fun await(): Boolean = lock.withLock {
        while (paused && !stopped) changed.await()
        !stopped
    }

    fun cancel() = lock.withLock {
        stopped = true
        changed.signalAll()
    }
}
