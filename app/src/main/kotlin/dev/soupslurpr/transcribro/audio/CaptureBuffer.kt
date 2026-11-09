package dev.soupslurpr.transcribro.audio

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Memory-only PCM between capture and the isolated recognizer. Capture never waits for model
 * loading or inference. A slow recognizer stops capture before this buffer can overwrite speech.
 */
internal class CaptureBuffer(
    private val realtime: Boolean,
    private val drainReserve: Int,
    private val startupLimit: Int = 30 * 32_000,
    private val listeningLimit: Int = 4 * 32_000,
) {
    sealed interface Read {
        data class Audio(val size: Int) : Read

        data class Boundary(val request: Long, val afterSamples: Long) : Read
    }

    private val lock = ReentrantLock()
    private val changed = lock.newCondition()
    private val bytes = ByteArray(startupLimit + drainReserve)
    private val boundaries = ArrayDeque<Read.Boundary>()
    private var head = 0
    private var size = 0
    private var written = 0L
    private var consumed = 0L
    private var ready = false
    private var limit = startupLimit
    private var ended = false
    private var cancelled = false

    init {
        require(listeningLimit > 0 && startupLimit >= listeningLimit && drainReserve > 0)
    }

    /** The already captured startup backlog gets time to drain before the normal bound applies. */
    fun ready() = lock.withLock {
        ready = true
        limit = minOf(startupLimit, maxOf(listeningLimit, size + listeningLimit))
        changed.signalAll()
    }

    /** Returns true when live capture must stop, leaving room for its final recorder drain. */
    fun append(source: ByteArray, count: Int): Boolean = lock.withLock {
        require(count in 0..source.size)
        check(!ended)
        // Supplied PCM has a lossless, backpressured source rather than a live microphone.
        while (!realtime && size + count > limit && !cancelled) changed.await()
        if (cancelled) return false
        check(size + count <= bytes.size) { "Capture drain exceeded its reserved space" }
        val tail = (head + size) % bytes.size
        val first = minOf(count, bytes.size - tail)
        source.copyInto(bytes, tail, 0, first)
        source.copyInto(bytes, 0, first, count)
        size += count
        written += count
        changed.signalAll()
        realtime && size >= limit
    }

    fun finishPhrase(request: Long) = lock.withLock {
        if (cancelled || ended) return
        // Our controller permits one unacknowledged manual command at a time.
        check(boundaries.size < 8)
        boundaries.addLast(Read.Boundary(request, written / 2))
        changed.signalAll()
    }

    /** The feeder alone owns the audio pipe, preserving PCM and manual commands in FIFO order. */
    fun read(target: ByteArray): Read? = lock.withLock {
        require(target.isNotEmpty())
        while (size == 0 && boundaries.isEmpty() && !ended && !cancelled) changed.await()
        if (cancelled) return null
        val boundary = boundaries.firstOrNull()
        if (boundary != null && boundary.afterSamples * 2 <= consumed) {
            return boundaries.removeFirst()
        }
        if (size == 0) return null
        val count =
            minOf(
                target.size,
                size,
                boundary?.let { (it.afterSamples * 2 - consumed).toInt() } ?: size,
            )
        val first = minOf(count, bytes.size - head)
        bytes.copyInto(target, 0, head, head + first)
        bytes.copyInto(target, first, 0, count - first)
        bytes.fill(0, head, head + first)
        bytes.fill(0, 0, count - first)
        head = (head + count) % bytes.size
        size -= count
        consumed += count
        if (ready && size <= listeningLimit) limit = listeningLimit
        changed.signalAll()
        Read.Audio(count)
    }

    fun finish() = lock.withLock {
        ended = true
        changed.signalAll()
    }

    fun cancel() = lock.withLock {
        cancelled = true
        bytes.fill(0)
        boundaries.clear()
        size = 0
        changed.signalAll()
    }
}
