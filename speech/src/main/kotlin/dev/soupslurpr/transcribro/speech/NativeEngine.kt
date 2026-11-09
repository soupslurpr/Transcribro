package dev.soupslurpr.transcribro.speech

import android.content.res.AssetManager

/**
 * Owned by one isolated recognition session. VAD and inference have separate contexts. cancel() may
 * run concurrently; close() must follow completion of both processing threads.
 */
class NativeEngine(assets: AssetManager, previews: Boolean = true) : AutoCloseable {
    private var handle = nativeCreate(assets, previews)

    init {
        check(handle != 0L) { "Could not load the speech models" }
    }

    fun speechProbability(samples: FloatArray): Float {
        require(samples.size == 512)
        return nativeVad(handle, samples)
    }

    /** Finalized fragments and current preview; empty when analysis is deliberately deferred. */
    fun stream(
        samples: FloatArray,
        finishPhrase: Boolean,
        deferAnalysis: Boolean = false,
    ): List<String> {
        require(samples.size <= 8192)
        return nativeStream(handle, samples, finishPhrase, deferAnalysis).map {
            it.toString(Charsets.UTF_8).trim()
        }
    }

    @Synchronized
    fun setPaused(paused: Boolean) {
        if (handle != 0L) nativePause(handle, paused)
    }

    @Synchronized
    fun cancel() {
        if (handle != 0L) nativeCancel(handle)
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0
    }

    private external fun nativeCreate(assets: AssetManager, previews: Boolean): Long

    private external fun nativeVad(handle: Long, samples: FloatArray): Float

    private external fun nativeStream(
        handle: Long,
        samples: FloatArray,
        finishPhrase: Boolean,
        deferAnalysis: Boolean,
    ): Array<ByteArray>

    private external fun nativeCancel(handle: Long)

    private external fun nativePause(handle: Long, paused: Boolean)

    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("transcribro")
        }
    }
}
