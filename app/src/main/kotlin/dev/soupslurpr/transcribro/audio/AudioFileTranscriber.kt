package dev.soupslurpr.transcribro.audio

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.res.AssetFileDescriptor
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.ParcelFileDescriptor
import dev.soupslurpr.transcribro.engine.DictationPriority
import dev.soupslurpr.transcribro.engine.ITranscriptionEvents
import dev.soupslurpr.transcribro.engine.ITranscriptionWorker
import dev.soupslurpr.transcribro.engine.TranscriptionWorker
import java.io.IOException
import java.util.UUID
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext

/** Platform decoding feeds bounded PCM to a unique isolated recognition worker. */
internal class AudioFileTranscriber(private val context: Context) {
    suspend fun transcribe(
        descriptor: AssetFileDescriptor,
        onProgress: (Long, Long) -> Unit,
        onPhrase: (String) -> Unit,
    ): Unit = coroutineScope {
        val pipe = ParcelFileDescriptor.createPipe()
        val duration = java.util.concurrent.atomic.AtomicLong(-1)
        try {
            val recognition = async {
                recognize(
                    pipe[0],
                    { elapsed, reported ->
                        onProgress(
                            elapsed,
                            if (reported >= 0) {
                                reported
                            } else {
                                duration.get()
                            },
                        )
                    },
                    onPhrase,
                )
            }
            val decoding =
                launch(Dispatchers.IO) {
                    try {
                        ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { stream ->
                            AudioFileDecoder { !isActive }
                                .decode(
                                    descriptor.parcelFileDescriptor,
                                    descriptor.startOffset,
                                    descriptor.declaredLength,
                                ) { samples, length ->
                                    duration.set(length)
                                    val bytes = ByteArray(samples.size * 2)
                                    samples.forEachIndexed { index, sample ->
                                        bytes[index * 2] = sample.toByte()
                                        bytes[index * 2 + 1] = (sample.toInt() shr 8).toByte()
                                    }
                                    try {
                                        stream.write(bytes)
                                    } finally {
                                        bytes.fill(0)
                                    }
                                }
                        }
                    } catch (failure: Exception) {
                        if (isActive) {
                            throw IOException(
                                if (failure is AudioFileDecoder.AudioTooLongException) {
                                    "Choose audio no longer than 30 minutes. Completed text is kept."
                                } else {
                                    "Couldn't decode this audio. Try a WAV, MP3, or M4A file. Completed text is kept."
                                },
                                failure,
                            )
                        }
                    }
                }
            recognition.await()
            decoding.join()
        } finally {
            runCatching { pipe[0].close() }
            runCatching { pipe[1].close() }
        }
    }

    private suspend fun recognize(
        pcm: ParcelFileDescriptor,
        onProgress: (Long, Long) -> Unit,
        onPhrase: (String) -> Unit,
    ): Unit =
        withContext(Dispatchers.Main.immediate) {
            suspendCancellableCoroutine { continuation ->
                val main = Handler(Looper.getMainLooper())
                var worker: ITranscriptionWorker? = null
                var bound = false
                var priority: Job? = null
                lateinit var connection: ServiceConnection
                lateinit var timeout: Runnable
                fun release() {
                    main.removeCallbacks(timeout)
                    priority?.cancel()
                    priority = null
                    runCatching { worker?.cancel() }
                    worker = null
                    runCatching { pcm.close() }
                    if (bound) {
                        bound = false
                        context.unbindService(connection)
                    }
                }
                fun fail(message: String) {
                    if (!continuation.isActive) return
                    release()
                    continuation.resumeWithException(IOException(message))
                }
                timeout = Runnable {
                    fail("Audio processing stopped responding. Completed text is kept.")
                }
                fun heartbeat() {
                    main.removeCallbacks(timeout)
                    if (!DictationPriority.active.value) main.postDelayed(timeout, 120_000)
                }
                val events =
                    object : ITranscriptionEvents.Stub() {
                        override fun phraseFinished(request: Long) = Unit

                        private fun deliver(action: () -> Unit) {
                            main.post {
                                if (continuation.isActive) {
                                    try {
                                        heartbeat()
                                        action()
                                    } catch (_: Exception) {
                                        fail("Couldn't finish this file. Completed text is kept.")
                                    }
                                }
                            }
                        }

                        override fun ready() = deliver {}

                        override fun speech(speaking: Boolean) = Unit

                        override fun level(rms: Float) = Unit

                        override fun inputEnded(reason: Int) = deliver {}

                        override fun progress(completedMs: Long, durationMs: Long) = deliver {
                            require(completedMs in 0..AudioFileDecoder.MAX_DURATION_MS)
                            require(durationMs in -1..AudioFileDecoder.MAX_DURATION_MS)
                            onProgress(completedMs, durationMs)
                        }

                        override fun segment(text: String, phraseEnd: Boolean) = deliver {
                            require(text.length <= 16_000)
                            if (text.isNotBlank()) onPhrase(text)
                        }

                        override fun hypothesis(text: String) = Unit

                        override fun finished() = deliver {
                            release()
                            continuation.resume(Unit)
                        }

                        override fun failed(reason: Int) = deliver {
                            fail(
                                if (reason == TranscriptionWorker.ERROR_FILE_TOO_LONG) {
                                    "Choose audio no longer than 30 minutes. Completed text is kept."
                                } else {
                                    "Couldn't transcribe this audio. Try a WAV, MP3, or M4A file. Completed text is kept."
                                }
                            )
                        }
                    }
                connection =
                    object : ServiceConnection {
                        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                            if (!continuation.isActive) {
                                release()
                                return
                            }
                            try {
                                worker = ITranscriptionWorker.Stub.asInterface(binder)
                                worker?.setPaused(DictationPriority.active.value)
                                worker?.startFile(pcm, events)
                                pcm.close()
                            } catch (_: Exception) {
                                fail("Couldn't open the selected audio file.")
                            }
                        }

                        override fun onServiceDisconnected(name: ComponentName) =
                            fail("Audio processing stopped. Completed text is kept.")

                        override fun onBindingDied(name: ComponentName) =
                            onServiceDisconnected(name)

                        override fun onNullBinding(name: ComponentName) =
                            onServiceDisconnected(name)
                    }
                continuation.invokeOnCancellation { main.post { release() } }
                priority =
                    CoroutineScope(continuation.context).launch(
                        start = CoroutineStart.UNDISPATCHED
                    ) {
                        DictationPriority.active.collect { paused ->
                            try {
                                worker?.setPaused(paused)
                                heartbeat()
                            } catch (_: Exception) {
                                fail("Audio processing stopped. Completed text is kept.")
                            }
                        }
                    }
                try {
                    bound =
                        context.bindIsolatedService(
                            Intent(context, TranscriptionWorker::class.java),
                            Context.BIND_AUTO_CREATE,
                            "file_" + UUID.randomUUID().toString().replace("-", ""),
                            context.mainExecutor,
                            connection,
                        )
                    if (bound) heartbeat() else fail("Couldn't start audio processing.")
                } catch (_: Exception) {
                    fail("Couldn't start audio processing.")
                }
            }
        }
}
