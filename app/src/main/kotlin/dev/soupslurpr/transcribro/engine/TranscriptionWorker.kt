package dev.soupslurpr.transcribro.engine

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.speech.SpeechRecognizer
import dev.soupslurpr.transcribro.audio.AudioFileDecoder
import dev.soupslurpr.transcribro.audio.DescriptorInput
import dev.soupslurpr.transcribro.speech.NativeEngine
import java.io.Closeable
import java.io.IOException
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.log10
import kotlin.math.sqrt

/** Unique isolated UID: no microphone permission, network, or access to app-private files. */
class TranscriptionWorker : Service() {
    private val started = AtomicBoolean()
    private val cancelled = AtomicBoolean()
    private val terminal = AtomicBoolean()
    private val reportedInputEnd = AtomicBoolean()
    private val gate = ProcessingGate()

    private data class PhraseBoundary(val request: Long, val afterSamples: Long)

    private val requestedPhrase = AtomicReference<PhraseBoundary?>()
    private val completionReason = AtomicInteger(ListeningLimits.COMPLETE)
    private val engineState = Any()

    @Volatile private var engine: NativeEngine? = null

    @Volatile private var input: Closeable? = null

    @Volatile private var events: ITranscriptionEvents? = null
    private val inference =
        ThreadPoolExecutor(
            1,
            1,
            0,
            TimeUnit.SECONDS,
            // Warm-device calls can take nearly three seconds even when average inference
            // stays ahead of speech. Buffer four seconds without dropping those words.
            ArrayBlockingQueue(8),
            { task -> Thread(task, "Transcription") },
            ThreadPoolExecutor.AbortPolicy(),
        )

    override fun onBind(intent: Intent): IBinder =
        object : ITranscriptionWorker.Stub() {
            override fun start(
                audio: ParcelFileDescriptor,
                stopOnSilence: Boolean,
                realtime: Boolean,
                bufferedCapture: Boolean,
                previews: Boolean,
                pauseMillis: Int,
                callback: ITranscriptionEvents,
            ) {
                launch(audio, callback, previews) { native ->
                    DescriptorInput(audio, cancelled::get).use { stream ->
                        val pipeline = Pipeline(native, realtime, pauseMillis, bufferedCapture)
                        val bytes = ByteArray(1024)
                        var everSpoke = false
                        val limits = ListeningLimits(realtime)
                        while (!cancelled.get()) {
                            var count = 0
                            while (count < bytes.size) {
                                val read = stream.read(bytes, count, bytes.size - count)
                                if (read < 0) break
                                count += read
                            }
                            if (count == 0) break
                            if (count % 2 != 0) throw IOException("Incomplete PCM sample")
                            val samples =
                                ShortArray(count / 2) { index ->
                                    ((bytes[index * 2].toInt() and 255) or
                                            (bytes[index * 2 + 1].toInt() shl 8))
                                        .toShort()
                                }
                            pipeline.accept(samples)
                            samples.fill(0)
                            everSpoke = everSpoke || pipeline.segmenter.speaking
                            if (completionReason.get() == ListeningLimits.COMPLETE) {
                                if (
                                    stopOnSilence && pipeline.segmenter.completedPhrase && everSpoke
                                ) {
                                    break
                                }
                                // Capture can signal overload on a Binder thread while VAD is
                                // running. Never replace that protective stop with COMPLETE.
                                val reason = limits.accept(count / 2, pipeline.segmenter.speaking)
                                if (
                                    completionReason.compareAndSet(
                                        ListeningLimits.COMPLETE,
                                        reason,
                                    ) && reason == ListeningLimits.NO_SPEECH
                                ) {
                                    // Stop the microphone now, then drain audio it already
                                    // captured. The recognizer can still recover words.
                                    reportInputEnded()
                                }
                            }
                            val reason = completionReason.get()
                            if (
                                reason != ListeningLimits.COMPLETE &&
                                    reason != ListeningLimits.OVERLOAD &&
                                    reason != ListeningLimits.NO_SPEECH
                            )
                                break
                            // The broker stops capture on overload or a no-speech timeout.
                            // Drain its remaining PCM through EOF, retaining the protective
                            // reason so neither final words nor pauses can trigger sending.
                            if (count < bytes.size) break
                        }
                        bytes.fill(0)
                        pipeline.finish()
                    }
                }
            }

            override fun startFile(audio: ParcelFileDescriptor, callback: ITranscriptionEvents) {
                launch(audio, callback, previews = false) { native ->
                    val pipeline = Pipeline(native, realtime = false)
                    var samples = 0L
                    var reported = -1L
                    DescriptorInput(audio, cancelled::get).use { stream ->
                        val bytes = ByteArray(1024)
                        while (!cancelled.get()) {
                            if (!gate.await()) break
                            var count = 0
                            while (count < bytes.size) {
                                val read = stream.read(bytes, count, bytes.size - count)
                                if (read < 0) break
                                count += read
                            }
                            if (count == 0) break
                            if (count % 2 != 0) throw IOException("Incomplete PCM sample")
                            val chunk =
                                ShortArray(count / 2) { index ->
                                    ((bytes[index * 2].toInt() and 255) or
                                            (bytes[index * 2 + 1].toInt() shl 8))
                                        .toShort()
                                }
                            samples += chunk.size
                            if (samples > AudioFileDecoder.MAX_DURATION_MS * 16) {
                                throw AudioFileDecoder.AudioTooLongException()
                            }
                            pipeline.accept(chunk)
                            chunk.fill(0)
                            val elapsed = samples / 16
                            if (elapsed - reported >= 250) {
                                events?.progress(elapsed, -1)
                                reported = elapsed
                            }
                            if (count < bytes.size) break
                        }
                        bytes.fill(0)
                    }
                    pipeline.finish()
                    events?.progress(samples / 16, samples / 16)
                }
            }

            override fun cancel() = cancelWork()

            override fun captureOverloaded() {
                completionReason.set(ListeningLimits.OVERLOAD)
            }

            override fun finishPhrase(request: Long, afterSamples: Long) {
                if (request > 0 && afterSamples >= 0 && !cancelled.get())
                    requestedPhrase.compareAndSet(null, PhraseBoundary(request, afterSamples))
            }

            override fun setPaused(paused: Boolean) {
                synchronized(engineState) {
                    gate.setPaused(paused)
                    engine?.setPaused(paused)
                }
            }
        }

    private fun launch(
        audio: ParcelFileDescriptor,
        callback: ITranscriptionEvents,
        previews: Boolean = true,
        consume: (NativeEngine) -> Unit,
    ) {
        if (!started.compareAndSet(false, true) || cancelled.get()) {
            audio.close()
            return
        }
        events = callback
        input = audio
        Thread(
                {
                    var native: NativeEngine? = null
                    try {
                        if (!gate.await()) return@Thread
                        val loaded = NativeEngine(assets, previews)
                        native = loaded
                        synchronized(engineState) {
                            engine = loaded
                            loaded.setPaused(gate.paused)
                        }
                        if (cancelled.get()) return@Thread
                        events?.ready()
                        consume(loaded)
                        reportInputEnded()
                        audio.close()
                        input = null
                        if (!cancelled.get()) events?.speech(false)
                        inference.shutdown()
                        if (!inference.awaitTermination(60, TimeUnit.SECONDS)) {
                            fail(SpeechRecognizer.ERROR_SERVER)
                        }
                        if (!cancelled.get() && terminal.compareAndSet(false, true))
                            events?.finished()
                    } catch (failure: Exception) {
                        if (!cancelled.get()) {
                            fail(
                                when (failure) {
                                    is AudioFileDecoder.AudioTooLongException -> ERROR_FILE_TOO_LONG
                                    is IOException,
                                    is IllegalArgumentException -> SpeechRecognizer.ERROR_AUDIO
                                    else -> SpeechRecognizer.ERROR_SERVER
                                }
                            )
                        }
                    } finally {
                        input = null
                        runCatching { audio.close() }
                        native?.cancel()
                        discardPendingInference()
                        // Native memory remains owned until the inference thread actually returns.
                        while (!inference.awaitTermination(1, TimeUnit.SECONDS)) {}
                        engine = null
                        native?.close()
                    }
                },
                "Audio processing",
            )
            .start()
    }

    private inner class Pipeline(
        private val native: NativeEngine,
        private val realtime: Boolean,
        pauseMillis: Int = 3000,
        private val bufferedCapture: Boolean = false,
    ) {
        val segmenter = SpeechSegmenter((pauseMillis.coerceIn(500, 10_000) + 31) / 32)
        private val frame = FloatArray(512)
        private var size = 0
        private var frames = 0
        private var lastSpeaking = false
        private val pending = FloatArray(8192)
        private var pendingSize = 0
        private var acceptedSamples = 0L

        fun accept(samples: ShortArray) {
            finishRequestedPhrase()
            for (sample in samples) {
                if (cancelled.get()) return
                frame[size++] = sample / 32768f
                acceptedSamples++
                if (size == frame.size) processFrame()
                else if (requestedPhrase.get()?.afterSamples == acceptedSamples)
                    finishRequestedPhrase()
            }
        }

        private fun processFrame() {
            // Silero decides actual speech pauses independently of Moonshine's internal lines.
            segmenter.accept(native.speechProbability(frame))
            frame.copyInto(pending, pendingSize)
            pendingSize += frame.size
            if (pendingSize == pending.size || segmenter.completedPhrase)
                flush(segmenter.completedPhrase)
            if (lastSpeaking != segmenter.speaking) {
                lastSpeaking = segmenter.speaking
                events?.speech(lastSpeaking)
            }
            if (realtime && frames % 4 == 0) {
                val rms = sqrt(frame.sumOf { (it * it).toDouble() } / frame.size)
                events?.level((20 * log10(rms.coerceAtLeast(0.00001))).toFloat())
            }
            frames++
            size = 0
            finishRequestedPhrase()
        }

        private fun finishRequestedPhrase() {
            val boundary = requestedPhrase.get() ?: return
            if (
                acceptedSamples >= boundary.afterSamples &&
                    requestedPhrase.compareAndSet(boundary, null)
            ) {
                if (size > 0) {
                    frame.fill(0f, size)
                    processFrame()
                }
                flush(finishPhrase = true, phraseEnd = false, request = boundary.request)
            }
        }

        fun finish() {
            if (cancelled.get()) return
            if (size > 0) {
                frame.fill(0f, size)
                processFrame()
            }
            finishRequestedPhrase()
            flush(finishPhrase = true, phraseEnd = false)
            frame.fill(0f)
        }

        private fun flush(
            finishPhrase: Boolean,
            phraseEnd: Boolean = finishPhrase,
            request: Long = 0,
        ) {
            val samples = pending.copyOf(pendingSize)
            pending.fill(0f)
            pendingSize = 0
            enqueue(native, samples, realtime, bufferedCapture, finishPhrase, phraseEnd, request)
        }
    }

    private var lastHypothesis = ""

    private fun enqueue(
        native: NativeEngine,
        samples: FloatArray,
        realtime: Boolean,
        bufferedCapture: Boolean,
        finishPhrase: Boolean,
        phraseEnd: Boolean = false,
        request: Long = 0,
    ) {
        if (cancelled.get()) {
            samples.fill(0f)
            return
        }
        val task =
            AudioTask(samples) {
                try {
                    if (!gate.await()) return@AudioTask
                    val result =
                        native.stream(
                            samples,
                            finishPhrase,
                            // Catch up by combining obsolete preview work. Native input and
                            // every final/manual boundary remain in their original FIFO order.
                            deferAnalysis = realtime && !finishPhrase && inference.queue.size >= 2,
                        )
                    if (cancelled.get()) return@AudioTask
                    if (result.isEmpty()) return@AudioTask
                    val completed = result.dropLast(1).filter(String::isNotBlank)
                    for (text in completed) events?.segment(text, false)
                    val hypothesis = result.lastOrNull().orEmpty()
                    if (hypothesis != lastHypothesis || completed.isNotEmpty()) {
                        events?.hypothesis(hypothesis)
                        lastHypothesis = hypothesis
                    }
                    if (phraseEnd && completionReason.get() == ListeningLimits.COMPLETE)
                        events?.segment("", true)
                    if (request != 0L) events?.phraseFinished(request)
                } catch (_: Exception) {
                    if (!cancelled.get()) fail(SpeechRecognizer.ERROR_SERVER)
                }
            }
        // Files and supplied PCM can wait for recognition, rather than overflowing a live queue.
        if (!realtime) {
            task.run()
        } else {
            try {
                inference.execute(task)
            } catch (_: java.util.concurrent.RejectedExecutionException) {
                if (!cancelled.get() && !inference.isShutdown) {
                    if (!bufferedCapture && completionReason.get() != ListeningLimits.OVERLOAD) {
                        completionReason.set(ListeningLimits.OVERLOAD)
                        // Stop collecting new speech, keeping the current and queued chunks.
                        // The queue remains bounded while the producer waits for inference.
                        reportInputEnded()
                    }
                    // The microphone broker owns a separate bounded capture buffer, including
                    // speech recorded during model loading. It detects sustained backlog and
                    // stops capture itself; a burst of startup PCM is not an overload here.
                    while (!cancelled.get() && !inference.isShutdown) {
                        if (inference.queue.offer(task, 100, TimeUnit.MILLISECONDS)) {
                            if ((cancelled.get() || inference.isShutdown) && inference.remove(task))
                                task.discard()
                            return
                        }
                    }
                }
                task.discard()
            }
        }
    }

    private class AudioTask(private val samples: FloatArray, private val transcribe: () -> Unit) :
        Runnable {
        override fun run() {
            try {
                transcribe()
            } finally {
                discard()
            }
        }

        fun discard() = samples.fill(0f)
    }

    private fun discardPendingInference() {
        inference.shutdownNow().forEach { (it as? AudioTask)?.discard() }
    }

    private fun reportInputEnded() {
        if (!cancelled.get() && reportedInputEnd.compareAndSet(false, true))
            events?.inputEnded(completionReason.get())
    }

    private fun fail(reason: Int) {
        if (terminal.compareAndSet(false, true)) runCatching { events?.failed(reason) }
        cancelWork()
    }

    private fun cancelWork() {
        cancelled.set(true)
        gate.cancel()
        engine?.cancel()
        runCatching { input?.close() }
        discardPendingInference()
    }

    override fun onDestroy() {
        cancelWork()
        super.onDestroy()
    }

    companion object {
        const val ERROR_FILE_TOO_LONG = 1000
    }
}
