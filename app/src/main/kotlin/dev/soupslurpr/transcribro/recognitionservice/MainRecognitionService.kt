package dev.soupslurpr.transcribro.recognitionservice

import android.Manifest
import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Context
import android.content.ContextParams
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.Process
import android.speech.RecognitionService
import android.speech.RecognitionSupport
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import dev.soupslurpr.transcribro.audio.CaptureBuffer
import dev.soupslurpr.transcribro.audio.DescriptorInput
import dev.soupslurpr.transcribro.dictation.PhraseCommands
import dev.soupslurpr.transcribro.engine.DictationPriority
import dev.soupslurpr.transcribro.engine.ITranscriptionEvents
import dev.soupslurpr.transcribro.engine.ITranscriptionWorker
import dev.soupslurpr.transcribro.engine.ListeningLimits
import dev.soupslurpr.transcribro.engine.Transcript
import dev.soupslurpr.transcribro.engine.TranscriptionWorker
import java.io.InputStream
import java.io.OutputStream
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.log10
import kotlin.math.sqrt

/** Permission-aware microphone broker. All native inference runs in an isolated UID. */
class MainRecognitionService : RecognitionService() {
    private val main = Handler(Looper.getMainLooper())
    private var current: Session? = null

    @SuppressLint(
        "MissingPermission"
    ) // RecognitionService + caller-attributed AudioRecord enforce this.
    override fun onStartListening(intent: Intent, callback: Callback) {
        val caller = callback.callingAttributionSource
        if (
            checkPermission(Manifest.permission.RECORD_AUDIO, caller.pid, caller.uid) !=
                PackageManager.PERMISSION_GRANTED
        ) {
            closeSuppliedAudio(intent)
            callback.error(SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
            return
        }
        // Register attribution before returning to RecognitionService's dispatch.
        // AudioRecord enforces live microphone access through this context. Supplied
        // PCM never opens the microphone and must not start microphone data delivery.
        val attributed =
            try {
                createContext(ContextParams.Builder().setNextAttributionSource(caller).build())
            } catch (_: SecurityException) {
                closeSuppliedAudio(intent)
                callback.error(SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
                return
            }
        if (current != null) {
            closeSuppliedAudio(intent)
            callback.error(SpeechRecognizer.ERROR_RECOGNIZER_BUSY)
            return
        }
        if (!supportsLanguage(intent)) {
            closeSuppliedAudio(intent)
            callback.error(SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED)
            return
        }
        val session =
            Session(callback, intent.getBooleanExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false))
        current = session
        if (caller.uid == Process.myUid()) {
            session.controlToken = intent.getStringExtra(EXTRA_CONTROL_TOKEN)
            session.controlToken?.let { token ->
                PhraseCommands.register(token) { request ->
                    if (!session.stopping) session.requestedPhrase.compareAndSet(0, request)
                }
            }
        }
        try {
            val supplied =
                intent.getParcelableExtra(
                    RecognizerIntent.EXTRA_AUDIO_SOURCE,
                    ParcelFileDescriptor::class.java,
                )
            if (intent.hasExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE) && supplied == null) {
                finish(session, SpeechRecognizer.ERROR_AUDIO)
                return
            }
            if (supplied != null) {
                if (
                    intent.getIntExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000) !=
                        16000 ||
                        intent.getIntExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1) !=
                            1 ||
                        intent.getIntExtra(
                            RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                            AudioFormat.ENCODING_PCM_16BIT,
                        ) != AudioFormat.ENCODING_PCM_16BIT
                ) {
                    supplied.close()
                    finish(session, SpeechRecognizer.ERROR_AUDIO)
                    return
                }
                session.source = DescriptorInput(supplied) { session.stopping }
            } else {
                if (
                    checkSelfPermission(Manifest.permission.RECORD_AUDIO) !=
                        PackageManager.PERMISSION_GRANTED
                ) {
                    finish(session, SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
                    return
                }
                session.recorder =
                    AudioRecord.Builder()
                        .setContext(attributed)
                        .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                        .setAudioFormat(
                            AudioFormat.Builder()
                                .setSampleRate(16000)
                                .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .build()
                        )
                        .setBufferSizeInBytes(
                            maxOf(
                                16384,
                                AudioRecord.getMinBufferSize(
                                    16000,
                                    AudioFormat.CHANNEL_IN_MONO,
                                    AudioFormat.ENCODING_PCM_16BIT,
                                ),
                            )
                        )
                        .build()
                check(session.recorder?.state == AudioRecord.STATE_INITIALIZED)
            }
            session.realtime = session.recorder != null
            session.capture =
                CaptureBuffer(
                    session.realtime,
                    drainReserve = (session.recorder?.bufferSizeInFrames ?: 0) * 2 + 2048,
                )
            val pipe = ParcelFileDescriptor.createPipe()
            session.readPipe = pipe[0]
            session.output = ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])
            session.stopOnSilence =
                if (intent.hasExtra(EXTRA_CONTINUOUS)) {
                    !intent.getBooleanExtra(EXTRA_CONTINUOUS, false)
                } else intent.getBooleanExtra(EXTRA_AUTO_STOP, true)
            session.previews = session.partials && intent.getBooleanExtra(EXTRA_LIVE_TEXT, true)
            session.pauseMillis =
                intent
                    .getLongExtra(
                        RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
                        3000L,
                    )
                    .coerceIn(500L, 10_000L)
                    .toInt()
            session.bound =
                bindIsolatedService(
                    Intent(this, TranscriptionWorker::class.java),
                    // Keep the isolated worker important while the speech broker is active.
                    Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT,
                    "dictation_" + UUID.randomUUID().toString().replace("-", ""),
                    mainExecutor,
                    session.connection,
                )
            if (!session.bound) {
                finish(session, SpeechRecognizer.ERROR_SERVER)
            } else {
                main.postDelayed(session.startTimeout, 30_000)
                startCapture(session)
            }
        } catch (_: SecurityException) {
            finish(session, SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS)
        } catch (_: Exception) {
            finish(session, SpeechRecognizer.ERROR_AUDIO)
        }
    }

    private fun closeSuppliedAudio(intent: Intent) {
        runCatching {
            intent
                .getParcelableExtra(
                    RecognizerIntent.EXTRA_AUDIO_SOURCE,
                    ParcelFileDescriptor::class.java,
                )
                ?.close()
        }
    }

    override fun onStopListening(listener: Callback) {
        current
            ?.takeIf { it.callback === listener }
            ?.let { session ->
                stopCapture(session)
                // A dead or stuck worker must never leave the client waiting forever.
                main.postDelayed(session.finishTimeout, 60_000)
            }
    }

    override fun onCancel(listener: Callback) {
        current?.takeIf { it.callback === listener }?.let { finish(it, null, cancelled = true) }
    }

    override fun onCheckRecognitionSupport(intent: Intent, callback: SupportCallback) {
        if (!supportsLanguage(intent)) {
            callback.onError(SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED)
        } else {
            callback.onSupportResult(
                RecognitionSupport.Builder()
                    .addInstalledOnDeviceLanguage("en-US")
                    .addSupportedOnDeviceLanguage("en-US")
                    .build()
            )
        }
    }

    private fun supportsLanguage(intent: Intent): Boolean {
        val language = intent.getStringExtra(RecognizerIntent.EXTRA_LANGUAGE) ?: "en"
        return language.equals("en", ignoreCase = true) ||
            language.startsWith("en-", ignoreCase = true)
    }

    private fun startCapture(session: Session) {
        if (current !== session || session.stopping) return
        val recorder = session.recorder
        val source = session.source
        val capture = checkNotNull(session.capture)
        try {
            recorder?.startRecording()
            if (recorder != null) {
                check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING)
            }
            session.callback.readyForSpeech(Bundle())
        } catch (_: Exception) {
            finish(session, SpeechRecognizer.ERROR_AUDIO)
            return
        }
        session.captureStarted = true
        Thread(
                {
                    val bytes = ByteArray(2048)
                    var meterReads = 0
                    var drainedAfterStop = false
                    fun write(count: Int) {
                        if (capture.append(bytes, count) && !session.stopping) {
                            session.endReason = ListeningLimits.OVERLOAD
                            session.stopping = true
                            main.post {
                                if (current === session) {
                                    runCatching { session.worker?.captureOverloaded() }
                                    main.postDelayed(session.finishTimeout, 60_000)
                                }
                            }
                        }
                        if (recorder != null && meterReads++ % 2 == 0) {
                            var power = 0.0
                            for (index in 0 until count - 1 step 2) {
                                val sample =
                                    ((bytes[index].toInt() and 255) or
                                            (bytes[index + 1].toInt() shl 8))
                                        .toShort() / 32768.0
                                power += sample * sample
                            }
                            val rms =
                                (20 * log10(sqrt(power / (count / 2)).coerceAtLeast(0.00001)))
                                    .toFloat()
                            main.post {
                                if (current === session && !session.stopping)
                                    runCatching { session.callback.rmsChanged(rms) }
                            }
                        }
                    }
                    fun drainBufferedAudio() {
                        if (recorder == null || drainedAfterStop) return
                        var remaining = recorder.bufferSizeInFrames * 2
                        while (remaining > 0 && !session.discardCapture) {
                            val count =
                                recorder.read(
                                    bytes,
                                    0,
                                    minOf(bytes.size, remaining),
                                    AudioRecord.READ_NON_BLOCKING,
                                )
                            if (count <= 0) break
                            write(count)
                            remaining -= count
                        }
                        if (session.stopping) drainedAfterStop = true
                    }
                    try {
                        while (!session.stopping) {
                            val count =
                                source?.read(bytes) ?: recorder?.read(bytes, 0, bytes.size) ?: break
                            if (count < 0) {
                                if (source == null && !session.stopping) {
                                    throw IllegalStateException("Audio recorder failed")
                                }
                                break
                            }
                            if (count > 0 && !session.discardCapture) {
                                write(count)
                            }
                            val request = session.requestedPhrase.getAndSet(0)
                            if (request != 0L && !session.stopping) {
                                drainBufferedAudio()
                                if (!session.stopping) capture.finishPhrase(request)
                            }
                        }
                        if (session.stopping && !session.discardCapture && recorder != null) {
                            // Finish owns the last read and any already captured microphone
                            // frames. Draining is bounded by the recorder's existing buffer;
                            // it does not wait for additional speech after the user's tap.
                            drainBufferedAudio()
                        }
                    } catch (_: Exception) {
                        if (!session.stopping)
                            main.post { finish(session, SpeechRecognizer.ERROR_AUDIO) }
                    } finally {
                        bytes.fill(0)
                        // Capture owns the final successful read and recorder drain. The feeder
                        // then owns EOF, even when Finish precedes model readiness.
                        runCatching { recorder?.stop() }
                        runCatching { recorder?.release() }
                        runCatching { source?.close() }
                        main.post {
                            session.recorder = null
                            session.source = null
                            if (current === session) {
                                stopCapture(session)
                                reportCaptureStopped(session)
                                main.postDelayed(session.finishTimeout, 60_000)
                            }
                        }
                        capture.finish()
                    }
                },
                "Microphone",
            )
            .start()
    }

    private fun feedWorker(session: Session) {
        main.removeCallbacks(session.startTimeout)
        val capture = checkNotNull(session.capture)
        val output = checkNotNull(session.output)
        val worker = checkNotNull(session.worker)
        capture.ready()
        Thread(
                {
                    val bytes = ByteArray(2048)
                    try {
                        while (!session.discardCapture) {
                            when (val next = capture.read(bytes) ?: break) {
                                is CaptureBuffer.Read.Audio -> output.write(bytes, 0, next.size)
                                // Install the watermark before writing any subsequent PCM.
                                is CaptureBuffer.Read.Boundary ->
                                    worker.finishPhrase(next.request, next.afterSamples)
                            }
                            bytes.fill(0)
                        }
                    } catch (_: Exception) {
                        // The worker may close its input at a normal endpoint. Its terminal
                        // callback owns the result; stop the microphone if it has not already.
                        main.post {
                            if (current === session) {
                                stopCapture(session, discard = true)
                                main.postDelayed(session.finishTimeout, 60_000)
                            }
                        }
                    } finally {
                        bytes.fill(0)
                        capture.cancel()
                        runCatching { output.close() }
                    }
                },
                "Captured audio",
            )
            .start()
    }

    private fun stopCapture(session: Session, discard: Boolean = false) {
        if (discard) {
            session.discardCapture = true
            session.capture?.cancel()
            // Cancellation may interrupt a blocked read/write; completed Finish must drain it.
            runCatching { session.recorder?.stop() }
            runCatching { session.source?.close() }
            runCatching { session.output?.close() }
        }
        if (session.stopping) return
        session.stopping = true
        if (!session.captureStarted) {
            runCatching { session.recorder?.release() }
            runCatching { session.source?.close() }
            runCatching { session.output?.close() }
            session.recorder = null
            session.source = null
            session.output = null
            session.capture?.finish()
            reportCaptureStopped(session)
        }
    }

    private fun reportCaptureStopped(session: Session) {
        if (session.reportedCaptureStop) return
        session.reportedCaptureStop = true
        runCatching { session.callback.endOfSpeech() }
    }

    private fun finish(session: Session, error: Int?, cancelled: Boolean = false) {
        if (current !== session) return
        current = null
        session.controlToken?.let(PhraseCommands::unregister)
        main.removeCallbacks(session.startTimeout)
        main.removeCallbacks(session.finishTimeout)
        stopCapture(session, discard = true)
        runCatching { session.worker?.cancel() }
        runCatching { session.readPipe?.close() }
        if (session.bound) {
            session.bound = false
            unbindService(session.connection)
        }
        session.priority.close()
        if (cancelled) return
        runCatching {
            when {
                error != null -> session.callback.error(error)
                session.text.isBlank() ->
                    session.callback.error(
                        if (session.endReason == ListeningLimits.NO_SPEECH)
                            SpeechRecognizer.ERROR_SPEECH_TIMEOUT
                        else SpeechRecognizer.ERROR_NO_MATCH
                    )
                else ->
                    session.callback.results(
                        resultBundle(session.text, endReason = session.endReason)
                    )
            }
        }
    }

    private fun resultBundle(
        text: String,
        phraseEnd: Boolean = false,
        confirmedLength: Int = text.length,
        endReason: Int = 0,
    ) =
        Bundle().apply {
            putStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION, arrayListOf(text))
            putBoolean(EXTRA_PHRASE_END, phraseEnd)
            putInt(EXTRA_CONFIRMED_LENGTH, confirmedLength)
            putInt(EXTRA_END_REASON, endReason)
        }

    override fun onDestroy() {
        current?.let { finish(it, null, cancelled = true) }
        super.onDestroy()
    }

    private inner class Session(val callback: Callback, val partials: Boolean) {
        val priority = DictationPriority.acquire()
        @Volatile var stopping = false
        @Volatile var discardCapture = false
        var captureStarted = false
        var reportedCaptureStop = false
        var realtime = false
        var capture: CaptureBuffer? = null

        @Volatile var recorder: AudioRecord? = null

        @Volatile var source: InputStream? = null

        @Volatile var output: OutputStream? = null
        var readPipe: ParcelFileDescriptor? = null
        var worker: ITranscriptionWorker? = null
        var bound = false
        var stopOnSilence = true
        var previews = true
        var pauseMillis = 3000
        var text = ""
        @Volatile var endReason = ListeningLimits.COMPLETE
        var controlToken: String? = null
        val requestedPhrase = AtomicLong()
        var beganSpeech = false
        val startTimeout = Runnable { finish(this, SpeechRecognizer.ERROR_SERVER) }
        val finishTimeout = Runnable { finish(this, SpeechRecognizer.ERROR_SERVER) }

        val events =
            object : ITranscriptionEvents.Stub() {
                private fun deliver(block: () -> Unit) {
                    main.post {
                        if (current === this@Session) {
                            try {
                                block()
                            } catch (_: Exception) {
                                finish(this@Session, SpeechRecognizer.ERROR_SERVER)
                            }
                        }
                    }
                }

                override fun ready() = deliver { feedWorker(this@Session) }

                override fun speech(speaking: Boolean) = deliver {
                    if (speaking && !beganSpeech) {
                        beganSpeech = true
                        callback.beginningOfSpeech()
                    }
                }

                override fun level(rms: Float) = deliver {
                    if (!realtime && rms.isFinite()) callback.rmsChanged(rms)
                }

                override fun progress(completedMs: Long, durationMs: Long) = Unit

                override fun inputEnded(reason: Int) = deliver {
                    if (endReason == ListeningLimits.COMPLETE) endReason = reason
                    stopCapture(this@Session)
                    main.postDelayed(finishTimeout, 60_000)
                }

                override fun segment(segment: String, phraseEnd: Boolean) = deliver {
                    require(segment.length <= 16000)
                    text = Transcript.join(text, segment)
                    if (partials)
                        callback.partialResults(
                            resultBundle(
                                text,
                                phraseEnd && endReason == ListeningLimits.COMPLETE,
                                endReason = endReason,
                            )
                        )
                }

                override fun hypothesis(hypothesis: String) = deliver {
                    require(hypothesis.length <= 16000)
                    if (partials)
                        callback.partialResults(
                            resultBundle(
                                Transcript.join(text, hypothesis),
                                confirmedLength = text.length,
                                endReason = endReason,
                            )
                        )
                }

                override fun phraseFinished(request: Long) = deliver {
                    if (partials)
                        callback.partialResults(
                            resultBundle(text, endReason = endReason).apply {
                                putLong(EXTRA_FINISHED_PHRASE, request)
                            }
                        )
                }

                override fun finished() = deliver { finish(this@Session, null) }

                override fun failed(reason: Int) = deliver {
                    val valid =
                        setOf(
                            SpeechRecognizer.ERROR_AUDIO,
                            SpeechRecognizer.ERROR_SERVER,
                            SpeechRecognizer.ERROR_RECOGNIZER_BUSY,
                            SpeechRecognizer.ERROR_SPEECH_TIMEOUT,
                        )
                    finish(
                        this@Session,
                        if (reason in valid) reason else SpeechRecognizer.ERROR_SERVER,
                    )
                }
            }

        val connection =
            object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                    if (current !== this@Session) return
                    worker = ITranscriptionWorker.Stub.asInterface(binder)
                    try {
                        val pipe = checkNotNull(readPipe)
                        worker?.start(
                            pipe,
                            stopOnSilence,
                            realtime,
                            true,
                            previews,
                            pauseMillis,
                            events,
                        )
                        if (endReason == ListeningLimits.OVERLOAD) worker?.captureOverloaded()
                        pipe.close()
                        readPipe = null
                    } catch (_: Exception) {
                        finish(this@Session, SpeechRecognizer.ERROR_SERVER)
                    }
                }

                override fun onServiceDisconnected(name: ComponentName) =
                    finish(this@Session, SpeechRecognizer.ERROR_SERVER)

                override fun onBindingDied(name: ComponentName) =
                    finish(this@Session, SpeechRecognizer.ERROR_SERVER)

                override fun onNullBinding(name: ComponentName) =
                    finish(this@Session, SpeechRecognizer.ERROR_SERVER)
            }
    }

    companion object {
        internal const val EXTRA_CONTROL_TOKEN = "dev.soupslurpr.transcribro.CONTROL_TOKEN"
        internal const val EXTRA_FINISHED_PHRASE = "dev.soupslurpr.transcribro.FINISHED_PHRASE"
        internal const val EXTRA_LIVE_TEXT = "dev.soupslurpr.transcribro.LIVE_TEXT"
        const val EXTRA_END_REASON = "dev.soupslurpr.transcribro.END_REASON"
        const val EXTRA_CONTINUOUS = "dev.soupslurpr.transcribro.CONTINUOUS"
        // Retain the published production opt-in for clients requesting successive phrases.
        const val EXTRA_AUTO_STOP = "dev.soupslurpr.transcribro.EXTRA_AUTO_STOP"
        const val EXTRA_PHRASE_END = "dev.soupslurpr.transcribro.PHRASE_END"
        const val EXTRA_CONFIRMED_LENGTH = "dev.soupslurpr.transcribro.CONFIRMED_LENGTH"
    }
}
