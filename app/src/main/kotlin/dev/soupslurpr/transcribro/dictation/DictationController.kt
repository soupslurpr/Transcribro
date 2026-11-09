package dev.soupslurpr.transcribro.dictation

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import dev.soupslurpr.transcribro.engine.ListeningLimits
import dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class DictationPhase {
    Ready,
    Preparing,
    Listening,
    Finishing,
}

data class DictationState(
    val phase: DictationPhase = DictationPhase.Ready,
    val level: Float = 0f,
    val message: String? = null,
    val failure: Boolean = false,
    val finishingPhrase: Boolean = false,
    val listening: Boolean = false,
) {
    val active: Boolean
        get() = phase != DictationPhase.Ready
}

/** Main-thread owner. Every run has its own listener and generation; stale results are ignored. */
class DictationController(private val context: Context) : AutoCloseable {
    private val mutableState = MutableStateFlow(DictationState())
    val state = mutableState.asStateFlow()
    private var recognizer: SpeechRecognizer? = null
    private var generation = 0
    private var delivered = ""
    private var controlToken = ""
    private var phraseRequest = 0L
    private var afterPhrase: (() -> Unit)? = null
    var onText: (String) -> Unit = {}
    var onFinished: (String) -> Unit = {}
    var onPhraseEnd: () -> Unit = {}
    var onHypothesis: (String) -> Unit = {}

    fun start(continuous: Boolean = true, liveText: Boolean = true, pauseMillis: Int = 3000) {
        if (state.value.active) return
        cancel()
        val run = ++generation
        delivered = ""
        controlToken = UUID.randomUUID().toString()
        mutableState.value = DictationState(phase = DictationPhase.Preparing)
        val speech =
            SpeechRecognizer.createSpeechRecognizer(
                context,
                ComponentName(context, MainRecognitionService::class.java),
            )
        recognizer = speech
        speech.setRecognitionListener(
            object : RecognitionListener {
                private fun current() = generation == run

                override fun onReadyForSpeech(params: Bundle?) {
                    if (current() && state.value.phase != DictationPhase.Finishing) {
                        mutableState.value =
                            state.value.copy(phase = DictationPhase.Listening, listening = true)
                        if (afterPhrase != null) requestPhrase()
                    }
                }

                override fun onBeginningOfSpeech() = Unit

                override fun onRmsChanged(rmsdB: Float) {
                    if (current()) {
                        mutableState.value =
                            state.value.copy(level = ((rmsdB + 55f) / 45f).coerceIn(0f, 1f))
                    }
                }

                override fun onBufferReceived(buffer: ByteArray?) = Unit

                override fun onEndOfSpeech() {
                    if (current()) {
                        mutableState.value =
                            state.value.copy(
                                phase = DictationPhase.Finishing,
                                level = 0f,
                                listening = false,
                            )
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    if (current()) deliver(partialResults)
                    if (
                        current() &&
                            afterPhrase != null &&
                            partialResults?.getLong(MainRecognitionService.EXTRA_FINISHED_PHRASE) ==
                                phraseRequest
                    ) {
                        val action = afterPhrase
                        afterPhrase = null
                        mutableState.value = state.value.copy(finishingPhrase = false)
                        if (partialResults.getInt(MainRecognitionService.EXTRA_END_REASON) == 0)
                            action?.invoke()
                    }
                    if (
                        current() &&
                            afterPhrase == null &&
                            partialResults?.getInt(MainRecognitionService.EXTRA_END_REASON, 0) ==
                                0 &&
                            partialResults?.getBoolean(MainRecognitionService.EXTRA_PHRASE_END) ==
                                true
                    )
                        onPhraseEnd()
                }

                override fun onResults(results: Bundle?) {
                    if (!current()) return
                    deliver(results)
                    if (!current()) return
                    val text = delivered
                    val reason = results?.getInt(MainRecognitionService.EXTRA_END_REASON) ?: 0
                    val action = afterPhrase.takeIf { reason == ListeningLimits.COMPLETE }
                    val message =
                        when (reason) {
                            ListeningLimits.NO_SPEECH ->
                                "Stopped after 12 seconds without detected speech. Your words are kept; tap to continue."
                            ListeningLimits.INACTIVITY ->
                                "Stopped after a minute without speech. Your completed words are kept."
                            ListeningLimits.DURATION ->
                                "Five minutes reached. Your words are kept; tap to continue."
                            ListeningLimits.OVERLOAD ->
                                "Paused to catch up. Your captured words are kept; tap to continue."
                            else -> null
                        }
                    mutableState.value =
                        state.value.copy(
                            phase = DictationPhase.Ready,
                            level = 0f,
                            message = message,
                            finishingPhrase = false,
                            listening = false,
                        )
                    release()
                    // A protective timeout is an interruption, not a request to send or switch.
                    if (action != null) action()
                    else if (reason == ListeningLimits.COMPLETE) onFinished(text)
                }

                override fun onError(error: Int) {
                    if (!current()) return
                    mutableState.value =
                        state.value.copy(
                            phase = DictationPhase.Ready,
                            level = 0f,
                            message = errorMessage(error),
                            failure =
                                error != SpeechRecognizer.ERROR_NO_MATCH &&
                                    error != SpeechRecognizer.ERROR_SPEECH_TIMEOUT,
                            finishingPhrase = false,
                            listening = false,
                        )
                    release()
                    onHypothesis("")
                }

                override fun onEvent(eventType: Int, params: Bundle?) = Unit
            }
        )
        speech.startListening(
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(
                    RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
                )
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                .putExtra(MainRecognitionService.EXTRA_CONTINUOUS, continuous)
                .putExtra(MainRecognitionService.EXTRA_CONTROL_TOKEN, controlToken)
                .putExtra(MainRecognitionService.EXTRA_LIVE_TEXT, liveText)
                .putExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
                    pauseMillis.toLong(),
                )
        )
    }

    /**
     * Finalize in capture order, perform a deliberate edit, and continue the same microphone run.
     */
    fun finishPhrase(action: () -> Unit) {
        if (!state.value.active) {
            action()
            return
        }
        if (afterPhrase != null) return
        afterPhrase = action
        phraseRequest++
        mutableState.value = state.value.copy(finishingPhrase = true)
        if (state.value.phase == DictationPhase.Listening) requestPhrase()
        // Preparing requests wait for ready; an already finishing run uses onResults.
    }

    private fun requestPhrase() {
        if (!PhraseCommands.finish(controlToken, phraseRequest)) {
            interrupt("Couldn't finish this phrase. Your completed words are kept; try again.")
        }
    }

    private fun deliver(bundle: Bundle?) {
        val text =
            bundle?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                ?: return
        val confirmedLength =
            bundle.getInt(MainRecognitionService.EXTRA_CONFIRMED_LENGTH, text.length)
        if (confirmedLength !in 0..text.length) return
        val confirmed = text.take(confirmedLength)
        if (!confirmed.startsWith(delivered)) return
        val addition = confirmed.substring(delivered.length).trimStart()
        delivered = confirmed
        val run = generation
        if (addition.isNotEmpty()) onText(addition)
        if (generation == run) onHypothesis(text.substring(confirmedLength).trimStart())
    }

    fun stop() {
        if (state.value.active && state.value.phase != DictationPhase.Finishing) {
            mutableState.value = state.value.copy(phase = DictationPhase.Finishing, level = 0f)
            recognizer?.stopListening()
        }
    }

    fun cancel() {
        generation++
        recognizer?.cancel()
        release()
        mutableState.value =
            state.value.copy(
                phase = DictationPhase.Ready,
                level = 0f,
                finishingPhrase = false,
                listening = false,
            )
        onHypothesis("")
    }

    fun clearMessage() {
        mutableState.value = state.value.copy(message = null, failure = false)
    }

    fun interrupt(message: String) {
        cancel()
        mutableState.value = state.value.copy(message = message, failure = true)
    }

    private fun release() {
        generation++
        recognizer?.destroy()
        recognizer = null
        delivered = ""
        afterPhrase = null
    }

    override fun close() = cancel()

    companion object {
        fun errorMessage(error: Int): String =
            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS ->
                    "Allow microphone access to dictate."
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT ->
                    "No speech detected. Tap Start to try again."
                SpeechRecognizer.ERROR_NO_MATCH -> "No words recognized. Tap Start to try again."
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY ->
                    "Recognition is busy. Your completed text is kept; try again shortly."
                SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ->
                    "English speech is supported in this build."
                SpeechRecognizer.ERROR_AUDIO ->
                    "Couldn't read the audio. Check microphone access and try again."
                else -> "Recognition stopped. Your completed text is kept; tap to try again."
            }
    }
}
