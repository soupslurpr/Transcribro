package dev.soupslurpr.transcribro.ui.action_recognize_speech

import android.app.Application
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.media.MediaPlayer
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import dev.soupslurpr.transcribro.R
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class SpeechRecognizerViewModel(application: Application) : AndroidViewModel(application) {

    class SpeechRecognizerUiState {
        val speechRecognizer: MutableState<SpeechRecognizer?> = mutableStateOf(null)
        var isSpeaking by mutableStateOf(false)
        var isRecognizing by mutableStateOf(false)
        var showInsufficientPermissionsError by mutableStateOf(false)
        var showRecognizerBusyOrClientError by mutableStateOf(false)
    }

    private val _uiState = MutableStateFlow(SpeechRecognizerUiState())
    val uiState: StateFlow<SpeechRecognizerUiState> = _uiState.asStateFlow()

    open class MainSpeechRecognitionListener(
        application: Application,
        private val speechRecognizerViewModel: SpeechRecognizerViewModel,
    ) : RecognitionListener {

        private val audioManager = application.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        private val startedRecognitionMediaPlayer: MediaPlayer =
            MediaPlayer.create(application, R.raw.started_recognition)
        private val stoppedRecognitionMediaPlayer: MediaPlayer =
            MediaPlayer.create(application, R.raw.stopped_recognition)

        override fun onReadyForSpeech(params: Bundle?) {
            speechRecognizerViewModel.setIsRecognizing(true)

            if (audioManager.ringerMode == AudioManager.RINGER_MODE_NORMAL) {
                startedRecognitionMediaPlayer.start()
            }
        }

        override fun onBeginningOfSpeech() {
            speechRecognizerViewModel.setIsSpeaking(true)
        }

        override fun onRmsChanged(rmsdB: Float) {
//                TODO("Not yet implemented")
        }

        override fun onBufferReceived(buffer: ByteArray?) {
//                TODO("Not yet implemented")
        }

        override fun onEndOfSpeech() {
            speechRecognizerViewModel.setIsSpeaking(false)
        }

        override fun onError(error: Int) {
            when (error) {
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                    speechRecognizerViewModel.setShowInsufficientPermissionsError(true)
                }

                SpeechRecognizer.ERROR_RECOGNIZER_BUSY, SpeechRecognizer.ERROR_CLIENT -> {
                    speechRecognizerViewModel.setShowRecognizerBusyOrClientError(true)
                }

                else -> {
                }
            }
        }

        override fun onResults(results: Bundle?) {
            speechRecognizerViewModel.setIsRecognizing(false)

            if (audioManager.ringerMode == AudioManager.RINGER_MODE_NORMAL) {
                stoppedRecognitionMediaPlayer.start()
            }
        }

        override fun onPartialResults(partialResults: Bundle?) {
        }

        override fun onEvent(eventType: Int, params: Bundle?) {
//            TODO("Not yet implemented")
        }
    }

    fun setSpeechRecognizer(speechRecognizer: SpeechRecognizer) {
        _uiState.value.speechRecognizer.value = speechRecognizer
    }

    fun setRecognitionListener(recognitionListener: MainSpeechRecognitionListener) {
        _uiState.value.speechRecognizer.value?.setRecognitionListener(recognitionListener)
    }

    fun setIsRecognizing(value: Boolean) {
        _uiState.value.isRecognizing = value
    }

    fun setIsSpeaking(value: Boolean) {
        _uiState.value.isSpeaking = value
    }

    fun setShowInsufficientPermissionsError(value: Boolean) {
        _uiState.value.showInsufficientPermissionsError = value
    }

    fun setShowRecognizerBusyOrClientError(value: Boolean) {
        _uiState.value.showRecognizerBusyOrClientError = value
    }

    fun startListening(recognizerIntent: Intent) {
        uiState.value.speechRecognizer.value?.startListening(recognizerIntent)
    }

    fun stopListening() {
        uiState.value.speechRecognizer.value?.stopListening()
    }

    override fun onCleared() {
        super.onCleared()

        _uiState.value.speechRecognizer.value?.destroy()
    }
}