package dev.soupslurpr.transcribro.ui.action_recognize_speech

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import dev.soupslurpr.transcribro.dictation.DictationController
import kotlinx.coroutines.flow.MutableStateFlow

class SpeechViewModel(application: Application) : AndroidViewModel(application) {
    val controller = DictationController(application)
    var started = false
    var pendingResultSent = false
    val result = MutableStateFlow<String?>(null)

    init {
        controller.onFinished = { result.value = it }
    }

    override fun onCleared() = controller.close()
}
