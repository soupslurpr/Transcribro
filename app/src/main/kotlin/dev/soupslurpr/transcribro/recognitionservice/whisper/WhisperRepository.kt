package dev.soupslurpr.transcribro.recognitionservice.whisper

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import com.whispercpp.whisper.WhisperContext

class WhisperRepository(
    private val whisperLocalDataSource: WhisperLocalDataSource
) {

    private var whisperContext: MutableState<WhisperContext?> =
        mutableStateOf(null)

    private suspend fun loadWhisperContextIfNull() {
        if (whisperContext.value == null) {
            whisperContext.value = whisperLocalDataSource.getWhisperContext()
        }
    }

    suspend fun transcribeAudio(data: ShortArray): String {
        loadWhisperContextIfNull()
        // assume we only have one channel
        val buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }
        return whisperContext.value?.transcribeData(buffer, ((data.size / 16000f) * 1000f).toLong()) ?: ""
    }

    suspend fun release() {
        whisperContext.value?.release()
    }
}