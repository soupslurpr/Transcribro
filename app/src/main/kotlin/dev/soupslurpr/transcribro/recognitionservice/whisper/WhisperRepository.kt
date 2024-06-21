package dev.soupslurpr.transcribro.recognitionservice.whisper

import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import com.whispercpp.whisper.WhisperContext

class WhisperRepository(
    private val whisperLocalDataSource: WhisperLocalDataSource
) {
    private val TAG = "WhisperRepository"

    private var whisperContext: MutableState<WhisperContext?> =
        mutableStateOf(null)

    private suspend fun loadWhisperContextIfNull() {
        Log.d(TAG, "Checking if whisperContext is null")
        if (whisperContext.value == null) {
            Log.d(TAG, "whisperContext is null, loading from local data source")
            whisperContext.value = whisperLocalDataSource.getWhisperContext()
            Log.d(TAG, "whisperContext loaded: $whisperContext")
        } else {
            Log.d(TAG, "whisperContext is already initialized")
        }
    }

    suspend fun transcribeAudio(data: ShortArray): String {
        Log.d(TAG, "transcribeAudio called with data of size: ${data.size}")
        loadWhisperContextIfNull()

        var buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }
        Log.d(TAG, "Audio data normalized to float array")

        if (data.size < 32000) {
            Log.d(TAG, "Data size is less than 32000, resizing buffer")
            val newBuffer = FloatArray(32000)
            for ((i, value) in buffer.withIndex()) {
                newBuffer[i] = value
            }
            newBuffer.fill(0f, data.size, newBuffer.size)
            buffer = newBuffer
            Log.d(TAG, "Buffer resized to 32000 with zero-padding")
        }

        val transcript = whisperContext.value?.transcribeData(buffer, ((data.size / 16000f) * 1000f).toLong()) ?: ""
        Log.d(TAG, "Transcription completed: $transcript")
        return transcript.removeSuffix(" .").also {
            Log.d(TAG, "Transcription after removing suffix: $it")
        }
    }

    suspend fun release() {
        Log.d(TAG, "Releasing whisperContext")
        whisperContext.value?.release()
        Log.d(TAG, "whisperContext released")
    }
}
