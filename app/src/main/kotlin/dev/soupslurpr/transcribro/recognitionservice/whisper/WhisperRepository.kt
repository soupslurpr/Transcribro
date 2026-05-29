package dev.soupslurpr.transcribro.recognitionservice.whisper

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class WhisperRepository(
    private val whisperLocalDataSource: WhisperLocalDataSource
) {

    private var whisperContext: MutableState<WhisperContext?> =
        mutableStateOf(null)

    /** Asset path of the model currently loaded into [whisperContext], if any. */
    private var loadedModelSpec: String? = null

    /** Serializes (re)loading and releasing the native context. */
    private val loadMutex = Mutex()

    /**
     * Ensures [whisperContext] holds the model identified by [modelSpec], (re)loading it if no model
     * is loaded yet or a different model was previously loaded. Switching models releases the
     * previous native context first.
     */
    private suspend fun ensureWhisperContext(modelSpec: String) {
        if (whisperContext.value != null && loadedModelSpec == modelSpec) {
            return
        }
        loadMutex.withLock {
            if (whisperContext.value == null || loadedModelSpec != modelSpec) {
                whisperContext.value?.release()
                whisperContext.value = null
                loadedModelSpec = null

                whisperContext.value = whisperLocalDataSource.getWhisperContext(modelSpec)
                loadedModelSpec = modelSpec
            }
        }
    }

    suspend fun transcribeAudio(
        data: ShortArray,
        language: String = WhisperLanguage.AUTO,
        modelSpec: String = WhisperModel.DEFAULT.assetPath
    ): String {
        ensureWhisperContext(modelSpec)
        // assume we only have one channel
        var buffer = FloatArray(data.size) { index ->
            (data[index] / 32767.0f).coerceIn(-1f..1f)
        }

        if (data.size < 32000) {
            val newBuffer = FloatArray(32000)

            for ((i, value) in buffer.withIndex()) {
                newBuffer[i] = value
            }

            newBuffer.fill(0f, data.size, newBuffer.size)

            buffer = newBuffer
        }

        val transcript =
            whisperContext.value?.transcribeData(buffer, ((data.size / 16000f) * 1000f).toLong(), language)
                ?: ""
        return transcript.removeSuffix(" .") // remove hallucination
    }

    suspend fun release() {
        loadMutex.withLock {
            whisperContext.value?.release()
            whisperContext.value = null
            loadedModelSpec = null
        }
    }
}
