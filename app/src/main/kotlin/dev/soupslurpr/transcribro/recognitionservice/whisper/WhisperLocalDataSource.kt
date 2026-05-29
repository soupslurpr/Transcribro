package dev.soupslurpr.transcribro.recognitionservice.whisper

import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

class WhisperLocalDataSource(
    private val whisperApi: WhisperApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getWhisperContext(modelSpec: String): WhisperContext {
        return withContext(ioDispatcher) {
            whisperApi.getWhisperContext(modelSpec)
        }
    }
}

interface WhisperApi {
    /**
     * Creates a context for the given model. [modelSpec] is either a bundled asset path
     * (e.g. "models/whisper/ggml-base-q8_0.bin") or the absolute file path of a model
     * imported from the file system.
     */
    fun getWhisperContext(modelSpec: String): WhisperContext
}
