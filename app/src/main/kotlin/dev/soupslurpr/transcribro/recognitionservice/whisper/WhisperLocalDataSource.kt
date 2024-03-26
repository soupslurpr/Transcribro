package dev.soupslurpr.transcribro.recognitionservice.whisper

import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

class WhisperLocalDataSource(
    private val whisperApi: WhisperApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getWhisperContext(): WhisperContext {
        return withContext(ioDispatcher) {
            whisperApi.getWhisperContext()
        }
    }
}

interface WhisperApi {
    fun getWhisperContext(): WhisperContext
}