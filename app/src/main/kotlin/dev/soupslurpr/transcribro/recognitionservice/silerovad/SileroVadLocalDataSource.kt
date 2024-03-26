package dev.soupslurpr.transcribro.recognitionservice.silerovad

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext

class SileroVadLocalDataSource(
    private val sileroVadApi: SileroVadApi,
    private val ioDispatcher: CoroutineDispatcher,
) {
    suspend fun getSileroVadDetector(): SileroVadDetector {
        return withContext(ioDispatcher) {
            sileroVadApi.getSileroVadDetector()
        }
    }
}

interface SileroVadApi {
    fun getSileroVadDetector(): SileroVadDetector
}