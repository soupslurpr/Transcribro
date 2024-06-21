package dev.soupslurpr.transcribro.recognitionservice.whisper

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import org.json.JSONObject

class LocalAIDataSource(
    private val apiUrl: String,
    private val ioDispatcher: CoroutineDispatcher,
    private val okHttpClient: OkHttpClient
) : LocalAIDataSourceApi {

    companion object {
        private const val TAG = "LocalAIDataSource"
    }

    override suspend fun transcribeAudio(data: ByteArray, model: String): String {
        Log.d(TAG, "Starting transcription request")
        return withContext(ioDispatcher) {
            Log.d(TAG, "Preparing request body and multipart form data")
            val requestBody = RequestBody.create("audio/wav".toMediaTypeOrNull(), data)
            val multipartBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "audio.wav", requestBody)
                .addFormDataPart("model", model)
                .build()

            val request = Request.Builder()
                .url("$apiUrl/v1/audio/transcriptions")
                .post(multipartBody)
                .build()

            Log.d(TAG, "Sending request to $apiUrl/v1/audio/transcriptions")
            okHttpClient.newCall(request).execute().use { response ->
                Log.d(TAG, "Received response with status code: ${response.code}")
                handleResponse(response)
            }
        }
    }

    private fun handleResponse(response: Response): String {
        if (!response.isSuccessful) {
            Log.e(TAG, "Failed to transcribe audio: ${response.message}")
            throw RuntimeException("Failed to transcribe audio: ${response.message}")
        }
        val responseBody = response.body?.string()
        Log.d(TAG, "Response Body: $responseBody")
        if (responseBody == null) {
            Log.e(TAG, "Empty response body")
            throw RuntimeException("Empty response body")
        }
        val jsonResponse = JSONObject(responseBody)
        val transcribedText = jsonResponse.optString("text", "")
        if (transcribedText.isEmpty()) {
            Log.e(TAG, "No 'text' field in the response body")

            throw RuntimeException("No 'text' field in the response body")
        }
        Log.d(TAG, "Successfully transcribed audio: $transcribedText")
        return transcribedText
    }
}

interface LocalAIDataSourceApi {
    suspend fun transcribeAudio(data: ByteArray, model: String): String
}
