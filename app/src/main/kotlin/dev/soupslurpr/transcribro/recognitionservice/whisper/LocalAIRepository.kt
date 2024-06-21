package dev.soupslurpr.transcribro.recognitionservice.whisper

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

class LocalAIRepository(
    private val localAIDataSource: LocalAIDataSourceApi
) {

    companion object {
        private const val TAG = "LocalAIRepository"
        private const val SAMPLE_RATE = 16000
    }

    suspend fun transcribeAudio(data: ShortArray): String {
        Log.d(TAG, "transcribeAudio called with data of size: ${data.size}")
        val byteArray = prepareAudioFile(data)
        Log.d(TAG, "Prepared audio file byte array of size: ${byteArray.size}")
        return withContext(Dispatchers.IO) {
            val result = localAIDataSource.transcribeAudio(byteArray, "ggml-faster-whisper-large-v3")
            Log.d(TAG, "Transcription result received")
            result
        }
    }

    private fun prepareAudioFile(data: ShortArray): ByteArray {
        Log.d(TAG, "Preparing audio file from short array")
        val byteArray = convertShortArrayToByteArray(data)
        val wavFile = createWavFile(byteArray)
        Log.d(TAG, "Audio file preparation complete. File path: ${wavFile.absolutePath}")
        return wavFile.readBytes()
    }

    private fun convertShortArrayToByteArray(data: ShortArray): ByteArray {
        Log.d(TAG, "Converting short array to byte array")
        val byteBuffer = ByteArrayOutputStream()
        for (value in data) {
            byteBuffer.write(value.toInt() and 0xFF)
            byteBuffer.write((value.toInt() shr 8) and 0xFF)
        }
        val byteArray = byteBuffer.toByteArray()
        Log.d(TAG, "Conversion complete. Byte array size: ${byteArray.size}")
        return byteArray
    }

    private fun createWavFile(audioData: ByteArray): File {
        val file = File.createTempFile("audio", ".wav")
        val out = FileOutputStream(file)
        val header = createWavHeader(audioData.size, SAMPLE_RATE.toLong(), 1, 16)
        out.write(header)
        out.write(audioData)
        out.close()
        return file
    }

    private fun createWavHeader(dataSize: Int, sampleRate: Long, channels: Int, bitsPerSample: Int): ByteArray {
        val totalDataLen = dataSize + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8
        return ByteArray(44).apply {
            // RIFF/WAVE header
            System.arraycopy("RIFF".toByteArray(), 0, this, 0, 4)
            System.arraycopy(intToByteArray(totalDataLen), 0, this, 4, 4)
            System.arraycopy("WAVE".toByteArray(), 0, this, 8, 4)
            System.arraycopy("fmt ".toByteArray(), 0, this, 12, 4)
            System.arraycopy(intToByteArray(16), 0, this, 16, 4) // Sub-chunk size
            System.arraycopy(shortToByteArray(1), 0, this, 20, 2) // AudioFormat (1 = PCM)
            System.arraycopy(shortToByteArray(channels), 0, this, 22, 2)
            System.arraycopy(intToByteArray(sampleRate.toInt()), 0, this, 24, 4)
            System.arraycopy(intToByteArray(byteRate.toInt()), 0, this, 28, 4)
            System.arraycopy(shortToByteArray((channels * bitsPerSample / 8)), 0, this, 32, 2)
            System.arraycopy(shortToByteArray(bitsPerSample), 0, this, 34, 2)
            System.arraycopy("data".toByteArray(), 0, this, 36, 4)
            System.arraycopy(intToByteArray(dataSize), 0, this, 40, 4)
        }
    }

    private fun intToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            (value shr 8 and 0xFF).toByte(),
            (value shr 16 and 0xFF).toByte(),
            (value shr 24 and 0xFF).toByte()
        )
    }

    private fun shortToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            (value shr 8 and 0xFF).toByte()
        )
    }

    suspend fun release() {
        Log.d(TAG, "release called")
        // No specific release logic for the API client
    }
}
