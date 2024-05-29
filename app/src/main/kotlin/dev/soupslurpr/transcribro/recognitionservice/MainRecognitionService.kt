package dev.soupslurpr.transcribro.recognitionservice

import android.content.ContextParams
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.RemoteException
import android.speech.RecognitionService
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.whispercpp.whisper.WhisperContext
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadApi
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadDetector
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.silerovad.SileroVadRepository
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperApi
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperLocalDataSource
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.lang.System.currentTimeMillis
import java.util.concurrent.atomic.AtomicBoolean

private data class Transcription(
    val audioData: MutableList<Short> = mutableListOf(),
    var start: Double?,
    var end: Double?,
    var text: String?,
)

class MainRecognitionService : RecognitionService() {
    companion object {
        const val EXTRA_AUTO_STOP = "dev.soupslurpr.transcribro.EXTRA_AUTO_STOP"
    }

    private val recordAndTranscribeScope = CoroutineScope(Dispatchers.IO)

    private val transcribeScope = CoroutineScope(Dispatchers.IO)

    private val isRecording = AtomicBoolean(true)

    private var recordAndTranscribeJob: Job? = null

//    private var loadedModel: WhichModel? = null

    private var isSpeaking by mutableStateOf(false)

    private var stopListening by mutableStateOf(false)

    private val transcribeJobs = mutableListOf<Job>()

    private val whisperRepository: WhisperRepository =
        WhisperRepository(
            WhisperLocalDataSource(
                whisperApi =
                object : WhisperApi {
                    override fun getWhisperContext(): WhisperContext {
                        return WhisperContext.createContextFromAsset(
                            application.assets,
                            "models/whisper/ggml-model-whisper-tiny.en-q8_0.bin"
                        )
                    }
                },
                ioDispatcher = Dispatchers.IO,
            )
        )

    private val sileroVadRepository = SileroVadRepository(
        SileroVadLocalDataSource(
            object : SileroVadApi {
                override fun getSileroVadDetector(): SileroVadDetector {
                    val SAMPLE_RATE = 16000
                    val START_THRESHOLD = 0.6f
                    val END_THRESHOLD = 0.45f
                    val MIN_SILENCE_DURATION_MS = 3000
                    val SPEECH_PAD_MS = 0

                    val model =
                        this@MainRecognitionService.assets.open("models/silero_vad/silero_vad.with_runtime_opt.ort")

                    val modelBytes = model.readBytes()

                    model.close()

                    return SileroVadDetector(
                        modelBytes,
                        START_THRESHOLD,
                        END_THRESHOLD,
                        SAMPLE_RATE,
                        MIN_SILENCE_DURATION_MS,
                        SPEECH_PAD_MS
                    )
                }
            },
            Dispatchers.IO
        )
    )

    override fun onStartListening(recognizerIntent: Intent?, listener: Callback?) {
        val autoStopRecognition = recognizerIntent?.extras?.getBoolean(EXTRA_AUTO_STOP) ?: true
        val isPartialResults = recognizerIntent?.extras?.getBoolean(RecognizerIntent.EXTRA_PARTIAL_RESULTS)
        val speechStartPadMs = 24000

        if (recordAndTranscribeJob?.isActive == true) {
            listener?.error(SpeechRecognizer.ERROR_RECOGNIZER_BUSY)
        }

        val attributionContext = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            this.createContext(
                ContextParams.Builder()
                    .setNextAttributionSource(listener?.callingAttributionSource)
                    .build()
            )
        } else {
            null
        }

        val sampleRate = 16000
        val encoding = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding)

        val audioFormat = AudioFormat.Builder()
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .setSampleRate(sampleRate)
            .setEncoding(encoding)
            .build()

        val audioRecord = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && attributionContext != null) {
            try {
                AudioRecord.Builder()
                    .setContext(attributionContext)
                    .setAudioFormat(audioFormat)
                    .build()
            } catch (e: SecurityException) {
                throw SecurityException(e)
            }
        } else {
            try {
                AudioRecord.Builder()
                    .setAudioFormat(audioFormat)
                    .build()
            } catch (e: SecurityException) {
                throw SecurityException(e)
            }
        }


//        val model = when (recognizerIntent?.extras?.getString(RecognizerIntent.EXTRA_LANGUAGE_MODEL)) {
//            "TINY_EN_Q8_0" -> WhichModel.TINY_EN_Q8_0
//            "TINY_EN_Q4_0" -> WhichModel.TINY_EN_Q4_0
//            "BASE_Q8_0" -> WhichModel.BASE_Q8_0
//            "BASE_Q4_0" -> WhichModel.BASE_Q4_0
//
//            // These k-quants are currently very slow and inaccurate
//            "BASE_Q4K" -> WhichModel.BASE_Q4K
//            "BASE_Q2K" -> WhichModel.BASE_Q2K
//            else -> WhichModel.TINY_EN_Q4_0
//        }
//
//        val isMultilingual = when (model) {
//            WhichModel.TINY_EN_Q8_0 -> false
//            WhichModel.TINY_EN_Q4_0 -> false
//            WhichModel.BASE_Q8_0 -> true
//            WhichModel.BASE_Q4_0 -> true
//            WhichModel.BASE_Q4K -> true
//            WhichModel.BASE_Q2K -> true
//        }
//
//        println(model)
//
//        if (!isWhisperLoaded() || loadedModel != model) {
//            val modelResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_q8_0_model
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_q4_0_model
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_q8_0_model
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_q4_0_model
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_q2k_model
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_q2k_model
//            }
//            val configResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_config
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_config
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_config
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_config
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_config
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_config
//            }
//            val tokenizerResId = when (model) {
//                WhichModel.TINY_EN_Q8_0 -> R.raw.whisper_tiny_en_tokenizer
//                WhichModel.TINY_EN_Q4_0 -> R.raw.whisper_tiny_en_tokenizer
//                WhichModel.BASE_Q8_0 -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q4_0 -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q4K -> R.raw.whisper_base_tokenizer
//                WhichModel.BASE_Q2K -> R.raw.whisper_base_tokenizer
//            }
//
//            println(recognizerIntent?.extras?.getString(RecognizerIntent.EXTRA_LANGUAGE_MODEL))
//
//            val modelFileDescriptor = resources.openRawResourceFd(modelResId)
//            val configFileDescriptor = resources.openRawResourceFd(configResId)
//            val tokenizerFileDescriptor = resources.openRawResourceFd(tokenizerResId)
//
//            loadWhisper(
//                modelFileDescriptor.parcelFileDescriptor.detachFd(),
//                modelFileDescriptor.startOffset.toULong(),
//                modelFileDescriptor.length.toULong(),
//
//                configFileDescriptor.parcelFileDescriptor.detachFd(),
//                configFileDescriptor.startOffset.toULong(),
//                configFileDescriptor.length.toULong(),
//
//                tokenizerFileDescriptor.parcelFileDescriptor.detachFd(),
//                tokenizerFileDescriptor.startOffset.toULong(),
//                tokenizerFileDescriptor.length.toULong(),
//            )
//
//            modelFileDescriptor.close()
//            configFileDescriptor.close()
//            tokenizerFileDescriptor.close()
//
//            loadedModel = model
//
//            println("loaded model")
//        }


        var totalTranscriptionTime = 0L

        recordAndTranscribeJob = recordAndTranscribeScope.launch recordAndTranscribe@{
            audioRecord.startRecording()
            isRecording.set(true)
            isSpeaking = false
            stopListening = false

            val audioRmsScope = CoroutineScope(Dispatchers.IO)
            transcribeJobs.clear()
            val transcriptions = mutableMapOf<Int, Transcription>()
            var transcriptionIndex by mutableIntStateOf(0)

            listener?.readyForSpeech(Bundle())

            while (isRecording.get() && isActive) {
                val buffer = ShortArray(bufferSize)

                val numberOfShorts = audioRecord.read(buffer, 0, bufferSize)

                for (i in 0 until numberOfShorts) {
                    if (transcriptions[transcriptionIndex] == null) {
                        transcriptions[transcriptionIndex] = Transcription(start = null, end = null, text = null)
                    }
                    transcriptions[transcriptionIndex]!!.audioData.add(buffer[i])
                }

                if (!isActive) {
                    audioRecord.stop()
                    audioRecord.release()
                    return@recordAndTranscribe
                }

                audioRmsScope.launch vadAndTranscribe@{
                    if (stopListening) {
                        isRecording.set(false)
                        listener?.endOfSpeech()

                        if ((transcriptionIndex != 0) && !autoStopRecognition && !isSpeaking) {
                            // break if we already transcribed once or more,
                            // is not auto stop recognition, and not speaking
                            // because we just want to stop then with the already transcribed parts.
                            // if there were no transcriptions then we assume VAD didn't work,
                            // so we transcribe everything all at once.
                            // if there were transcriptions, then VAD was working.
                            // in that case, we can break as if it was working at first it was
                            // probably working afterward, so processing the audio could
                            // be detrimental.
                            return@vadAndTranscribe
                        }

                        val transcription = transcriptions[transcriptionIndex]!!

                        if ((transcriptionIndex == 0) && !isSpeaking) {
                            transcription.start = 0.toDouble()
                        }

                        transcriptions[transcriptionIndex]!!.end =
                            (transcriptions[transcriptionIndex]!!.audioData.size - 1).toDouble()

                        val transcribeJob = transcribeScope.launch {
                            val timeBeforeTranscription = currentTimeMillis()

                            val transcriptionText =
                                whisperRepository.transcribeAudio(
                                    transcription.audioData.slice(
                                        ((transcription.start!!.toInt() - speechStartPadMs).coerceAtLeast(
                                            0
                                        ))..((transcription.end!!.toInt()).coerceAtMost(transcription.audioData.size - 1))
                                    )
                                        .toShortArray(),
                                )

                            transcription.text = transcriptionText

                            totalTranscriptionTime += currentTimeMillis() - timeBeforeTranscription

                            for (job in transcribeJobs.iterator().withIndex()) {
                                if (job.index < transcriptionIndex - 1) {
                                    job.value.join()
                                }
                            }

                            transcription.audioData.clear()

                            if (isPartialResults == true) {
                                val bundle = Bundle().apply {
                                    putStringArrayList(
                                        SpeechRecognizer.RESULTS_RECOGNITION,
                                        arrayListOf(
                                            transcription.text!!
                                        )
                                    )
                                }

                                if (!isActive) {
                                    return@launch
                                }

                                listener?.partialResults(bundle)
                            }
                        }

                        transcribeJobs.add(transcribeJob)

                        transcribeJob.start() // start transcribing this segment right now
                    } else {
                        sileroVadRepository.detect(buffer)?.forEach {
                            when (it.key) {
                                "start" -> {
                                    isSpeaking = true
                                    listener?.beginningOfSpeech()

                                    if (transcriptions[transcriptionIndex] == null) {
                                        transcriptions[transcriptionIndex] =
                                            Transcription(start = it.value, end = null, text = null)
                                    } else {
                                        transcriptions[transcriptionIndex]!!.start =
                                            it.value
                                    }
                                }

                                "end" -> {
                                    val transcription = transcriptions[transcriptionIndex]!!

                                    transcriptionIndex += 1
                                    sileroVadRepository.reset()

                                    if (transcription.end == null) {
                                        isSpeaking = false
                                        listener?.endOfSpeech()

                                        transcription.end = it.value

                                        val transcribeJob = transcribeScope.launch {
                                            val timeBeforeTranscription = currentTimeMillis()

                                            transcription.text =
                                                whisperRepository.transcribeAudio(
                                                    transcription.audioData.slice(
                                                        ((transcription.start!!.toInt() - speechStartPadMs).coerceAtLeast(
                                                            0
                                                        ))..((transcription.end!!.toInt()).coerceAtMost(transcription.audioData.size - 1))
                                                    )
                                                        .toShortArray(),
                                                )

                                            totalTranscriptionTime += currentTimeMillis() - timeBeforeTranscription

                                            for (job in transcribeJobs.iterator().withIndex()) {
                                                if (job.index < transcriptionIndex - 1) {
                                                    job.value.join()
                                                }
                                            }

                                            transcription.audioData.clear()

                                            if (isPartialResults == true) {
                                                val bundle = Bundle().apply {
                                                    putStringArrayList(
                                                        SpeechRecognizer.RESULTS_RECOGNITION,
                                                        arrayListOf(
                                                            transcription.text!!
                                                        )
                                                    )
                                                }

                                                if (!isActive) {
                                                    return@launch
                                                }

                                                listener?.partialResults(bundle)
                                            }
                                        }

                                        transcribeJobs.add(transcribeJob)

                                        transcribeJob.start() // start transcribing this segment right now

                                        if (autoStopRecognition) {
                                            isRecording.set(false)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (stopListening) {
                    break
                }
            }

            audioRecord.stop()
            audioRecord.release()

            sileroVadRepository.reset()

            if (!isActive) {
                return@recordAndTranscribe
            }

            runBlocking {
                transcribeJobs.forEach {
                    if (!isActive) {
                        return@forEach
                    }
                    it.join()
                }
            }

            if (!isActive) {
                return@recordAndTranscribe
            }

            val transcription = transcriptions.toSortedMap().values.joinToString { it.text ?: "" }

            val bundle = Bundle().apply {
                putStringArrayList(
                    SpeechRecognizer.RESULTS_RECOGNITION,
                    arrayListOf(
                        transcription
                    )
                )
            }


            try {
                listener?.results(bundle)
            } catch (e: RemoteException) {
                throw RuntimeException(e)
            }
        }
    }

    override fun onCancel(listener: Callback?) {
        recordAndTranscribeJob?.cancel()
        transcribeJobs.forEach {
            it.cancel()
        }
        sileroVadRepository.reset()
    }

    override fun onStopListening(listener: Callback?) {
        stopListening = true
    }

    override fun onDestroy() {
        super.onDestroy()
        recordAndTranscribeJob?.cancel()
        transcribeJobs.forEach {
            it.cancel()
        }
        runBlocking {
            whisperRepository.release()
            sileroVadRepository.release()
        }
    }
}
