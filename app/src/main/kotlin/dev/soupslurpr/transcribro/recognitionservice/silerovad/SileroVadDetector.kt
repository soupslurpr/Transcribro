package dev.soupslurpr.transcribro.recognitionservice.silerovad

import ai.onnxruntime.OrtException
import kotlin.math.max

class SileroVadDetector(
    modelBytes: ByteArray,
    startThreshold: Float,
    endThreshold: Float,
    samplingRate: Int,
    minSilenceDurationMs: Int,
    speechPadMs: Int
) {
    // OnnxModel model used for speech processing
    private val model: SileroVadOnnxModel

    // Threshold for speech start
    private val startThreshold: Float

    // Threshold for speech end
    private val endThreshold: Float

    // Sampling rate
    private val samplingRate: Int

    // Minimum number of silence samples to determine the end threshold of speech
    private val minSilenceSamples: Float

    // Additional number of samples for speech start or end to calculate speech start or end time
    private val speechPadSamples: Float

    // Whether in the triggered state (i.e. whether speech is being detected)
    private var triggered = false

    // Temporarily stored number of speech end samples
    private var tempEnd = 0

    // Number of samples currently being processed
    private var currentSample = 0


    init {
        // Check if the sampling rate is 8000 or 16000, if not, throw an exception
        require(!(samplingRate != 8000 && samplingRate != 16000)) { "does not support sampling rates other than [8000, 16000]" }

        // Initialize the parameters
        this.model = SileroVadOnnxModel(modelBytes)
        this.startThreshold = startThreshold
        this.endThreshold = endThreshold
        this.samplingRate = samplingRate
        this.minSilenceSamples = samplingRate * minSilenceDurationMs / 1000f
        this.speechPadSamples = samplingRate * speechPadMs / 1000f
        // Reset the state
        reset()
    }

    // Method to reset the state, including the model state, trigger state, temporary end time, and current sample count
    fun reset() {
        model.resetStates()
        triggered = false
        tempEnd = 0
        currentSample = 0
    }

    // apply method for processing the audio array, returning possible speech start or end times
    fun apply(audioData: FloatArray, returnSeconds: Boolean): Map<String, Double> {
        // Get the length of the audio array as the window size
        val windowSizeSamples = audioData.size
        // Update the current sample count
        currentSample += windowSizeSamples

        // Call the model to get the prediction probability of speech
        var speechProb = 0f
        try {
            speechProb = model.call(arrayOf(audioData), samplingRate)[0]
        } catch (e: OrtException) {
            throw RuntimeException(e)
        }

        // If the speech probability is greater than the threshold and the temporary end time is not 0, reset the temporary end time
        // This indicates that the speech duration has exceeded expectations and needs to recalculate the end time
        if (speechProb >= startThreshold && tempEnd != 0) {
            tempEnd = 0
        }

        // If the speech probability is greater than the threshold and not in the triggered state, set to triggered state and calculate the speech start time
        if (speechProb >= startThreshold && !triggered) {
            triggered = true
            var speechStart = (currentSample - speechPadSamples)
            speechStart = max(speechStart, 0.0f)
            val result: MutableMap<String, Double> = HashMap()

            result["start"] = speechStart.toDouble()

            return result
        }

        // If the speech probability is less than a certain threshold and in the triggered state, calculate the speech end time
        if (speechProb < endThreshold && triggered) {
            // Initialize or update the temporary end time
            if (tempEnd == 0) {
                tempEnd = currentSample
            }
            // If the number of silence samples between the current sample and the temporary end time is less than the minimum silence samples, return null
            // This indicates that it is not yet possible to determine whether the speech has ended
            if (currentSample - tempEnd < minSilenceSamples) {
                return emptyMap()
            } else {
                // Calculate the speech end time, reset the trigger state and temporary end time
                val speechEnd = (tempEnd + speechPadSamples)
                tempEnd = 0
                triggered = false
                val result: MutableMap<String, Double> = HashMap()

                result["end"] = speechEnd.toDouble()

                return result
            }
        }

        // If the above conditions are not met, return null by default
        return emptyMap()
    }

    @Throws(OrtException::class)
    fun close() {
        reset()
        model.close()
    }
}