package dev.soupslurpr.transcribro.recognitionservice.silerovad

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession

class SileroVadOnnxModel(modelBytes: ByteArray) {
    // Define private variable OrtSession
    private val session: OrtSession
    private lateinit var h: Array<Array<FloatArray>>
    private lateinit var c: Array<Array<FloatArray>>

    // Define the last sample rate
    private var lastSr = 0

    // Define the last batch size
    private var lastBatchSize = 0

    // Constructor
    init {
        // Get the ONNX runtime environment
        val env: OrtEnvironment = OrtEnvironment.getEnvironment()
        // Create an ONNX session options object
        val opts: OrtSession.SessionOptions = OrtSession.SessionOptions()
        // Set the InterOp thread count to 1, InterOp threads are used for parallel processing of different computation graph operations
        opts.setInterOpNumThreads(1)
        // Set the IntraOp thread count to 1, IntraOp threads are used for parallel processing within a single operation
        opts.setIntraOpNumThreads(1)
        // Add a CPU device, setting to false disables CPU execution optimization
        opts.addCPU(true)

//        opts.addConfigEntry("session.load_model_format", "ORT")
        // Create an ONNX session using the environment, model path, and options
        session = env.createSession(modelBytes, opts)
        // Reset states
        resetStates()
    }

    /**
     * Reset states
     */
    fun resetStates() {
        h = Array(2) { Array(1) { FloatArray(64) } }
        c = Array(2) { Array(1) { FloatArray(64) } }
        lastSr = 0
        lastBatchSize = 0
    }

    @Throws(OrtException::class)
    fun close() {
        session.close()
    }

    /**
     * Define inner class ValidationResult
     */
    class ValidationResult // Constructor
        (val x: Array<FloatArray?>, val sr: Int)

    /**
     * Function to validate input data
     */
    private fun validateInput(x: Array<FloatArray?>, sr: Int): ValidationResult {
        // Process the input data with dimension 1
        var x = x
        var sr = sr
        if (x.size == 1) {
            x = arrayOf(x[0])
        }
        // Throw an exception when the input data dimension is greater than 2
        require(x.size <= 2) { "Incorrect audio data dimension: " + x[0]!!.size }

        // Process the input data when the sample rate is not equal to 16000 and is a multiple of 16000
        if (sr != 16000 && (sr % 16000 == 0)) {
            val step = sr / 16000
            val reducedX = arrayOfNulls<FloatArray>(x.size)

            for (i in x.indices) {
                val current = x[i]
                val newArr = FloatArray((current!!.size + step - 1) / step)

                var j = 0
                var index = 0
                while (j < current.size) {
                    newArr[index] = current[j]
                    j += step
                    index++
                }

                reducedX[i] = newArr
            }

            x = reducedX
            sr = 16000
        }

        // If the sample rate is not in the list of supported sample rates, throw an exception
        require(SAMPLE_RATES.contains(sr)) { "Only supports sample rates $SAMPLE_RATES (or multiples of 16000)" }

        // If the input audio block is too short, throw an exception
        require(!((sr.toFloat()) / x[0]!!.size > 31.25)) { "Input audio is too short" }

        // Return the validated result
        return ValidationResult(x, sr)
    }

    /**
     * Method to call the ONNX model
     */
    @Throws(OrtException::class)
    fun call(x: Array<FloatArray?>, sr: Int): FloatArray {
        var x = x
        var sr = sr
        val result = validateInput(x, sr)
        x = result.x
        sr = result.sr

        val batchSize = x.size

        if (lastBatchSize == 0 || lastSr != sr || lastBatchSize != batchSize) {
            resetStates()
        }

        val env: OrtEnvironment = OrtEnvironment.getEnvironment()

        var inputTensor: OnnxTensor? = null
        var hTensor: OnnxTensor? = null
        var cTensor: OnnxTensor? = null
        var srTensor: OnnxTensor? = null
        var ortOutputs: OrtSession.Result? = null

        try {
            // Create input tensors
            inputTensor = OnnxTensor.createTensor(env, x)
            hTensor = OnnxTensor.createTensor(env, h)
            cTensor = OnnxTensor.createTensor(env, c)
            srTensor = OnnxTensor.createTensor(env, longArrayOf(sr.toLong()))

            val inputs: MutableMap<String, OnnxTensor?> = HashMap()
            inputs["input"] = inputTensor
            inputs["sr"] = srTensor
            inputs["h"] = hTensor
            inputs["c"] = cTensor

            // Call the ONNX model for calculation
            ortOutputs = session.run(inputs)
            // Get the output results
            val output = ortOutputs.get(0).value as Array<FloatArray>
            h = ortOutputs.get(1).value as Array<Array<FloatArray>>
            c = ortOutputs.get(2).value as Array<Array<FloatArray>>

            lastSr = sr
            lastBatchSize = batchSize
            return output[0]
        } finally {
            inputTensor?.close()
            hTensor?.close()
            cTensor?.close()
            srTensor?.close()
            ortOutputs?.close()
        }
    }

    companion object {
        // Define a list of supported sample rates
        private val SAMPLE_RATES: List<Int> = mutableListOf(8000, 16000)
    }
}