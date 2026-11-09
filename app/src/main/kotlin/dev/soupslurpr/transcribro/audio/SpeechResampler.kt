package dev.soupslurpr.transcribro.audio

import kotlin.math.*

/** Streaming, anti-aliased conversion to 16 kHz. State does not grow with file duration. */
internal class SpeechResampler(private val inputRate: Int) {
    init {
        require(inputRate in 8_000..192_000)
    }

    private val ratio = inputRate / 16_000.0
    private val radius = ceil(32.0 * max(1.0, inputRate / 48_000.0)).toInt()
    private val ring = FloatArray(radius * 2 + 2)
    private var received = 0L
    private var emitted = 0L
    private var finished = false
    private val kernels =
        Array(256) { phase ->
            val fraction = phase / 256.0
            val cutoff = min(1.0, 16_000.0 / inputRate) * 0.92
            val weights =
                DoubleArray(radius * 2 + 1) { index ->
                    val distance = index - radius - fraction
                    val sinc =
                        if (abs(distance) < 1e-9) {
                            cutoff
                        } else {
                            sin(PI * cutoff * distance) / (PI * distance)
                        }
                    val window =
                        if (abs(distance) > radius) {
                            0.0
                        } else {
                            0.5 + 0.5 * cos(PI * distance / radius)
                        }
                    sinc * window
                }
            val sum = weights.sum()
            FloatArray(weights.size) { (weights[it] / sum).toFloat() }
        }

    fun process(input: FloatArray): ShortArray {
        check(!finished)
        if (inputRate == 16_000) {
            received += input.size
            emitted += input.size
            return ShortArray(input.size) { pcm(input[it]) }
        }
        val output = ShortArray(ceil(input.size / ratio).toInt() + 4)
        var size = 0
        for (sample in input) {
            ring[(received++ % ring.size).toInt()] =
                if (sample.isFinite()) sample.coerceIn(-1f, 1f) else 0f
            while (floor(emitted * ratio).toLong() + radius < received) {
                output[size++] = interpolate()
                emitted++
            }
        }
        return output.copyOf(size)
    }

    fun finish(): ShortArray {
        check(!finished)
        finished = true
        val expected = (received * 16_000 + inputRate - 1) / inputRate
        return ShortArray((expected - emitted).toInt()) { interpolate().also { emitted++ } }
    }

    private fun interpolate(): Short {
        val position = emitted * ratio
        val center = floor(position).toLong()
        val kernel = kernels[((position - center) * kernels.size).toInt().coerceIn(kernels.indices)]
        var value = 0f
        for (index in kernel.indices) {
            val sample = center + index - radius
            if (sample >= 0 && sample < received && received - sample <= ring.size) {
                value += ring[(sample % ring.size).toInt()] * kernel[index]
            }
        }
        return pcm(value)
    }

    private fun pcm(value: Float): Short =
        if (!value.isFinite()) {
            0
        } else {
            (value.coerceIn(-1f, 1f) * 32768f).roundToInt().coerceIn(-32768, 32767).toShort()
        }
}
