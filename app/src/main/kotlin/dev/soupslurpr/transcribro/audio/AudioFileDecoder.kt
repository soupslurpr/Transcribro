package dev.soupslurpr.transcribro.audio

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

/** Talks to Android's media services, using only a user-selected, read-only descriptor. */
internal class AudioFileDecoder(private val cancelled: () -> Boolean) {
    fun decode(
        descriptor: ParcelFileDescriptor,
        offset: Long,
        length: Long,
        consume: (ShortArray, Long) -> Unit,
    ) {
        require(offset >= 0 && length >= -1)
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        fun checkActive() {
            if (cancelled()) throw CancellationException()
        }
        try {
            checkActive()
            if (length >= 0) {
                extractor.setDataSource(descriptor.fileDescriptor, offset, length)
            } else {
                require(offset == 0L)
                extractor.setDataSource(descriptor.fileDescriptor)
            }
            val track =
                (0 until extractor.trackCount.coerceAtMost(64)).firstOrNull {
                    extractor
                        .getTrackFormat(it)
                        .getString(MediaFormat.KEY_MIME)
                        ?.startsWith("audio/") == true
                } ?: throw IOException("No supported audio track")
            val format = extractor.getTrackFormat(track)
            val duration =
                if (format.containsKey(MediaFormat.KEY_DURATION)) {
                    format.getLong(MediaFormat.KEY_DURATION)
                } else {
                    -1L
                }
            if (duration > MAX_DURATION_MS * 1000) throw AudioTooLongException()
            extractor.selectTrack(track)
            var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            var resampler = SpeechResampler(sampleRate)
            var outputSamples = 0L
            var consumedAudio = false
            fun deliver(samples: ShortArray) {
                checkActive()
                outputSamples += samples.size
                if (outputSamples > MAX_DURATION_MS * 16) throw AudioTooLongException()
                if (samples.isNotEmpty()) {
                    try {
                        consume(samples, if (duration > 0) duration / 1000 else -1)
                    } finally {
                        samples.fill(0)
                    }
                }
            }
            fun pcm(buffer: ByteBuffer) {
                require(channels in 1..8)
                val sampleBytes =
                    when (encoding) {
                        AudioFormat.ENCODING_PCM_16BIT -> 2
                        AudioFormat.ENCODING_PCM_FLOAT -> 4
                        else -> throw IOException("Unsupported PCM format")
                    }
                val frameBytes = sampleBytes * channels
                if (buffer.remaining() > MAX_BUFFER || buffer.remaining() % frameBytes != 0) {
                    throw IOException("Invalid PCM buffer")
                }
                buffer.order(ByteOrder.LITTLE_ENDIAN)
                // Small chunks also bound cancellation latency and resampler allocations.
                while (buffer.hasRemaining()) {
                    checkActive()
                    val mono =
                        FloatArray(minOf(4096, buffer.remaining() / frameBytes)) {
                            var sum = 0f
                            repeat(channels) {
                                val value =
                                    if (sampleBytes == 2) {
                                        buffer.short / 32768f
                                    } else {
                                        buffer.float
                                    }
                                if (value.isFinite()) sum += value.coerceIn(-1f, 1f)
                            }
                            sum / channels
                        }
                    consumedAudio = true
                    try {
                        deliver(resampler.process(mono))
                    } finally {
                        mono.fill(0f)
                    }
                }
            }
            val mime = checkNotNull(format.getString(MediaFormat.KEY_MIME))
            if (mime == "audio/raw") {
                if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                    encoding = format.getInteger(MediaFormat.KEY_PCM_ENCODING)
                }
                val buffer = ByteBuffer.allocateDirect(MAX_BUFFER)
                while (true) {
                    checkActive()
                    buffer.clear()
                    if (extractor.sampleSize > buffer.capacity()) {
                        throw IOException("Oversized audio packet")
                    }
                    val count = extractor.readSampleData(buffer, 0)
                    if (count < 0) break
                    buffer.position(0)
                    buffer.limit(count)
                    pcm(buffer)
                    if (!extractor.advance()) break
                }
            } else {
                format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                val decoder = MediaCodec.createDecoderByType(mime)
                codec = decoder
                decoder.configure(format, null, null, 0)
                decoder.start()
                var inputEnded = false
                var outputEnded = false
                var lastProgress = SystemClock.elapsedRealtime()
                val info = MediaCodec.BufferInfo()
                while (!outputEnded) {
                    checkActive()
                    if (!inputEnded) {
                        val inputIndex = decoder.dequeueInputBuffer(10_000)
                        if (inputIndex >= 0) {
                            val input = checkNotNull(decoder.getInputBuffer(inputIndex))
                            if (extractor.sampleSize > input.capacity()) {
                                throw IOException("Oversized audio packet")
                            }
                            val count = extractor.readSampleData(input, 0)
                            if (count < 0) {
                                decoder.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    0,
                                    0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                                )
                                inputEnded = true
                            } else {
                                decoder.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    count,
                                    extractor.sampleTime,
                                    0,
                                )
                                extractor.advance()
                            }
                            lastProgress = SystemClock.elapsedRealtime()
                        }
                    }
                    val outputIndex = decoder.dequeueOutputBuffer(info, 10_000)
                    if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val output = decoder.outputFormat
                        val nextRate = output.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        if (consumedAudio && nextRate != sampleRate) {
                            throw IOException("Sample rate changed")
                        }
                        if (nextRate != sampleRate) {
                            sampleRate = nextRate
                            resampler = SpeechResampler(sampleRate)
                        }
                        channels = output.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        encoding =
                            if (output.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                output.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            } else {
                                AudioFormat.ENCODING_PCM_16BIT
                            }
                    } else if (outputIndex >= 0) {
                        try {
                            if (info.size > 0) {
                                val output = checkNotNull(decoder.getOutputBuffer(outputIndex))
                                require(
                                    info.offset >= 0 &&
                                        info.size <= MAX_BUFFER &&
                                        info.size <= output.capacity() - info.offset
                                )
                                output.position(info.offset)
                                output.limit(info.offset + info.size)
                                pcm(output)
                            }
                            outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        } finally {
                            decoder.releaseOutputBuffer(outputIndex, false)
                        }
                        // consume() performs recognition synchronously; that is not a codec stall.
                        lastProgress = SystemClock.elapsedRealtime()
                    }
                    if (SystemClock.elapsedRealtime() - lastProgress > 10_000) {
                        throw IOException("Audio decoder stalled")
                    }
                }
            }
            deliver(resampler.finish())
        } finally {
            try {
                codec?.release()
            } finally {
                extractor.release()
            }
        }
    }

    class AudioTooLongException : IOException()

    companion object {
        const val MAX_DURATION_MS = 30L * 60 * 1000
        private const val MAX_BUFFER = 1024 * 1024
    }
}
