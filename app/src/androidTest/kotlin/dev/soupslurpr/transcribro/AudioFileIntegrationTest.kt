package dev.soupslurpr.transcribro

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.os.ParcelFileDescriptor
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.audio.AudioFileDecoder
import dev.soupslurpr.transcribro.audio.AudioFileTranscriber
import java.io.File
import java.io.IOException
import kotlin.math.log10
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AudioFileIntegrationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Before
    fun noMicrophonePermission() {
        assertNotEquals(
            "Revoke the candidate's RECORD_AUDIO before running file-only tests",
            PackageManager.PERMISSION_GRANTED,
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO),
        )
    }

    private suspend fun withAudio(name: String, action: suspend (AssetFileDescriptor) -> Unit) {
        val file = File(context.cacheDir, "file-integration-$name")
        try {
            instrumentation.context.assets.open(name).use { source ->
                file.outputStream().use(source::copyTo)
            }
            AssetFileDescriptor(
                    ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY),
                    0,
                    file.length(),
                )
                .use {
                    action(it)
                }
        } finally {
            assertTrue(file.delete())
        }
    }

    @Test
    fun wavAndAacDecodeAndTranscribeInAnIsolatedWorkerWithoutMicrophoneAccess() = runBlocking {
        for (name in listOf("dictation-stereo.wav", "dictation-stereo.m4a")) {
            withAudio(name) { file ->
                val text = StringBuilder()
                val progress = mutableListOf<Long>()
                withTimeout(90_000) {
                    AudioFileTranscriber(context)
                        .transcribe(
                            file,
                            onProgress = { elapsed, _ -> progress.add(elapsed) },
                            onPhrase = { text.append(it).append(' ') },
                        )
                }
                assertTrue("$name: $text", text.contains("voice", true))
                // Check speech from both sentences. Exact-word accuracy is measured
                // separately; the bundled recognizer can mishear the lossy AAC fixture.
                val words = text.toString().lowercase()
                assertTrue("$name: $text", words.indexOf("yellow") > words.indexOf("voice"))
                assertTrue("$name: $text", words.indexOf("table") > words.indexOf("yellow"))
                assertTrue(progress.isNotEmpty())
                assertTrue(progress.zipWithNext().all { (a, b) -> b >= a })
            }
        }
    }

    @Test
    fun wavDownmixAndResamplingPreserveTheSourceWaveform() = runBlocking {
        val reference =
            instrumentation.context.assets.open("dictation-fixture.pcm").use {
                it.readBytes()
            }
        try {
            withAudio("dictation-stereo.wav") { file ->
                var frames = 0
                var signal = 0.0
                var error = 0.0
                AudioFileDecoder { false }
                    .decode(file.parcelFileDescriptor, 0, file.length) { samples, _ ->
                        for (sample in samples) {
                            assertTrue("Decoder added frames", frames * 2 + 1 < reference.size)
                            val mono =
                                ((reference[frames * 2].toInt() and 255) or
                                        (reference[frames * 2 + 1].toInt() shl 8))
                                    .toShort()
                                    .toDouble()
                            // The stereo fixture has 3 dB less amplitude in each channel than
                            // the mono reference. Downmixing must preserve that source level.
                            val expected = mono / kotlin.math.sqrt(2.0)
                            signal += expected * expected
                            error += (sample - expected) * (sample - expected)
                            frames++
                        }
                    }
                assertEquals("Decoder lost frames", reference.size / 2, frames)
                val snr = 10 * log10(signal / error)
                assertTrue("Downmixed waveform differs from reference: $snr dB", snr > 30)
            }
        } finally {
            reference.fill(0)
        }
    }

    @Test
    fun arbitraryTextIsNotTreatedAsAudio() = runBlocking {
        withAudio("invalid-audio.txt") { file ->
            try {
                withTimeout(30_000) {
                    AudioFileTranscriber(context)
                        .transcribe(
                            file,
                            { _, _ -> },
                            { fail("Not speech") },
                        )
                }
                fail("Invalid input must fail")
            } catch (_: IOException) {}
        }
    }

    @Test
    fun cancellationDuringPreparationDoesNotCommitLateText() = runBlocking {
        withAudio("dictation-stereo.m4a") { file ->
            var cancelled = false
            var lateText = false
            val task = launch {
                AudioFileTranscriber(context)
                    .transcribe(
                        file,
                        { _, _ -> },
                        {
                            if (cancelled) {
                                lateText = true
                            }
                        },
                    )
            }
            delay(100)
            cancelled = true
            withTimeout(5000) { task.cancelAndJoin() }
            delay(1000)
            assertFalse(lateText)
        }
    }
}
