package dev.soupslurpr.transcribro

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RecognitionIntegrationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private var client: ActivityScenario<RecognitionClientActivity>? = null
    private var clientContext: Context? = null
    private var previousVoiceService: String? = null
    private var changedVoiceService = false

    @Before
    fun microphonePermissionIsGranted() {
        assertEquals(
            "Grant RECORD_AUDIO to the verification client before running integration tests",
            PackageManager.PERMISSION_GRANTED,
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO),
        )
        if (context.packageName == "dev.soupslurpr.transcribro.verification") {
            // Android only propagates background microphone capabilities to the
            // selected recognition service (or a preinstalled/on-device service).
            // Exercise the real user setup, and restore it even if a test fails.
            previousVoiceService =
                Settings.Secure.getString(
                    context.contentResolver,
                    "voice_recognition_service",
                )
            val service =
                "dev.soupslurpr.transcribro.reimagined/" +
                    "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService"
            if (previousVoiceService != service) {
                setVoiceService(service)
                changedVoiceService = true
            }
            // The public recognition contract performs while-in-use permission
            // checks even for supplied PCM. Model a visible requesting app.
            client = ActivityScenario.launch(RecognitionClientActivity::class.java)
            // RESUMED can precede window focus. Wait for the visible caller.
            val deadline = android.os.SystemClock.elapsedRealtime() + 5_000
            var focused = false
            while (!focused && android.os.SystemClock.elapsedRealtime() < deadline) {
                client?.onActivity {
                    clientContext = it
                    focused = it.hasWindowFocus()
                }
                if (!focused) Thread.sleep(25)
            }
            assertTrue("Recognition client never became visible and focused", focused)
        }
    }

    @After
    fun closeClient() {
        try {
            client?.close()
        } finally {
            clientContext = null
            if (changedVoiceService) setVoiceService(previousVoiceService)
        }
    }

    private fun setVoiceService(value: String?) {
        val automation = instrumentation.uiAutomation
        automation.adoptShellPermissionIdentity(Manifest.permission.WRITE_SECURE_SETTINGS)
        try {
            check(
                Settings.Secure.putString(
                    context.contentResolver,
                    "voice_recognition_service",
                    value,
                )
            )
        } finally {
            // Never retain shell identity during a recognition permission check.
            automation.dropShellPermissionIdentity()
        }
    }

    private class Result : RecognitionListener {
        val ready = CountDownLatch(1)
        val done = CountDownLatch(1)
        var text = ""
        var error = 0
        val partials = mutableListOf<String>()
        val confirmed = mutableListOf<String>()
        val phraseEnds = mutableListOf<String>()

        override fun onReadyForSpeech(params: Bundle?) {
            ready.countDown()
        }

        override fun onBeginningOfSpeech() = Unit

        override fun onRmsChanged(rmsdB: Float) = Unit

        override fun onBufferReceived(buffer: ByteArray?) = Unit

        override fun onEndOfSpeech() = Unit

        override fun onError(error: Int) {
            this.error = error
            done.countDown()
        }

        override fun onResults(results: Bundle?) {
            text =
                results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()
            done.countDown()
        }

        override fun onPartialResults(partialResults: Bundle?) {
            partials +=
                partialResults
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()
            val length =
                partialResults?.getInt("dev.soupslurpr.transcribro.CONFIRMED_LENGTH", -1) ?: -1
            assertTrue("Missing or invalid confirmed length", length in 0..partials.last().length)
            confirmed += partials.last().take(length)
            if (partialResults?.getBoolean("dev.soupslurpr.transcribro.PHRASE_END") == true) {
                phraseEnds += partials.last()
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) = Unit
    }

    private fun recognize(
        bytes: ByteArray,
        sampleRate: Int = 16000,
        extras: Bundle = Bundle(),
    ): Result {
        val pipe = ParcelFileDescriptor.createPipe()
        val result = Result()
        lateinit var recognizer: SpeechRecognizer
        instrumentation.runOnMainSync {
            recognizer =
                SpeechRecognizer.createSpeechRecognizer(
                    clientContext ?: context,
                    ComponentName(
                        "dev.soupslurpr.transcribro.reimagined",
                        "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService",
                    ),
                )
            recognizer.setRecognitionListener(result)
            recognizer.startListening(
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe[0])
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, sampleRate)
                    .putExtra(
                        RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                        AudioFormat.ENCODING_PCM_16BIT,
                    )
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                    .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    .putExtras(extras)
            )
        }
        val writer = Thread {
            try {
                ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { it.write(bytes) }
            } catch (_: java.io.IOException) {
                /* Rejected/cancelled input closes its pipe. */
            }
        }
            .apply { start() }
        try {
            assertTrue("Recognition did not terminate", result.done.await(60, TimeUnit.SECONDS))
            return result
        } finally {
            instrumentation.runOnMainSync {
                recognizer.cancel()
                recognizer.destroy()
            }
            pipe[0].close()
            runCatching { pipe[1].close() }
            writer.join(3000)
            assertFalse("Audio writer leaked", writer.isAlive)
        }
    }

    private fun rejectedRequest(intent: Intent): Result {
        val result = Result()
        lateinit var recognizer: SpeechRecognizer
        instrumentation.runOnMainSync {
            recognizer =
                SpeechRecognizer.createSpeechRecognizer(
                    clientContext ?: context,
                    ComponentName(
                        "dev.soupslurpr.transcribro.reimagined",
                        "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService",
                    ),
                )
            recognizer.setRecognitionListener(result)
            recognizer.startListening(intent)
        }
        try {
            assertTrue(
                "Rejected request did not terminate",
                result.done.await(10, TimeUnit.SECONDS),
            )
            assertEquals("Rejected request started capture", 1L, result.ready.count)
            assertTrue(result.text.isEmpty())
            assertTrue(result.partials.isEmpty())
            return result
        } finally {
            instrumentation.runOnMainSync {
                recognizer.cancel()
                recognizer.destroy()
            }
        }
    }

    @Test
    fun missingSuppliedDescriptorDoesNotFallBackToTheMicrophone() {
        val intent =
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, null as ParcelFileDescriptor?)
        assertEquals(SpeechRecognizer.ERROR_AUDIO, rejectedRequest(intent).error)
    }

    @Test
    fun serviceWithoutMicrophonePermissionAcceptsPcmButRejectsLiveCapture() {
        val servicePackage = "dev.soupslurpr.transcribro.reimagined"
        val initiallyGranted =
            context.packageManager.checkPermission(
                Manifest.permission.RECORD_AUDIO,
                servicePackage,
            ) == PackageManager.PERMISSION_GRANTED
        instrumentation.uiAutomation.revokeRuntimePermission(
            servicePackage,
            Manifest.permission.RECORD_AUDIO,
        )
        try {
            val denied = rejectedRequest(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH))
            assertEquals(SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS, denied.error)
            val pcm =
                instrumentation.context.assets.open("dictation-fixture.pcm").use { it.readBytes() }
            val supplied = recognize(pcm)
            assertEquals(0, supplied.error)
            assertTrue(
                "Supplied PCM was not transcribed: ${supplied.text}",
                supplied.text.contains("on the table", true),
            )
        } finally {
            if (initiallyGranted) {
                instrumentation.uiAutomation.grantRuntimePermission(
                    servicePackage,
                    Manifest.permission.RECORD_AUDIO,
                )
            }
        }
    }

    @Test
    fun liveMicrophoneUsesCallerAttributionAndCancels() {
        val result = Result()
        lateinit var recognizer: SpeechRecognizer
        instrumentation.runOnMainSync {
            recognizer =
                SpeechRecognizer.createSpeechRecognizer(
                    clientContext ?: context,
                    ComponentName(
                        "dev.soupslurpr.transcribro.reimagined",
                        "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService",
                    ),
                )
            recognizer.setRecognitionListener(result)
            recognizer.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH))
        }
        try {
            assertTrue(
                "Microphone did not become ready (error ${result.error})",
                result.ready.await(10, TimeUnit.SECONDS),
            )
            assertEquals(0, result.error)
        } finally {
            instrumentation.runOnMainSync {
                recognizer.cancel()
                recognizer.destroy()
            }
        }
    }

    @Test
    fun generatedSpeechPassesThroughTheRealIsolatedEngine() {
        val pcm =
            instrumentation.context.assets.open("dictation-fixture.pcm").use {
                it.readBytes()
            }
        val result = recognize(pcm)
        assertEquals("Recognition error ${result.error}", 0, result.error)
        assertTrue(
            "Expected recognizable speech, got: ${result.text}",
            result.text.contains("voice", true),
        )
        assertTrue(
            "Expected the second sentence, got: ${result.text}",
            result.text.contains("on the table", true),
        )
        assertTrue(result.partials.isNotEmpty())
        assertEquals(result.text, result.partials.last())
        assertTrue(
            "Streaming text never appeared before finalization",
            result.partials.zip(result.confirmed).any { (all, stable) ->
                all.length > stable.length
            },
        )
        result.confirmed.zipWithNext().forEach { (a, b) -> assertTrue(b.startsWith(a)) }
        assertEquals(result.text, result.confirmed.last())
    }

    @Test
    fun silenceDoesNotHallucinateText() {
        val result = recognize(ByteArray(16000 * 2 * 2))
        assertEquals(SpeechRecognizer.ERROR_NO_MATCH, result.error)
        assertEquals("", result.text)
        assertTrue(result.partials.isEmpty())
    }

    @Test
    fun pauseBoundaryFollowsCommittedSpeechThroughTheExternalRecognitionContract() {
        val pcm =
            instrumentation.context.assets.open("dictation-fixture.pcm").use { it.readBytes() }
        // Force a real pause after the fixture, rather than relying on EOF to finish it.
        val result = recognize(pcm + ByteArray(16_000 * 2 * 4))
        assertEquals(0, result.error)
        assertTrue("Missing speech-pause boundary", result.phraseEnds.isNotEmpty())
        assertTrue(result.phraseEnds.all { it.isNotBlank() })
        assertEquals(result.text, result.phraseEnds.last())
    }

    @Test
    fun legacyContinuousOptionRetainsSuccessivePhrasesAndNewOptionTakesPrecedence() {
        val pcm =
            instrumentation.context.assets.open("dictation-fixture.pcm").use { it.readBytes() }
        val successive = pcm + ByteArray(16_000 * 2 * 4) + pcm
        val legacy =
            Bundle().apply { putBoolean("dev.soupslurpr.transcribro.EXTRA_AUTO_STOP", false) }
        val continuous = recognize(successive, extras = legacy)
        assertEquals(0, continuous.error)
        assertEquals(
            continuous.text,
            2,
            Regex("voice", RegexOption.IGNORE_CASE).findAll(continuous.text).count(),
        )
        val overridden =
            recognize(
                successive,
                extras =
                    Bundle(legacy).apply {
                        putBoolean("dev.soupslurpr.transcribro.CONTINUOUS", false)
                    },
            )
        assertEquals(0, overridden.error)
        assertEquals(
            overridden.text,
            1,
            Regex("voice", RegexOption.IGNORE_CASE).findAll(overridden.text).count(),
        )
        successive.fill(0)
        pcm.fill(0)
    }

    @Test
    fun wrongSampleRateIsRejectedInsteadOfMisinterpreted() {
        assertEquals(SpeechRecognizer.ERROR_AUDIO, recognize(ByteArray(64), 44100).error)
    }

    @Test
    fun incompletePcmSampleIsRejected() {
        assertEquals(SpeechRecognizer.ERROR_AUDIO, recognize(byteArrayOf(1)).error)
    }

    @Test
    fun emptyInputTerminates() {
        assertEquals(SpeechRecognizer.ERROR_NO_MATCH, recognize(byteArrayOf()).error)
    }

    @Test
    fun cancellingAStalledSourceAllowsAnotherSession() {
        val pipe = ParcelFileDescriptor.createPipe()
        val result = Result()
        lateinit var recognizer: SpeechRecognizer
        instrumentation.runOnMainSync {
            recognizer =
                SpeechRecognizer.createSpeechRecognizer(
                    clientContext ?: context,
                    ComponentName(
                        "dev.soupslurpr.transcribro.reimagined",
                        "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService",
                    ),
                )
            recognizer.setRecognitionListener(result)
            recognizer.startListening(
                Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe[0])
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000)
                    .putExtra(
                        RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                        AudioFormat.ENCODING_PCM_16BIT,
                    )
                    .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
            )
        }
        try {
            val ready = result.ready.await(30, TimeUnit.SECONDS)
            assertTrue("Worker never became ready (error ${result.error})", ready)
            instrumentation.runOnMainSync { recognizer.cancel() }
            // Keep the producer open and silent: cancellation must not wait for EOF.
            assertEquals(SpeechRecognizer.ERROR_NO_MATCH, recognize(byteArrayOf()).error)
            assertEquals("Cancelled session delivered a terminal callback", 1L, result.done.count)
            assertTrue(result.partials.isEmpty())
        } finally {
            instrumentation.runOnMainSync { recognizer.destroy() }
            pipe.forEach { it.close() }
        }
    }
}

/** A visible external caller for Android's while-in-use microphone attribution. */
class RecognitionClientActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(
            TextView(this).apply {
                text = "Testing Transcribro recognition. Test audio is not saved."
                setPadding(48, 96, 48, 48)
            }
        )
    }
}
