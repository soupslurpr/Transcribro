package dev.soupslurpr.transcribro

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.os.Bundle
import android.os.OutcomeReceiver
import android.os.ParcelFileDescriptor
import android.os.PowerMonitor
import android.os.PowerMonitorReadings
import android.os.SystemClock
import android.os.health.SystemHealthManager
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.WindowManager
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.LockSupport
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in silent, paced PCM benchmark. Public speech and results stay in app-private test files. */
@RunWith(AndroidJUnit4::class)
class RecognitionBenchmark {
    @Test
    fun suppliedPcm() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val target = arguments.getString("targetPackage") ?: error("Specify targetPackage")
        val input = File(context.filesDir, arguments.getString("input") ?: "benchmark.pcm")
        val resultFile = File(context.filesDir, "benchmark-result.json")
        val component =
            ComponentName(
                target,
                "dev.soupslurpr.transcribro.recognitionservice.MainRecognitionService",
            )
        val previousService =
            Settings.Secure.getString(context.contentResolver, "voice_recognition_service")
        fun selectService(value: String?) {
            instrumentation.uiAutomation.adoptShellPermissionIdentity(
                Manifest.permission.WRITE_SECURE_SETTINGS
            )
            try {
                check(
                    Settings.Secure.putString(
                        context.contentResolver,
                        "voice_recognition_service",
                        value,
                    )
                )
            } finally {
                instrumentation.uiAutomation.dropShellPermissionIdentity()
            }
        }
        assertTrue(
            "Supply a nonempty PCM16 fixture",
            input.isFile && input.length() > 0 && input.length() % 2 == 0L,
        )
        selectService(component.flattenToString())
        try {
            ActivityScenario.launch(RecognitionClientActivity::class.java).use { activity ->
                lateinit var caller: Context
                var focused = false
                val focusDeadline = SystemClock.elapsedRealtime() + 5000
                while (!focused && SystemClock.elapsedRealtime() < focusDeadline) {
                    activity.onActivity {
                        caller = it
                        it.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                        focused = it.hasWindowFocus()
                    }
                    if (!focused) Thread.sleep(25)
                }
                assertTrue("Caller window has no focus", focused)
                val pipe = ParcelFileDescriptor.createPipe()
                val ready = CountDownLatch(1)
                val done = CountDownLatch(1)
                val events = JSONArray()
                val powerBefore = powerSnapshot(context)
                var error = 0
                var endReason = -1
                var text = ""
                val started = SystemClock.elapsedRealtimeNanos()
                fun elapsed() = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
                fun event(kind: String, value: String = "") {
                    synchronized(events) {
                        events.put(
                            JSONObject().put("event", kind).put("ms", elapsed()).put("text", value)
                        )
                    }
                }
                var readyMs = -1.0
                var finishedMs = -1.0
                var inputEndMs = -1.0
                var bytesSent = 0L
                var maxWriterLagMs = 0.0
                var writerError: String? = null
                lateinit var recognizer: SpeechRecognizer
                val writer = Thread {
                    try {
                        if (arguments.getString("startAudioImmediately") != "true") {
                            assertTrue(
                                "Service never became ready",
                                ready.await(45, TimeUnit.SECONDS),
                            )
                        }
                        if (done.count == 0L) return@Thread
                        val origin = SystemClock.elapsedRealtimeNanos()
                        input.inputStream().use { source ->
                            ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { output ->
                                val buffer = ByteArray(1024)
                                while (true) {
                                    val count = source.read(buffer)
                                    if (count < 0) break
                                    val targetTime = origin + bytesSent * 1_000_000_000L / 32000
                                    var remaining = targetTime - SystemClock.elapsedRealtimeNanos()
                                    while (remaining > 0) {
                                        LockSupport.parkNanos(remaining)
                                        remaining = targetTime - SystemClock.elapsedRealtimeNanos()
                                    }
                                    output.write(buffer, 0, count)
                                    bytesSent += count
                                    maxWriterLagMs =
                                        maxOf(
                                            maxWriterLagMs,
                                            (SystemClock.elapsedRealtimeNanos() - targetTime) / 1e6,
                                        )
                                }
                            }
                        }
                        inputEndMs = elapsed()
                        event("input_eof")
                    } catch (failure: Throwable) {
                        writerError = failure.toString()
                        event("writer_error", writerError.orEmpty())
                    }
                }
                instrumentation.runOnMainSync {
                    recognizer = SpeechRecognizer.createSpeechRecognizer(caller, component)
                    recognizer.setRecognitionListener(
                        object : RecognitionListener {
                            override fun onReadyForSpeech(params: Bundle?) {
                                readyMs = elapsed()
                                event("ready")
                                if (arguments.getString("targetVisible") == "true") {
                                    val launch =
                                        context.packageManager.getLaunchIntentForPackage(target)
                                            ?: error("Target has no launcher activity")
                                    context.startActivity(
                                        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    )
                                    event("target_activity_launched")
                                }
                                ready.countDown()
                            }

                            override fun onBeginningOfSpeech() {
                                event("speech_start")
                            }

                            override fun onRmsChanged(rmsdB: Float) = Unit

                            override fun onBufferReceived(buffer: ByteArray?) = Unit

                            override fun onEndOfSpeech() {
                                event("speech_end")
                            }

                            override fun onError(code: Int) {
                                error = code
                                finishedMs = elapsed()
                                event("error", code.toString())
                                done.countDown()
                                ready.countDown()
                            }

                            override fun onResults(results: Bundle?) {
                                endReason =
                                    results?.getInt("dev.soupslurpr.transcribro.END_REASON", -1)
                                        ?: -1
                                text =
                                    results
                                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                        ?.firstOrNull()
                                        .orEmpty()
                                finishedMs = elapsed()
                                event("results", text)
                                done.countDown()
                            }

                            override fun onPartialResults(partialResults: Bundle?) {
                                val hypothesis =
                                    partialResults
                                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                                        ?.firstOrNull()
                                        .orEmpty()
                                if (hypothesis.isNotBlank())
                                    event(
                                        if (
                                            partialResults?.getBoolean(
                                                "dev.soupslurpr.transcribro.PHRASE_END"
                                            ) == true
                                        )
                                            "phrase"
                                        else "partial",
                                        hypothesis,
                                    )
                            }

                            override fun onEvent(eventType: Int, params: Bundle?) = Unit
                        }
                    )
                    recognizer.startListening(
                        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                            .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe[0])
                            .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000)
                            .putExtra(
                                RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                                AudioFormat.ENCODING_PCM_16BIT,
                            )
                            .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                            .putExtra(
                                RecognizerIntent.EXTRA_PARTIAL_RESULTS,
                                arguments.getString("previews") != "false",
                            )
                            .putExtra("dev.soupslurpr.transcribro.CONTINUOUS", true)
                            .putExtra("dev.soupslurpr.transcribro.EXTRA_AUTO_STOP", false)
                    )
                }
                writer.start()
                try {
                    val complete = done.await(input.length() / 32000 + 180, TimeUnit.SECONDS)
                    writer.join(5000)
                    resultFile.writeText(
                        JSONObject()
                            .put("target", target)
                            .put("target_visible", arguments.getString("targetVisible") == "true")
                            .put(
                                "audio_starts_with_request",
                                arguments.getString("startAudioImmediately") == "true",
                            )
                            .put("input_bytes", input.length())
                            .put("bytes_sent", bytesSent)
                            .put("audio_seconds", input.length() / 32000.0)
                            .put("ready_ms", readyMs)
                            .put("finished_ms", finishedMs)
                            .put("input_end_ms", inputEndMs)
                            .put("max_writer_lag_ms", maxWriterLagMs)
                            .put("writer_error", writerError)
                            .put("error", error)
                            .put("end_reason", endReason)
                            .put("complete", complete)
                            .put("text", text)
                            .put("events", events)
                            .put("power_before", powerBefore)
                            .put("power_after", powerSnapshot(context))
                            .toString(2)
                    )
                    assertTrue(
                        "Benchmark did not complete; inspect benchmark-result.json",
                        complete,
                    )
                    assertTrue("Recognition error $error", error == 0)
                    if (target.contains(".reimagined")) {
                        assertTrue("Live pipeline ended with reason $endReason", endReason == 0)
                    }
                    assertTrue(
                        "Input delivery failed: $writerError",
                        writerError == null && bytesSent == input.length(),
                    )
                } finally {
                    instrumentation.runOnMainSync {
                        recognizer.cancel()
                        recognizer.destroy()
                    }
                    pipe.forEach { runCatching { it.close() } }
                    writer.interrupt()
                    writer.join(2000)
                }
            }
        } finally {
            selectService(previousService)
        }
    }

    private fun powerSnapshot(context: Context): JSONObject {
        val output = JSONObject()
        try {
            val manager = context.getSystemService(SystemHealthManager::class.java)
            var monitors = emptyList<PowerMonitor>()
            val listed = CountDownLatch(1)
            manager.getSupportedPowerMonitors(null) {
                monitors = it
                listed.countDown()
            }
            if (!listed.await(5, TimeUnit.SECONDS))
                return output.put("error", "monitor list timeout")
            if (monitors.isEmpty()) return output.put("unavailable", true)
            val read = CountDownLatch(1)
            manager.getPowerMonitorReadings(
                monitors,
                null,
                object : OutcomeReceiver<PowerMonitorReadings, RuntimeException> {
                    override fun onResult(result: PowerMonitorReadings) {
                        val values = JSONArray()
                        monitors.forEach { monitor ->
                            values.put(
                                JSONObject()
                                    .put("name", monitor.name)
                                    .put("type", monitor.type)
                                    .put("energy_uws", result.getConsumedEnergy(monitor))
                                    .put("timestamp_ms", result.getTimestampMillis(monitor))
                            )
                        }
                        output.put("monitors", values)
                        read.countDown()
                    }

                    override fun onError(error: RuntimeException) {
                        output.put("error", error.toString())
                        read.countDown()
                    }
                },
            )
            if (!read.await(5, TimeUnit.SECONDS)) output.put("error", "readings timeout")
        } catch (failure: Throwable) {
            output.put("error", failure.toString())
        }
        return output
    }
}
