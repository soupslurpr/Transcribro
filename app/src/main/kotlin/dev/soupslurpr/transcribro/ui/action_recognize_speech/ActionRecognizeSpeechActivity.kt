package dev.soupslurpr.transcribro.ui.action_recognize_speech

import android.Manifest
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognizerIntent
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.PreferencesRepository
import dev.soupslurpr.transcribro.ui.DictationControls
import dev.soupslurpr.transcribro.ui.ListeningFeedback
import dev.soupslurpr.transcribro.ui.TranscribroTheme
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

class ActionRecognizeSpeechActivity : ComponentActivity() {
    private val model by viewModels<SpeechViewModel>()
    private val controller
        get() = model.controller

    private val preferences = MutableStateFlow<Preferences?>(null)
    private var startWhenPreferencesReady = false

    private val microphone =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) start() else finish()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setResult(RESULT_CANCELED)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                PreferencesRepository(this@ActionRecognizeSpeechActivity).values.collect { settings
                    ->
                    preferences.value = settings
                    // Load stored feedback settings before starting, including after a cold launch.
                    if (!model.started || startWhenPreferencesReady) {
                        model.started = true
                        start()
                    }
                }
            }
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                model.result.collect { text ->
                    if (!text.isNullOrBlank()) {
                        val result =
                            Intent()
                                .putStringArrayListExtra(
                                    RecognizerIntent.EXTRA_RESULTS,
                                    arrayListOf(text),
                                )
                        val pending =
                            intent.getParcelableExtra(
                                RecognizerIntent.EXTRA_RESULTS_PENDINGINTENT,
                                PendingIntent::class.java,
                            )
                        if (pending != null) {
                            intent
                                .getBundleExtra(RecognizerIntent.EXTRA_RESULTS_PENDINGINTENT_BUNDLE)
                                ?.let(result::putExtras)
                            // Caller-supplied extras must not replace the actual transcript.
                            result.putStringArrayListExtra(
                                RecognizerIntent.EXTRA_RESULTS,
                                arrayListOf(text),
                            )
                            if (!model.pendingResultSent) {
                                model.pendingResultSent = true
                                runCatching {
                                    pending.send(
                                        this@ActionRecognizeSpeechActivity,
                                        RESULT_OK,
                                        result,
                                    )
                                }
                            }
                        } else {
                            setResult(RESULT_OK, result)
                        }
                        finish()
                    }
                }
            }
        }
        setContent {
            val state by controller.state.collectAsState()
            val settings by preferences.collectAsState()
            ListeningFeedback(state, settings ?: Preferences(sounds = false, haptics = false))
            TranscribroTheme {
                Surface {
                    Column(
                        Modifier.fillMaxWidth().padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(24.dp),
                    ) {
                        Text("Transcribro", style = MaterialTheme.typography.headlineSmall)
                        DictationControls(
                            state,
                            ::start,
                            controller::stop,
                            {
                                controller.cancel()
                                finish()
                            },
                        )
                        Text(
                            "English · Recognized on your device",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
            BackHandler {
                controller.cancel()
                finish()
            }
        }
    }

    private fun start() {
        val settings = preferences.value
        if (settings == null) {
            // A restored permission result can arrive before this activity's settings load.
            startWhenPreferencesReady = true
            return
        }
        startWhenPreferencesReady = false
        val language = intent.getStringExtra(RecognizerIntent.EXTRA_LANGUAGE)
        if (language != null && !language.equals("en", true) && !language.startsWith("en-", true)) {
            setResult(RecognizerIntent.RESULT_CLIENT_ERROR)
            finish()
            return
        }
        if (
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        ) {
            controller.start(
                continuous = false,
                liveText = settings.liveText,
                pauseMillis = settings.pauseLength.millis,
            )
        } else {
            microphone.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    override fun onStop() {
        if (!isChangingConfigurations) controller.cancel()
        super.onStop()
    }
}
