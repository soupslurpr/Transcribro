package dev.soupslurpr.transcribro

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import android.view.inputmethod.InputMethodManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.lifecycleScope
import dev.soupslurpr.transcribro.audio.FileTranscriptionService
import dev.soupslurpr.transcribro.ui.AppScreen
import dev.soupslurpr.transcribro.ui.MicrophonePermissionHelp
import dev.soupslurpr.transcribro.ui.TranscribroTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val model by viewModels<MainViewModel>()
    private var fileRequest by mutableIntStateOf(0)
    private var settingsRequest by mutableIntStateOf(0)
    private val notifications =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted)
                model.notice =
                    "Transcription can continue in the background. Open Transcribro to view progress or stop it."
        }
    private val microphone =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            model.refreshReadiness()
            if (!granted) model.microphoneHelp = true
        }
    private val openAudio =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                model.importAudio(uri)
                if (
                    checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
                        PackageManager.PERMISSION_GRANTED
                )
                    notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    private val save =
        registerForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
            model.saveDestination(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) handleDestination(intent)
        enableEdgeToEdge()
        setContent {
            TranscribroTheme {
                AppScreen(
                    model = model,
                    settingsRequest = settingsRequest,
                    fileRequest = fileRequest,
                    onOpenAudio = { openAudio.launch(arrayOf("audio/*")) },
                    onCopy = {
                        model.exportSnapshot()?.let { export ->
                            getSystemService(ClipboardManager::class.java)
                                .setPrimaryClip(ClipData.newPlainText("Transcription", export.text))
                            model.markPreserved(export)
                        }
                    },
                    onShare = {
                        model.exportSnapshot()?.let { export ->
                            model.playback.pause()
                            runCatching {
                                startActivity(
                                    Intent.createChooser(
                                        Intent(Intent.ACTION_SEND)
                                            .setType("text/plain")
                                            .putExtra(Intent.EXTRA_TEXT, export.text),
                                        null,
                                    )
                                )
                            }
                                .onFailure { model.notice = "No app is available to share text." }
                        }
                    },
                    onSave = {
                        if (model.prepareSave()) {
                            runCatching { save.launch("Transcription.txt") }
                                .onFailure { model.savePickerUnavailable() }
                        }
                    },
                    onEnableKeyboard = { openSettings(Settings.ACTION_INPUT_METHOD_SETTINGS) },
                    onChooseKeyboard = {
                        getSystemService(InputMethodManager::class.java).showInputMethodPicker()
                    },
                    onMicrophone = {
                        if (model.readiness.microphoneBlocked)
                            openSettings(Settings.ACTION_PRIVACY_SETTINGS)
                        else microphone.launch(Manifest.permission.RECORD_AUDIO)
                    },
                    // Android 17 routes VOICE_INPUT_SETTINGS to Assistant. The voice
                    // service chooser is under Language & region > Speech instead.
                    onVoiceSettings = { openSettings(Settings.ACTION_LOCALE_SETTINGS) },
                    setPreference = { key, enabled ->
                        lifecycleScope.launch {
                            runCatching { model.repository.set(key, enabled) }
                                .onFailure { model.notice = "Couldn't save that setting." }
                        }
                    },
                    setSendTiming = { timing ->
                        lifecycleScope.launch {
                            runCatching { model.repository.setSendTiming(timing) }
                                .onFailure { model.notice = "Couldn't save that setting." }
                        }
                    },
                    setPauseLength = { length ->
                        lifecycleScope.launch {
                            runCatching { model.repository.setPauseLength(length) }
                                .onFailure { model.notice = "Couldn't save that setting." }
                        }
                    },
                )
                if (model.microphoneHelp) MicrophonePermissionHelp { model.microphoneHelp = false }
            }
        }
    }

    private fun openSettings(action: String) {
        runCatching { startActivity(Intent(action)) }
            .onFailure { model.notice = "Couldn't open Android's settings." }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleDestination(intent)
    }

    private fun handleDestination(intent: Intent) {
        if (intent.component?.className == "dev.soupslurpr.transcribro.SettingsActivity") {
            settingsRequest++
            return
        }
        when (intent.action) {
            FileTranscriptionService.ACTION_OPEN -> fileRequest++
            Intent.ACTION_APPLICATION_PREFERENCES -> settingsRequest++
        }
    }

    override fun onResume() {
        super.onResume()
        model.refreshReadiness()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) model.refreshReadiness()
    }

    override fun onStop() {
        model.playback.pause()
        super.onStop()
    }
}
