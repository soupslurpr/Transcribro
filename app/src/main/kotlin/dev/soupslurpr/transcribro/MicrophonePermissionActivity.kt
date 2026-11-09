package dev.soupslurpr.transcribro

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import dev.soupslurpr.transcribro.ui.MicrophonePermissionHelp
import dev.soupslurpr.transcribro.ui.TranscribroTheme

/** The keyboard cannot show a runtime permission dialog itself. Never starts recording. */
class MicrophonePermissionActivity : ComponentActivity() {
    private var denied by mutableStateOf(false)
    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) finish() else denied = true
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        denied = savedInstanceState?.getBoolean("denied") ?: false
        setContent {
            TranscribroTheme {
                if (denied) MicrophonePermissionHelp(::finish)
            }
        }
        if (savedInstanceState == null) permission.launch(Manifest.permission.RECORD_AUDIO)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("denied", denied)
        super.onSaveInstanceState(outState)
    }
}
