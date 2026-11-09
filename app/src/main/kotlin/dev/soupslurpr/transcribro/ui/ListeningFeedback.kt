package dev.soupslurpr.transcribro.ui

import android.app.NotificationManager
import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.HapticFeedbackConstants
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import dev.soupslurpr.transcribro.dictation.DictationState
import dev.soupslurpr.transcribro.preferences.Preferences

/** Feedback follows the microphone's ready/end callbacks, not a request to start or finish. */
@Composable
internal fun ListeningFeedback(state: DictationState, preferences: Preferences) {
    val context = LocalContext.current.applicationContext
    val view = LocalView.current
    val tones = remember(context) { ListeningTones(context) }
    // Recreating an already-listening activity must not replay the ready cue.
    var wasListening by remember { mutableStateOf(state.listening) }
    DisposableEffect(tones) { onDispose(tones::close) }
    LaunchedEffect(state.listening) {
        if (state.listening != wasListening) {
            wasListening = state.listening
            if (preferences.haptics) {
                view.performHapticFeedback(
                    if (state.listening) HapticFeedbackConstants.GESTURE_START
                    else HapticFeedbackConstants.GESTURE_END
                )
            }
            if (preferences.sounds) tones.play(state.listening)
        }
    }
}

private class ListeningTones(private val context: Context) : AutoCloseable {
    private var generator: ToneGenerator? = null
    private var ending = false

    fun play(listening: Boolean) {
        if (!soundsAllowed()) return
        runCatching {
            val tones =
                generator ?: ToneGenerator(AudioManager.STREAM_SYSTEM, 55).also { generator = it }
            ending = !listening
            // Brief non-speech tones avoid a spoken prompt entering recognition. Never mute
            // capture around the ready cue: doing so could discard the user's first words.
            tones.startTone(
                if (listening) ToneGenerator.TONE_PROP_ACK else ToneGenerator.TONE_PROP_BEEP,
                60,
            )
        }
    }

    private fun soundsAllowed(): Boolean = runCatching {
        val audio = context.getSystemService(AudioManager::class.java)
        val notifications = context.getSystemService(NotificationManager::class.java)
        audio != null &&
            notifications != null &&
            audio.ringerMode == AudioManager.RINGER_MODE_NORMAL &&
            !audio.isStreamMute(AudioManager.STREAM_SYSTEM) &&
            audio.getStreamVolume(AudioManager.STREAM_SYSTEM) > 0 &&
            notifications.currentInterruptionFilter ==
                NotificationManager.INTERRUPTION_FILTER_ALL &&
            Settings.System.getInt(
                context.contentResolver,
                Settings.System.SOUND_EFFECTS_ENABLED,
                1,
            ) == 1
    }
        .getOrDefault(false)

    override fun close() {
        val tones = generator ?: return
        generator = null
        if (ending) {
            // Returning to the caller/typing keyboard must not truncate the short end cue.
            Handler(Looper.getMainLooper()).postDelayed({ tones.release() }, 100)
        } else {
            tones.release()
        }
    }
}
