package dev.soupslurpr.transcribro.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.preferences.PauseLength
import dev.soupslurpr.transcribro.preferences.Preference
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.SendTiming

@Composable
internal fun SettingsScreen(
    preferences: Preferences,
    setPreference: (Preference, Boolean) -> Unit,
    onEnableKeyboard: () -> Unit,
    onVoiceSettings: () -> Unit,
    setSendTiming: (SendTiming) -> Unit,
    setPauseLength: (PauseLength) -> Unit,
) {
    Column(
        Modifier.widthIn(max = 680.dp)
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        SettingsGroup("Voice keyboard") {
            PreferenceRow(
                "Live text",
                "Show words while speaking. Unfinished words may change. Turn off for apps that work better with completed phrases.",
                preferences.liveText,
            ) {
                setPreference(Preference.LiveText, it)
            }
            PreferenceRow(
                "Start automatically",
                "Begin listening when the voice keyboard opens. Always off in password fields.",
                preferences.autoStart,
            ) {
                setPreference(Preference.AutoStart, it)
            }
            PreferenceRow(
                "Finish at a pause",
                "Finish after the chosen pause instead of listening for the next phrase.",
                preferences.stopAtPause,
            ) {
                setPreference(Preference.StopAtPause, it)
            }
            PreferenceRow(
                "Return to typing",
                "Switch back to your previous keyboard after a successful dictation.",
                preferences.switchBack,
            ) {
                setPreference(Preference.SwitchBack, it)
            }
        }
        SettingsGroup("Pause length") {
            SingleChoiceButtons(
                PauseLength.entries.map { length ->
                    SingleChoiceOption(
                        label = length.label,
                        selected = preferences.pauseLength == length,
                        onClick = { setPauseLength(length) },
                    )
                }
            )
            Text(
                "Wait this long before completing a phrase. Also used by Finish at a pause and sending after each phrase.",
                Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        SettingsGroup("Automatic sending") {
            PreferenceRow(
                "Send automatically",
                "Send completed speech without another tap in fields that offer Send.",
                preferences.autoSend,
            ) {
                setPreference(Preference.AutoSend, it)
            }
            SingleChoiceButtons(
                SendTiming.entries.map { timing ->
                    SingleChoiceOption(
                        label = timing.label,
                        selected = preferences.sendTiming == timing,
                        enabled = preferences.autoSend,
                        onClick = { setSendTiming(timing) },
                    )
                }
            )
            Text(
                if (preferences.sendTiming == SendTiming.Phrase)
                    "Pauses divide speech into separate messages. Listening continues unless Finish at a pause is also enabled."
                else "Only completed speech is sent. Cancelling or an interruption does not send.",
                Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        SettingsGroup("Feedback") {
            PreferenceRow(
                "Sounds",
                "Hear when listening starts and stops. Respects system sound settings.",
                preferences.sounds,
            ) {
                setPreference(Preference.Sounds, it)
            }
            PreferenceRow(
                "Haptic feedback",
                "Feel when listening starts and stops, and when keys begin repeating.",
                preferences.haptics,
            ) {
                setPreference(Preference.Haptics, it)
            }
        }
        SettingsGroup("Android integration") {
            SettingsLink("Keyboard settings", R.drawable.ic_keyboard, onEnableKeyboard)
            Text(
                "Some apps use Android's voice input service separately from your keyboard. In Language & region, open Speech, then Voice input, and choose Transcribro.",
                Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            SettingsLink("Language & region settings", R.drawable.ic_mic, onVoiceSettings)
        }
    }
}

@Composable
private fun SettingsGroup(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            title,
            Modifier.padding(start = 16.dp).semantics { heading() },
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.primary,
        )
        Column(
            Modifier.clip(MaterialTheme.shapes.extraLarge),
            verticalArrangement = Arrangement.spacedBy(3.dp),
            content = content,
        )
    }
}

@Composable
private fun PreferenceRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainerLow) {
        Row(
            Modifier.fillMaxWidth()
                .toggleable(checked, role = Role.Switch, onValueChange = onChange)
                .padding(20.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked, onCheckedChange = null)
        }
    }
}

@Composable
private fun SettingsLink(title: String, @DrawableRes icon: Int, onClick: () -> Unit) {
    Surface(onClick = onClick, color = MaterialTheme.colorScheme.surfaceContainerLow) {
        Row(
            Modifier.fillMaxWidth().padding(20.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(painterResource(icon), null, tint = MaterialTheme.colorScheme.primary)
            Text(title, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
            Icon(painterResource(R.drawable.ic_arrow_forward), null, Modifier.size(20.dp))
        }
    }
}
