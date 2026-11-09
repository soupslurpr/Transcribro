package dev.soupslurpr.transcribro.preferences

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences as StoredPreferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import java.io.IOException
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map

private val Context.preferencesStore by
    preferencesDataStore(name = "preferences", produceMigrations = { listOf(AutoSendMigration) })

data class Preferences(
    val autoStart: Boolean = true,
    val stopAtPause: Boolean = false,
    val switchBack: Boolean = false,
    val liveText: Boolean = true,
    val autoSend: Boolean = false,
    val sendTiming: SendTiming = SendTiming.Session,
    val pauseLength: PauseLength = PauseLength.ThreeSeconds,
    val sounds: Boolean = true,
    val haptics: Boolean = true,
)

enum class SendTiming(val label: String) {
    Session("When dictation finishes"),
    Phrase("After each phrase"),
}

enum class PauseLength(val millis: Int, val label: String) {
    OneSecond(1_000, "1 second"),
    TwoSeconds(2_000, "2 seconds"),
    ThreeSeconds(3_000, "3 seconds"),
}

enum class Preference(val key: String) {
    AutoStart("AUTO_START_RECOGNITION"),
    StopAtPause("AUTO_STOP_RECOGNITION"),
    SwitchBack("AUTO_SWITCH_TO_PREVIOUS_INPUT_METHOD"),
    LiveText("LIVE_COMPOSING_TEXT"),
    AutoSend("AUTO_SEND_REIMAGINED"),
    Sounds("SOUND_FEEDBACK"),
    Haptics("HAPTIC_FEEDBACK"),
}

class PreferencesRepository internal constructor(private val store: DataStore<StoredPreferences>) {
    constructor(context: Context) : this(context.applicationContext.preferencesStore)

    val values =
        store.data
            .catch {
                if (it is IOException) emit(emptyPreferences()) else throw it
            }
            .map { data ->
                Preferences(
                    autoStart = data[booleanPreferencesKey(Preference.AutoStart.key)] ?: true,
                    stopAtPause = data[booleanPreferencesKey(Preference.StopAtPause.key)] ?: false,
                    switchBack = data[booleanPreferencesKey(Preference.SwitchBack.key)] ?: false,
                    liveText = data[booleanPreferencesKey(Preference.LiveText.key)] ?: true,
                    autoSend = data[booleanPreferencesKey(Preference.AutoSend.key)] ?: false,
                    sendTiming =
                        SendTiming.entries.firstOrNull {
                            it.name == data[stringPreferencesKey("AUTO_SEND_TIMING")]
                        } ?: SendTiming.Session,
                    pauseLength =
                        PauseLength.entries.firstOrNull {
                            it.name == data[stringPreferencesKey("SPEECH_PAUSE_LENGTH")]
                        } ?: PauseLength.ThreeSeconds,
                    sounds = data[booleanPreferencesKey(Preference.Sounds.key)] ?: true,
                    haptics = data[booleanPreferencesKey(Preference.Haptics.key)] ?: true,
                )
            }

    suspend fun set(preference: Preference, enabled: Boolean) {
        store.edit { it[booleanPreferencesKey(preference.key)] = enabled }
    }

    suspend fun setSendTiming(timing: SendTiming) {
        store.edit { it[stringPreferencesKey("AUTO_SEND_TIMING")] = timing.name }
    }

    suspend fun setPauseLength(length: PauseLength) {
        store.edit { it[stringPreferencesKey("SPEECH_PAUSE_LENGTH")] = length.name }
    }
}
