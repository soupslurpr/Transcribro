package dev.soupslurpr.transcribro.preferences

import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey

/** Preference pairs, the first is the preference key, and the second is the default value. */
data class PreferencesUiState(
    /** Pitch black background. */
    val pitchBlackBackground: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("PITCH_BLACK_BACKGROUND")),
        mutableStateOf(false)
    ),

    /** Whether the user has accepted the privacy policy and license. */
    val acceptedPrivacyPolicyAndLicense: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("ACCEPTED_PRIVACY_POLICY_AND_LICENSE_V0.3.0")),
        mutableStateOf(false)
    ),

    /** Whether to automatically switch to the previous input method when the keyboard is done transcribing */
    val autoSwitchToPreviousInputMethod: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_SWITCH_TO_PREVIOUS_INPUT_METHOD")),
        mutableStateOf(false)
    ),

    /** Whether to automatically stop recognition when speech stops being detected. */
    val autoStopRecognition: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_STOP_RECOGNITION")),
        mutableStateOf(false)
    ),

    /** Whether to automatically start recognition when switching from another input method. */
    val autoStartRecognition: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_START_RECOGNITION")),
        mutableStateOf(true)
    ),

    /** Whether to automatically send transcription when speech stops being detected. */
    val autoSendTranscription: Pair<Preferences.Key<Boolean>, MutableState<Boolean>> = Pair(
        (booleanPreferencesKey("AUTO_SEND_TRANSCRIPTION")),
        mutableStateOf(false)
    ),

    /** API URL for the service. */
    val apiUrl: Pair<Preferences.Key<String>, MutableState<String>> = Pair(
        stringPreferencesKey("API_URL"),
        mutableStateOf("http://tower.lan:9965")
    ),

    /** Model Name for the transcription. */
    val modelName: Pair<Preferences.Key<String>, MutableState<String>> = Pair(
        stringPreferencesKey("MODEL_NAME"),
        mutableStateOf("ggml-faster-whisper-large-v3")
    )
)
