package dev.soupslurpr.transcribro.preferences

import androidx.datastore.core.DataMigration
import androidx.datastore.preferences.core.Preferences as StoredPreferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey

/** Preserve v7's explicit opt-in and pause-to-send workflow on a same-package upgrade. */
internal object AutoSendMigration : DataMigration<StoredPreferences> {
    private val legacyAutoSend = booleanPreferencesKey("AUTO_SEND_TRANSCRIPTION")
    private val autoSend = booleanPreferencesKey(Preference.AutoSend.key)
    private val sendTiming = stringPreferencesKey("AUTO_SEND_TIMING")

    override suspend fun shouldMigrate(currentData: StoredPreferences): Boolean =
        currentData[legacyAutoSend] != null

    override suspend fun migrate(currentData: StoredPreferences): StoredPreferences {
        val enabled = currentData[legacyAutoSend] ?: return currentData
        return currentData
            .toMutablePreferences()
            .apply {
                if (currentData[autoSend] == null) {
                    this[autoSend] = enabled
                    if (enabled && currentData[sendTiming] == null) {
                        this[sendTiming] = SendTiming.Phrase.name
                    }
                }
                remove(legacyAutoSend)
            }
            .toPreferences()
    }

    override suspend fun cleanUp() = Unit
}
