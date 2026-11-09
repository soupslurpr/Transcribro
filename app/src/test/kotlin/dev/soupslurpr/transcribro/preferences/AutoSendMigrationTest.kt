package dev.soupslurpr.transcribro.preferences

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences as StoredPreferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.preferencesOf
import androidx.datastore.preferences.core.stringPreferencesKey
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AutoSendMigrationTest {
    @get:Rule val directory = TemporaryFolder()

    private val legacyAutoSend = booleanPreferencesKey("AUTO_SEND_TRANSCRIPTION")
    private val autoSend = booleanPreferencesKey("AUTO_SEND_REIMAGINED")
    private val sendTiming = stringPreferencesKey("AUTO_SEND_TIMING")

    @Test
    fun enabledV7SettingBecomesPhraseSendingAndPersistsWithoutChangingOtherSettings() =
        runBlocking {
            val file = storeFile()
            val old =
                preferencesOf(
                    legacyAutoSend to true,
                    booleanPreferencesKey("AUTO_START_RECOGNITION") to true,
                    booleanPreferencesKey("AUTO_STOP_RECOGNITION") to true,
                    booleanPreferencesKey("AUTO_SWITCH_TO_PREVIOUS_INPUT_METHOD") to true,
                    // Removed appearance preferences can remain in an upgraded store.
                    booleanPreferencesKey("PITCH_BLACK_BACKGROUND") to true,
                )
            seed(file, old)
            withStore(file) { store ->
                val preferences = PreferencesRepository(store).values.first()
                assertTrue(preferences.autoSend)
                assertEquals(SendTiming.Phrase, preferences.sendTiming)
                assertTrue(preferences.autoStart)
                assertTrue(preferences.stopAtPause)
                assertTrue(preferences.switchBack)
                assertEquals(PauseLength.ThreeSeconds, preferences.pauseLength)
            }
            // A reader without any migration still sees the persisted upgrade.
            withStore(file, migrate = false) { store ->
                val preferences = PreferencesRepository(store).values.first()
                assertTrue(preferences.autoSend)
                assertEquals(SendTiming.Phrase, preferences.sendTiming)
            }
        }

    @Test
    fun disabledV7SettingAndFreshInstallBothKeepAutoSendOff() = runBlocking {
        for (old in listOf(preferencesOf(legacyAutoSend to false), emptyPreferences())) {
            val file = storeFile()
            seed(file, old)
            withStore(file) { store ->
                val preferences = PreferencesRepository(store).values.first()
                assertTrue(preferences.autoStart)
                assertFalse(preferences.autoSend)
                assertEquals(SendTiming.Session, preferences.sendTiming)
            }
        }
    }

    @Test
    fun storedAutomaticStartChoiceSurvivesUpgradeAndLaterChanges() = runBlocking {
        for (enabled in listOf(false, true)) {
            val file = storeFile()
            seed(
                file,
                preferencesOf(
                    legacyAutoSend to false,
                    booleanPreferencesKey("AUTO_START_RECOGNITION") to enabled,
                ),
            )
            withStore(file) { store ->
                val repository = PreferencesRepository(store)
                assertEquals(enabled, repository.values.first().autoStart)
                repository.set(Preference.AutoStart, !enabled)
            }
            withStore(file) { store ->
                val preferences = PreferencesRepository(store).values.first()
                assertEquals(!enabled, preferences.autoStart)
                assertFalse(preferences.autoSend)
            }
        }
    }

    @Test
    fun existingReimaginedChoiceWinsOverTheLegacySetting() = runBlocking {
        for (enabled in listOf(false, true)) {
            for (timing in listOf(null, SendTiming.Session, SendTiming.Phrase)) {
                val old =
                    preferencesOf(legacyAutoSend to !enabled, autoSend to enabled)
                        .toMutablePreferences()
                if (timing != null) old[sendTiming] = timing.name
                val file = storeFile()
                seed(file, old.toPreferences())
                withStore(file) { store ->
                    val preferences = PreferencesRepository(store).values.first()
                    assertEquals(enabled, preferences.autoSend)
                    assertEquals(timing ?: SendTiming.Session, preferences.sendTiming)
                }
            }
        }
    }

    @Test
    fun anAlreadyChosenTimingIsPreservedDuringOptInMigration() = runBlocking {
        val file = storeFile()
        seed(file, preferencesOf(legacyAutoSend to true, sendTiming to SendTiming.Session.name))
        withStore(file) { store ->
            val preferences = PreferencesRepository(store).values.first()
            assertTrue(preferences.autoSend)
            assertEquals(SendTiming.Session, preferences.sendTiming)
        }
    }

    @Test
    fun disablingAutoSendAfterUpgradingSurvivesRestartAndLaterReenabling() = runBlocking {
        val file = storeFile()
        seed(file, preferencesOf(legacyAutoSend to true))
        withStore(file) { store ->
            val repository = PreferencesRepository(store)
            assertTrue(repository.values.first().autoSend)
            repository.set(Preference.AutoSend, false)
        }
        withStore(file) { store ->
            val repository = PreferencesRepository(store)
            assertFalse(repository.values.first().autoSend)
            assertEquals(SendTiming.Phrase, repository.values.first().sendTiming)
            repository.set(Preference.AutoSend, true)
            assertTrue(repository.values.first().autoSend)
            assertEquals(SendTiming.Phrase, repository.values.first().sendTiming)
        }
    }

    private fun storeFile() = File(directory.newFolder(), "preferences.preferences_pb")

    private suspend fun seed(file: File, preferences: StoredPreferences) {
        withStore(file, migrate = false) { it.updateData { preferences } }
    }

    private suspend fun withStore(
        file: File,
        migrate: Boolean = true,
        block: suspend (DataStore<StoredPreferences>) -> Unit,
    ) {
        val job = SupervisorJob()
        val store =
            PreferenceDataStoreFactory.create(
                migrations = if (migrate) listOf(AutoSendMigration) else emptyList(),
                scope = CoroutineScope(Dispatchers.IO + job),
                produceFile = { file },
            )
        try {
            block(store)
        } finally {
            job.cancelAndJoin()
        }
    }
}
