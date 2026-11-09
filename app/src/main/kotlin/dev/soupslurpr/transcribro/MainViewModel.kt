package dev.soupslurpr.transcribro

import android.Manifest
import android.app.Application
import android.content.ComponentName
import android.content.pm.PackageManager
import android.media.AudioManager
import android.net.Uri
import android.os.SystemClock
import android.provider.Settings
import android.view.inputmethod.InputMethodManager
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.soupslurpr.transcribro.audio.FileTranscriptions
import dev.soupslurpr.transcribro.audio.SourcePlayback
import dev.soupslurpr.transcribro.dictation.DraftHistory
import dev.soupslurpr.transcribro.engine.Transcript
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.PreferencesRepository
import dev.soupslurpr.transcribro.ui.voiceinput.VoiceInput
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class InputReadiness(
    val keyboardEnabled: Boolean = false,
    val keyboardSelected: Boolean = false,
    val microphoneAllowed: Boolean = false,
    val microphoneBlocked: Boolean = false,
) {
    val ready
        get() = keyboardEnabled && microphoneAllowed && !microphoneBlocked
}

data class FileTranscription(
    val id: String = UUID.randomUUID().toString(),
    val uri: Uri,
    val name: String = "Audio file",
    val text: String = "",
    val active: Boolean = true,
    val paused: Boolean = false,
    val completedMs: Long = 0,
    val durationMs: Long = -1,
    val message: String? = null,
    val failed: Boolean = false,
    val preservedText: String = "",
) {
    val unpreserved
        get() = text.isNotEmpty() && text != preservedText
}

data class TextExport(val fileId: String, val text: String)

/** Temporary practice input and a view onto the process-local file result. */
class MainViewModel(application: Application) : AndroidViewModel(application) {
    val repository = PreferencesRepository(application)
    val preferences =
        repository.values.stateIn(viewModelScope, SharingStarted.Eagerly, Preferences())
    val playback = SourcePlayback(application, viewModelScope)
    var readiness by mutableStateOf(InputReadiness())
        private set

    var testInput by mutableStateOf(TextFieldValue())
        private set

    val file: FileTranscription?
        get() = FileTranscriptions.current

    var notice by mutableStateOf<String?>(null)
    var microphoneHelp by mutableStateOf(false)
    private var savingText by mutableStateOf<TextExport?>(null)
    private var writingText by mutableStateOf<TextExport?>(null)
    private var failedSaveFileId by mutableStateOf<String?>(null)
    val saving: Boolean
        get() = savingText != null || writingText != null

    val saveFailed: Boolean
        get() = failedSaveFileId != null && failedSaveFileId == file?.id

    private var history = DraftHistory()
    private var lastManualChange = 0L

    fun refreshReadiness() {
        val app = getApplication<Application>()
        val manager = app.getSystemService(InputMethodManager::class.java)
        val component = ComponentName(app, VoiceInput::class.java)
        readiness =
            InputReadiness(
                keyboardEnabled =
                    manager.enabledInputMethodList.any {
                        ComponentName(it.packageName, it.serviceName) == component
                    },
                keyboardSelected =
                    ComponentName.unflattenFromString(
                        Settings.Secure.getString(
                                app.contentResolver,
                                Settings.Secure.DEFAULT_INPUT_METHOD,
                            )
                            .orEmpty()
                    ) == component,
                microphoneAllowed =
                    app.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                        PackageManager.PERMISSION_GRANTED,
                microphoneBlocked = app.getSystemService(AudioManager::class.java).isMicrophoneMute,
            )
    }

    fun updateTestInput(value: TextFieldValue) {
        if (value.text.length > Transcript.MAX_LENGTH) return
        if (value.text != testInput.text) {
            val now = SystemClock.uptimeMillis()
            if (lastManualChange == 0L || now - lastManualChange > 750)
                history.checkpoint(testInput)
            lastManualChange = now
        } else if (value.selection != testInput.selection) {
            lastManualChange = 0
        }
        testInput = value
    }

    fun undoTestInput() {
        history.undo(testInput)?.let { testInput = it }
        lastManualChange = 0
    }

    fun redoTestInput() {
        history.redo(testInput)?.let { testInput = it }
        lastManualChange = 0
    }

    fun clearTestInput() {
        testInput = TextFieldValue()
        history = DraftHistory()
        lastManualChange = 0
    }

    fun importAudio(uri: Uri) {
        if (uri.scheme != "content") {
            notice = "Choose an audio file using the file picker."
            return
        }
        playback.close()
        failedSaveFileId = null
        FileTranscriptions.start(getApplication(), uri)
    }

    fun stopImport() = FileTranscriptions.stop(getApplication())

    fun clearFile() {
        playback.close()
        failedSaveFileId = null
        FileTranscriptions.clear(getApplication())
    }

    fun exportSnapshot(): TextExport? =
        file?.takeIf { it.text.isNotBlank() }?.let { TextExport(it.id, it.text) }

    fun markPreserved(export: TextExport) {
        FileTranscriptions.markPreserved(export)
    }

    /** Retry takes a fresh snapshot, including any words completed since the failed save. */
    fun prepareSave(): Boolean {
        if (saving) return false
        savingText = exportSnapshot()
        return savingText != null
    }

    fun saveDestination(uri: Uri?) {
        val export = savingText
        savingText = null
        // Cancelling the picker preserves the previous failure and all transcript text.
        if (uri != null && export != null) save(uri, export)
    }

    fun savePickerUnavailable() {
        savingText?.let { failedSaveFileId = it.fileId }
        savingText = null
    }

    fun dismissSaveFailure() {
        failedSaveFileId = null
    }

    fun save(uri: Uri, export: TextExport) {
        if (writingText != null) return
        writingText = export
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    checkNotNull(
                            getApplication<Application>()
                                .contentResolver
                                .openOutputStream(uri, "wt")
                        )
                        .bufferedWriter(Charsets.UTF_8)
                        .use { it.write(export.text) }
                }
                markPreserved(export)
                if (failedSaveFileId == export.fileId) failedSaveFileId = null
                if (file?.id == export.fileId) notice = "Saved."
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                if (file?.id == export.fileId) failedSaveFileId = export.fileId
            } finally {
                writingText = null
            }
        }
    }

    override fun onCleared() {
        playback.close()
    }
}
