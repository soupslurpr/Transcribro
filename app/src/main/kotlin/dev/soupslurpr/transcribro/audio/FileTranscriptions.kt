package dev.soupslurpr.transcribro.audio

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.soupslurpr.transcribro.FileTranscription
import dev.soupslurpr.transcribro.TextExport
import dev.soupslurpr.transcribro.engine.DictationPriority

/** One temporary result, shared by the launcher and service. Accessed only on the main thread. */
internal object FileTranscriptions {
    var current by mutableStateOf<FileTranscription?>(null)
        private set

    fun start(context: Context, uri: Uri) {
        val selected = FileTranscription(uri = uri, paused = DictationPriority.active.value)
        current = selected
        try {
            // The temporary read grant follows the service when the picker task closes.
            // START_NOT_STICKY and the in-memory identity prevent job recovery after process death.
            context.startForegroundService(
                Intent(context, FileTranscriptionService::class.java)
                    .apply { clipData = ClipData.newRawUri("Audio", uri) }
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    .putExtra(FileTranscriptionService.EXTRA_ID, selected.id)
            )
        } catch (_: Exception) {
            updateActive(selected.id) {
                it.copy(
                    active = false,
                    paused = false,
                    failed = true,
                    message = "Couldn't start transcription. Choose the file again to retry.",
                )
            }
        }
    }

    fun updateActive(id: String, change: (FileTranscription) -> FileTranscription) {
        current?.takeIf { it.id == id && it.active }?.let { current = change(it) }
    }

    fun stop(context: Context) {
        current?.let {
            updateActive(it.id) { file ->
                file.copy(
                    active = false,
                    paused = false,
                    message = "Transcription stopped. Completed text is kept.",
                )
            }
            requestStop(context, it.id)
        }
    }

    fun clear(context: Context) {
        val previous = current
        current = null
        previous?.takeIf { it.active }?.let { requestStop(context, it.id) }
    }

    private fun requestStop(context: Context, id: String) {
        // Queue cancellation after the start command. stopService() before Android
        // delivers a pending foreground start can crash the entire app.
        context.startService(
            Intent(context, FileTranscriptionService::class.java)
                .setAction(FileTranscriptionService.ACTION_STOP)
                .putExtra(FileTranscriptionService.EXTRA_ID, id)
        )
    }

    fun markPreserved(export: TextExport) {
        current
            ?.takeIf { it.id == export.fileId }
            ?.let {
                current = it.copy(preservedText = export.text)
            }
    }
}
