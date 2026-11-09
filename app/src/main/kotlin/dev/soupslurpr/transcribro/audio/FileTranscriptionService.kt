package dev.soupslurpr.transcribro.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.provider.OpenableColumns
import androidx.core.net.toUri
import dev.soupslurpr.transcribro.FileTranscription
import dev.soupslurpr.transcribro.MainActivity
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.engine.DictationPriority
import dev.soupslurpr.transcribro.engine.Transcript
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** User-started media work; no microphone, persisted work request, or automatic restart. */
class FileTranscriptionService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var work: Job? = null
    private var priority: Job? = null
    private var activeId: String? = null
    private var lastNotification = 0L
    private lateinit var wakeLock: PowerManager.WakeLock
    private val notifications
        get() = getSystemService(NotificationManager::class.java)

    override fun onCreate() {
        super.onCreate()
        notifications.createNotificationChannel(
            NotificationChannel(CHANNEL, "Audio transcription", NotificationManager.IMPORTANCE_LOW)
        )
        wakeLock =
            getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:audio-transcription")
                .apply { setReferenceCounted(false) }
        // A picker result can be discarded before onStartCommand arrives. Even that
        // cancelled start must satisfy Android's foreground-service startup contract.
        try {
            startForeground(
                NOTIFICATION_ID,
                notification(
                    FileTranscriptions.current ?: FileTranscription(uri = Uri.EMPTY, active = false)
                ),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } catch (_: Exception) {
            FileTranscriptions.current?.let { selected ->
                FileTranscriptions.updateActive(selected.id) {
                    it.copy(
                        active = false,
                        paused = false,
                        failed = true,
                        message =
                            "Android couldn't start background transcription. Try again with Transcribro open.",
                    )
                }
            }
            stopSelf()
        }
    }

    override fun onBind(intent: Intent): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val selected = FileTranscriptions.current
        val requestedId = intent?.getStringExtra(EXTRA_ID)
        if (intent?.action == ACTION_STOP) {
            if (activeId == requestedId || selected?.id == requestedId) {
                requestedId?.let { id ->
                    FileTranscriptions.updateActive(id) {
                        it.copy(
                            active = false,
                            paused = false,
                            message = "Transcription stopped. Completed text is kept.",
                        )
                    }
                }
                work?.cancel()
                priority?.cancel()
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            } else if (activeId == null) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
            return START_NOT_STICKY
        }
        if (selected == null || !selected.active || selected.id != requestedId) {
            if (activeId == null) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
            return START_NOT_STICKY
        }
        activeId = selected.id
        work?.cancel()
        priority?.cancel()
        try {
            startForeground(
                NOTIFICATION_ID,
                notification(selected),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } catch (_: Exception) {
            FileTranscriptions.updateActive(selected.id) {
                it.copy(
                    active = false,
                    paused = false,
                    failed = true,
                    message =
                        "Android couldn't start background transcription. Try again with Transcribro open.",
                )
            }
            stopSelf(startId)
            return START_NOT_STICKY
        }
        priority = scope.launch {
            DictationPriority.active.collect { paused ->
                FileTranscriptions.updateActive(selected.id) { it.copy(paused = paused) }
                if (paused) releaseWakeLock() else holdWakeLock()
                updateNotification(force = true)
            }
        }
        work = scope.launch {
            try {
                val name =
                    withContext(Dispatchers.IO) {
                        runCatching {
                                contentResolver
                                    .query(
                                        selected.uri,
                                        arrayOf(OpenableColumns.DISPLAY_NAME),
                                        null,
                                        null,
                                        null,
                                    )
                                    ?.use { if (it.moveToFirst()) it.getString(0) else null }
                            }
                            .getOrNull()
                            ?.take(256)
                            ?.takeIf(String::isNotBlank) ?: "Audio file"
                    }
                FileTranscriptions.updateActive(selected.id) { it.copy(name = name) }
                withContext(Dispatchers.IO) {
                    checkNotNull(contentResolver.openAssetFileDescriptor(selected.uri, "r")).use {
                        descriptor ->
                        AudioFileTranscriber(this@FileTranscriptionService)
                            .transcribe(
                                descriptor,
                                onProgress = { elapsed, duration ->
                                    FileTranscriptions.updateActive(selected.id) {
                                        it.copy(completedMs = elapsed, durationMs = duration)
                                    }
                                    updateNotification()
                                },
                                onPhrase = { phrase ->
                                    FileTranscriptions.updateActive(selected.id) {
                                        it.copy(text = Transcript.join(it.text, phrase))
                                    }
                                },
                            )
                    }
                }
                FileTranscriptions.updateActive(selected.id) {
                    it.copy(
                        active = false,
                        paused = false,
                        message =
                            if (it.text.isBlank()) "No spoken words found in this file."
                            else "Transcription complete.",
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Exception) {
                FileTranscriptions.updateActive(selected.id) {
                    it.copy(
                        active = false,
                        paused = false,
                        failed = true,
                        message =
                            if (failure is java.io.IOException)
                                failure.message ?: "Couldn't read this audio file."
                            else "Couldn't open this audio file. Completed text is kept.",
                    )
                }
            } finally {
                if (activeId == selected.id && FileTranscriptions.current?.id == selected.id) {
                    priority?.cancel()
                    releaseWakeLock()
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf(startId)
                }
            }
        }
        return START_NOT_STICKY
    }

    private fun notification(file: FileTranscription): Notification {
        val open =
            PendingIntent.getActivity(
                this,
                0,
                Intent(this, MainActivity::class.java)
                    .setAction(ACTION_OPEN)
                    .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
        // The identity in the action prevents an old notification from stopping a replacement file.
        val stop =
            PendingIntent.getService(
                this,
                file.id.hashCode(),
                Intent(this, FileTranscriptionService::class.java)
                    .setAction(ACTION_STOP)
                    .setData("transcribro:stop/${file.id}".toUri())
                    .putExtra(EXTRA_ID, file.id),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_audio_file)
            .setContentTitle(if (file.paused) "Paused while dictating" else "Transcribing audio")
            .setContentText(
                if (file.paused) "Your file will continue when dictation ends."
                else "Tap to view the temporary result."
            )
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setShowWhen(false)
            .setCategory(Notification.CATEGORY_PROGRESS)
            .setVisibility(Notification.VISIBILITY_PRIVATE)
            .setProgress(
                1000,
                if (file.durationMs > 0)
                    (file.completedMs * 1000 / file.durationMs).toInt().coerceIn(0, 1000)
                else 0,
                file.durationMs <= 0 && !file.paused,
            )
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun updateNotification(force: Boolean = false) {
        val file = FileTranscriptions.current?.takeIf { it.active && it.id == activeId } ?: return
        val now = SystemClock.elapsedRealtime()
        if (force || now - lastNotification >= 1000) {
            notifications.notify(NOTIFICATION_ID, notification(file))
            lastNotification = now
        }
    }

    private fun holdWakeLock() {
        if (!wakeLock.isHeld) wakeLock.acquire(6 * 60 * 60 * 1000L)
    }

    private fun releaseWakeLock() {
        if (wakeLock.isHeld) wakeLock.release()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        activeId?.let { id ->
            FileTranscriptions.updateActive(id) {
                it.copy(
                    active = false,
                    paused = false,
                    message =
                        "Android's background processing limit was reached. Completed text is kept.",
                )
            }
        }
        work?.cancel()
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        activeId?.let { id ->
            FileTranscriptions.updateActive(id) {
                it.copy(
                    active = false,
                    paused = false,
                    message = "Transcription stopped. Completed text is kept.",
                )
            }
        }
        activeId = null
        scope.cancel()
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    companion object {
        const val ACTION_OPEN = "dev.soupslurpr.transcribro.OPEN_TRANSCRIPTION"
        internal const val ACTION_STOP = "dev.soupslurpr.transcribro.STOP_TRANSCRIPTION"
        internal const val EXTRA_ID = "transcription_id"
        internal const val NOTIFICATION_ID = 1
        private const val CHANNEL = "audio-transcription"
    }
}
