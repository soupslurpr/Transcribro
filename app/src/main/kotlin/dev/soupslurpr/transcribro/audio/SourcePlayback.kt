package dev.soupslurpr.transcribro.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class PlaybackState(
    val ready: Boolean = false,
    val playing: Boolean = false,
    val positionMs: Long = 0,
    val durationMs: Long = 0,
    val error: String? = null,
)

/** Foreground-only source playback; URI, position, and player never go to saved state. */
class SourcePlayback(private val context: Context, private val scope: CoroutineScope) :
    AutoCloseable {
    private val mutableState = MutableStateFlow(PlaybackState())
    val state = mutableState.asStateFlow()
    private val audio = context.getSystemService(AudioManager::class.java)
    private val attributes =
        AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
    private val focus =
        AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(attributes)
            .setOnAudioFocusChangeListener { change ->
                if (change != AudioManager.AUDIOFOCUS_GAIN) pause()
            }
            .build()
    private var player: MediaPlayer? = null
    private var ticker: Job? = null
    private var loading: Job? = null

    fun load(uri: Uri) {
        close()
        val current = MediaPlayer()
        player = current
        fun fail() {
            if (player !== current) return
            close()
            mutableState.value = PlaybackState(error = "This audio couldn't be played.")
        }
        current.setOnPreparedListener {
            if (player === current)
                mutableState.value =
                    PlaybackState(
                        ready = true,
                        durationMs = current.duration.toLong().coerceAtLeast(0),
                    )
        }
        current.setOnCompletionListener {
            if (player === current) {
                pause()
                mutableState.value = state.value.copy(positionMs = state.value.durationMs)
            }
        }
        current.setOnErrorListener { _, _, _ ->
            fail()
            true
        }
        current.setAudioAttributes(attributes)
        loading = scope.launch {
            try {
                // A remote document provider may be slow or unavailable. Keep its
                // descriptor acquisition off the UI thread and close late results.
                withContext(Dispatchers.IO) {
                    checkNotNull(context.contentResolver.openAssetFileDescriptor(uri, "r")).use {
                        file ->
                        withContext(Dispatchers.Main.immediate) {
                            if (player === current) {
                                if (file.declaredLength < 0)
                                    current.setDataSource(file.fileDescriptor)
                                else
                                    current.setDataSource(
                                        file.fileDescriptor,
                                        file.startOffset,
                                        file.declaredLength,
                                    )
                                current.prepareAsync()
                            }
                        }
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                fail()
            }
        }
    }

    fun toggle() {
        if (state.value.playing) {
            pause()
            return
        }
        val current = player ?: return
        if (!state.value.ready) return
        if (audio.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            mutableState.value =
                state.value.copy(error = "Audio is in use. Try playing again shortly.")
            return
        }
        try {
            if (state.value.positionMs >= state.value.durationMs) seek(0)
            current.start()
            mutableState.value = state.value.copy(playing = true, error = null)
            ticker?.cancel()
            ticker = scope.launch {
                while (isActive && player === current) {
                    mutableState.value =
                        state.value.copy(positionMs = current.currentPosition.toLong())
                    delay(250)
                }
            }
        } catch (_: Exception) {
            pause()
            mutableState.value = state.value.copy(error = "This audio couldn't be played.")
        }
    }

    fun seek(positionMs: Long) {
        if (!state.value.ready) return
        val position = positionMs.coerceIn(0, state.value.durationMs)
        runCatching { player?.seekTo(position, MediaPlayer.SEEK_CLOSEST) }
        mutableState.value = state.value.copy(positionMs = position)
    }

    fun pause() {
        ticker?.cancel()
        ticker = null
        if (state.value.playing) runCatching { player?.pause() }
        mutableState.value = state.value.copy(playing = false)
        audio.abandonAudioFocusRequest(focus)
    }

    override fun close() {
        loading?.cancel()
        loading = null
        pause()
        val previous = player
        player = null
        previous?.release()
        mutableState.value = PlaybackState()
    }
}
