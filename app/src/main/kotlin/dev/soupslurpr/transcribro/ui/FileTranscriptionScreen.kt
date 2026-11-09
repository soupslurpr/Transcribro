package dev.soupslurpr.transcribro.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.disabled
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.setProgress
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.FileTranscription
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.audio.PlaybackState

@Composable
internal fun FileTranscriptionScreen(
    file: FileTranscription?,
    playback: PlaybackState,
    onChoose: () -> Unit,
    onStop: () -> Unit,
    onDiscard: () -> Unit,
    onPlay: () -> Unit,
    onSeek: (Long) -> Unit,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onSave: () -> Unit,
    saveFailed: Boolean = false,
    saving: Boolean = false,
    onDismissSaveFailure: () -> Unit = {},
) {
    BoxWithConstraints(Modifier.widthIn(max = 800.dp).fillMaxSize()) {
        val recoveryMaxHeight = maxHeight / 2
        Column(Modifier.fillMaxSize()) {
            Column(
                Modifier.weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
            ) {
                if (file == null) {
                    Icon(
                        painterResource(R.drawable.ic_audio_file),
                        null,
                        Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.primary,
                    )
                    Text("From audio to words.", style = MaterialTheme.typography.headlineMedium)
                    Text(
                        "Choose a voice message or a recording. Transcribe it on your device, then use the words in BeauTyXT or another app."
                    )
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = onChoose,
                            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                        ) {
                            Text("Choose audio")
                        }
                        Text(
                            "Up to 30 minutes per file",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Text(
                        "No microphone access is needed. Transcriptions stay here temporarily; copy, share, or save anything you want to keep.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    Text(
                        file.name,
                        style = MaterialTheme.typography.titleLarge,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Surface(
                        shape = MaterialTheme.shapes.extraLarge,
                        color = MaterialTheme.colorScheme.surfaceContainer,
                    ) {
                        Column(
                            Modifier.fillMaxWidth().padding(20.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            if (file.active) {
                                Text(
                                    if (file.paused) "Paused while dictating"
                                    else "Transcribing audio",
                                    Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                                    style = MaterialTheme.typography.titleMedium,
                                )
                                if (file.durationMs > 0)
                                    LinearProgressIndicator(
                                        progress = {
                                            (file.completedMs.toFloat() / file.durationMs).coerceIn(
                                                0f,
                                                1f,
                                            )
                                        },
                                        modifier = Modifier.fillMaxWidth(),
                                    )
                                else LinearProgressIndicator(Modifier.fillMaxWidth())
                                Row(
                                    Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        if (file.completedMs == 0L) "Preparing…"
                                        else
                                            timestamp(file.completedMs) +
                                                if (file.durationMs > 0)
                                                    " / ${timestamp(file.durationMs)} processed"
                                                else " processed",
                                        Modifier.weight(1f),
                                        style = MaterialTheme.typography.bodyMedium,
                                    )
                                    TextButton(onClick = onStop) { Text("Stop") }
                                }
                                Text(
                                    if (file.paused) "Your file will continue when dictation ends."
                                    else
                                        "You can leave Transcribro or lock the screen. Stopping keeps completed text.",
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            } else {
                                Text(
                                    file.message.orEmpty(),
                                    Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                                    style = MaterialTheme.typography.bodyMedium,
                                    color =
                                        if (file.failed) MaterialTheme.colorScheme.error
                                        else MaterialTheme.colorScheme.onSurface,
                                )
                            }
                            SourcePlaybackControls(playback, onPlay, onSeek)
                        }
                    }
                    if (file.text.isNotEmpty()) {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            Text(
                                "Transcription",
                                Modifier.semantics { heading() },
                                style = MaterialTheme.typography.titleMedium,
                            )
                            SelectionContainer {
                                Text(file.text, style = MaterialTheme.typography.bodyLarge)
                            }
                        }
                        Text(
                            "To edit, share or save these words to BeauTyXT or another writing app.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onChoose) { Text("Choose another file") }
                        TextButton(onClick = onDiscard) { Text("Discard transcription") }
                    }
                }
            }
            if (!file?.text.isNullOrEmpty()) {
                FileExportActions(
                    onCopy,
                    onShare,
                    onSave,
                    saveFailed,
                    saving,
                    onDismissSaveFailure,
                    recoveryMaxHeight,
                )
            }
        }
    }
}

/** Keeps export actions available without scrolling through the transcript. */
@Composable
private fun FileExportActions(
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onSave: () -> Unit,
    saveFailed: Boolean,
    saving: Boolean,
    onDismissSaveFailure: () -> Unit,
    recoveryMaxHeight: Dp,
) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainerLow) {
        Column {
            HorizontalDivider()
            if (saveFailed) {
                Surface(color = MaterialTheme.colorScheme.errorContainer) {
                    Column(
                        Modifier.fillMaxWidth()
                            .heightIn(max = recoveryMaxHeight)
                            .verticalScroll(rememberScrollState())
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(
                            "Couldn't save the file. Your transcription is still here.",
                            Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            FilledTonalButton(
                                onClick = onSave,
                                enabled = !saving,
                                shapes = ButtonDefaults.shapes(),
                            ) {
                                Text("Try saving again")
                            }
                            TextButton(onClick = onDismissSaveFailure) { Text("Dismiss") }
                        }
                    }
                }
            }
            BoxWithConstraints(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)
            ) {
                val fontScale = LocalDensity.current.fontScale
                val showIcons = fontScale <= 1.3f && maxWidth >= 320.dp * maxOf(1f, fontScale)
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    listOf(
                            Triple("Copy", R.drawable.ic_content_copy, onCopy),
                            Triple("Share", R.drawable.ic_share, onShare),
                            Triple("Save", R.drawable.ic_download, onSave),
                        )
                        .forEach { (label, icon, action) ->
                            FilledTonalButton(
                                onClick = action,
                                enabled = label != "Save" || !saving,
                                modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                                contentPadding =
                                    PaddingValues(horizontal = 12.dp, vertical = 12.dp),
                                shapes = ButtonDefaults.shapes(),
                            ) {
                                if (showIcons) {
                                    Icon(painterResource(icon), null, Modifier.size(18.dp))
                                    Spacer(Modifier.width(8.dp))
                                }
                                Text(label)
                            }
                        }
                }
            }
        }
    }
}

@Composable
private fun SourcePlaybackControls(
    state: PlaybackState,
    onPlay: () -> Unit,
    onSeek: (Long) -> Unit,
) {
    var seeking by remember { mutableStateOf<Float?>(null) }
    val canSeek = state.ready && state.durationMs > 0
    val position = seeking ?: state.positionMs.toFloat()
    val range = 0f..state.durationMs.toFloat().coerceAtLeast(1f)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        FilledIconButton(onClick = onPlay, enabled = state.ready, modifier = Modifier.size(48.dp)) {
            Icon(
                painterResource(
                    if (state.playing) R.drawable.ic_pause else R.drawable.ic_play_arrow
                ),
                if (state.playing) "Pause audio" else "Play audio",
            )
        }
        Column(Modifier.weight(1f)) {
            Text("Source audio", style = MaterialTheme.typography.labelLarge)
            Slider(
                value = position,
                onValueChange = { seeking = it },
                onValueChangeFinished = {
                    seeking?.let { onSeek(it.toLong()) }
                    seeking = null
                },
                enabled = canSeek,
                valueRange = range,
                // Keep the name, time, and adjustment on one accessibility stop.
                modifier =
                    Modifier.clearAndSetSemantics {
                        contentDescription = "Audio position"
                        stateDescription =
                            "${timestamp(position.toLong())} of ${timestamp(state.durationMs)}"
                        progressBarRangeInfo = ProgressBarRangeInfo(position, range)
                        if (!canSeek) disabled()
                        setProgress { target ->
                            if (!canSeek || !target.isFinite()) false
                            else {
                                val next = target.coerceIn(range).toLong()
                                if (next == state.positionMs) false
                                else {
                                    seeking = null
                                    onSeek(next)
                                    true
                                }
                            }
                        }
                    },
            )
            Text(
                if (state.ready)
                    "${timestamp(seeking?.toLong() ?: state.positionMs)} / ${timestamp(state.durationMs)}"
                else if (state.error != null) "Playback unavailable" else "Preparing playback…",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
    state.error?.let {
        Text(
            it,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error,
        )
    }
}

private fun timestamp(ms: Long) = "${ms / 60_000}:${((ms / 1000) % 60).toString().padStart(2, '0')}"
