package dev.soupslurpr.transcribro.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.dictation.DictationPhase
import dev.soupslurpr.transcribro.dictation.DictationState

@Composable
fun VoiceLevel(state: DictationState, modifier: Modifier = Modifier) {
    val amplitude by
        animateFloatAsState(
            if (state.phase == DictationPhase.Listening) state.level else 0f,
            spring(stiffness = 450f),
            label = "Microphone level",
        )
    val color = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        val widths = size.width / 17f
        val envelope = listOf(.28f, .48f, .76f, .93f, 1f, .85f, .64f, .4f, .2f)
        envelope.forEachIndexed { index, weight ->
            val height = (4.dp.toPx() + amplitude * weight * (size.height - 4.dp.toPx()))
            drawRoundRect(
                color,
                Offset(index * widths * 2, (size.height - height) / 2),
                Size(widths, height),
                CornerRadius(widths / 2),
            )
        }
    }
}

@Composable
fun DictationControls(
    state: DictationState,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    compact: Boolean = false,
) {
    Column(
        modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(if (compact) 8.dp else 16.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text(
                    when (state.phase) {
                        DictationPhase.Ready -> "English · On device"
                        DictationPhase.Preparing -> "Preparing…"
                        DictationPhase.Listening -> "Listening"
                        DictationPhase.Finishing -> "Finishing…"
                    },
                    style =
                        if (state.active) MaterialTheme.typography.titleMedium
                        else MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
                if (!compact && state.active) {
                    Text(
                        "English · On device",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            VoiceLevel(state, Modifier.size(if (compact) 60.dp else 76.dp, 28.dp))
        }
        Button(
            onClick = { if (state.active) onStop() else onStart() },
            enabled = enabled && state.phase != DictationPhase.Finishing,
            shapes =
                ButtonDefaults.shapes(
                    shape = RoundedCornerShape(if (compact) 28.dp else 32.dp),
                    pressedShape = RoundedCornerShape(if (compact) 36.dp else 40.dp),
                ),
            modifier = Modifier.fillMaxWidth().heightIn(min = if (compact) 68.dp else 80.dp),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 18.dp),
        ) {
            if (
                state.phase == DictationPhase.Preparing || state.phase == DictationPhase.Finishing
            ) {
                CircularProgressIndicator(
                    Modifier.size(24.dp),
                    color = LocalContentColor.current,
                    strokeWidth = 2.dp,
                )
            } else {
                Icon(
                    painterResource(if (state.active) R.drawable.ic_stop else R.drawable.ic_mic),
                    null,
                    Modifier.size(28.dp),
                )
            }
            Spacer(Modifier.width(12.dp))
            Text(
                if (state.active) "Finish dictation" else "Start dictating",
                style = MaterialTheme.typography.titleLarge,
            )
        }
        if (state.active && !compact) TextButton(onClick = onCancel) { Text("Cancel dictation") }
        state.message?.let {
            Surface(
                color =
                    if (state.failure) MaterialTheme.colorScheme.errorContainer
                    else MaterialTheme.colorScheme.surfaceContainerHigh,
                shape = RoundedCornerShape(20.dp),
            ) {
                Text(
                    it,
                    Modifier.padding(16.dp).semantics { liveRegion = LiveRegionMode.Polite },
                    color =
                        if (state.failure) MaterialTheme.colorScheme.onErrorContainer
                        else MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
