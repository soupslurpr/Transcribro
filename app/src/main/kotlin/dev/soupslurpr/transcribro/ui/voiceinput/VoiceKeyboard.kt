package dev.soupslurpr.transcribro.ui.voiceinput

import android.view.HapticFeedbackConstants
import android.view.ViewConfiguration
import android.view.inputmethod.EditorInfo
import androidx.annotation.DrawableRes
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.PressInteraction
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.dictation.DictationPhase
import dev.soupslurpr.transcribro.dictation.DictationState
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.SendTiming
import dev.soupslurpr.transcribro.ui.ListeningFeedback
import dev.soupslurpr.transcribro.ui.TranscribroAppIcon
import dev.soupslurpr.transcribro.ui.VoiceLevel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Compact top: status in the header, voice actions above direct correction controls. */
@Composable
internal fun VoiceKeyboard(
    state: DictationState,
    preferences: Preferences,
    passwordField: Boolean,
    hasSelection: Boolean,
    action: Int,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onCancel: () -> Unit,
    onUndo: () -> Unit,
    onRedo: () -> Unit,
    onClear: () -> Unit,
    onLeft: () -> Unit,
    onRight: () -> Unit,
    onSwitch: () -> Unit,
    onSettings: () -> Unit,
    onSpace: () -> Unit,
    onDelete: () -> Unit,
    onReturn: () -> Unit,
    onAction: () -> Unit,
) {
    val window = LocalWindowInfo.current.containerDpSize
    val wide = window.width >= 600.dp && window.width > window.height
    ListeningFeedback(state, preferences)
    Surface(color = MaterialTheme.colorScheme.surfaceContainer) {
        Column(
            Modifier.fillMaxWidth()
                // Android reports its IME navigation buttons as a caption bar.
                .windowInsetsPadding(
                    WindowInsets.displayCutout
                        .union(WindowInsets.navigationBars)
                        .union(WindowInsets.captionBar)
                )
                .heightIn(max = window.height * if (wide) 0.85f else 0.65f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 12.dp)
                .padding(top = 4.dp, bottom = 8.dp)
        ) {
            KeyboardHeader(
                state,
                preferences,
                action,
                wide,
                onSettings,
                onSwitch,
            )
            if (passwordField || state.message != null) {
                Text(
                    state.message ?: "Tap to dictate in this field. Automatic start is off.",
                    Modifier.padding(vertical = 8.dp).semantics {
                        liveRegion = LiveRegionMode.Polite
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color =
                        if (state.failure) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (wide) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(24.dp),
                ) {
                    Column(Modifier.weight(1f)) {
                        Row(
                            Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            StatusText(state, Modifier.weight(1f), prominent = true)
                            VoiceLevel(state, Modifier.size(76.dp, 24.dp))
                        }
                        VoiceActions(state, onStart, onStop, onCancel)
                    }
                    EditingRows(
                        hasSelection,
                        action,
                        preferences.haptics,
                        onUndo,
                        onRedo,
                        onClear,
                        onDelete,
                        onLeft,
                        onRight,
                        onSpace,
                        onReturn,
                        onAction,
                        Modifier.weight(1f),
                    )
                }
            } else {
                VoiceActions(state, onStart, onStop, onCancel, Modifier.padding(bottom = 12.dp))
                EditingRows(
                    hasSelection,
                    action,
                    preferences.haptics,
                    onUndo,
                    onRedo,
                    onClear,
                    onDelete,
                    onLeft,
                    onRight,
                    onSpace,
                    onReturn,
                    onAction,
                )
            }
        }
    }
}

@Composable
private fun KeyboardHeader(
    state: DictationState,
    preferences: Preferences,
    action: Int,
    wide: Boolean,
    onSettings: () -> Unit,
    onSwitch: () -> Unit,
) {
    val timing =
        when {
            action != EditorInfo.IME_ACTION_SEND -> "Unavailable in this field"
            preferences.sendTiming == SendTiming.Phrase -> "After each phrase"
            else -> "When finished"
        }
    Row(
        Modifier.fillMaxWidth().heightIn(min = 48.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (!wide) {
            if (preferences.autoSend) {
                TextButton(
                    onClick = onSettings,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = PaddingValues(0.dp),
                ) {
                    HeaderStatus(state, timing, Modifier.weight(1f))
                }
            } else HeaderStatus(state, null, Modifier.weight(1f))
        } else if (preferences.autoSend) {
            TextButton(
                onClick = onSettings,
                modifier = Modifier.weight(1f).semantics { stateDescription = timing },
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 2.dp),
            ) {
                Icon(painterResource(R.drawable.ic_send), null, Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text("Auto-send", style = MaterialTheme.typography.labelMedium)
                    Text(
                        timing,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            TranscribroAppIcon(18.dp, Modifier.padding(start = 4.dp))
            Text(
                "Transcribro",
                Modifier.weight(1f).padding(start = 8.dp),
                style = MaterialTheme.typography.labelLarge,
            )
        }
        IconButton(onClick = onSettings) {
            Icon(painterResource(R.drawable.ic_tune), "Voice input settings")
        }
        if (
            LocalDensity.current.fontScale <= 1.3f &&
                LocalWindowInfo.current.containerDpSize.width >= 360.dp
        ) {
            TextButton(
                onClick = onSwitch,
                modifier = Modifier.semantics { contentDescription = "Return to typing keyboard" },
            ) {
                Icon(painterResource(R.drawable.ic_keyboard), null)
                Spacer(Modifier.width(6.dp))
                Text("Typing", Modifier.clearAndSetSemantics {})
            }
        } else
            IconButton(onClick = onSwitch) {
                Icon(painterResource(R.drawable.ic_keyboard), "Return to typing keyboard")
            }
    }
}

@Composable
private fun HeaderStatus(state: DictationState, timing: String?, modifier: Modifier = Modifier) {
    Column(modifier.padding(horizontal = 4.dp, vertical = 8.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            StatusText(state, Modifier.weight(1f))
            VoiceLevel(state, Modifier.size(48.dp, 18.dp))
        }
        if (timing != null)
            Text(
                "Auto-send · $timing",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
    }
}

@Composable
private fun StatusText(
    state: DictationState,
    modifier: Modifier = Modifier,
    prominent: Boolean = false,
) {
    Text(
        when (state.phase) {
            DictationPhase.Ready -> if (state.failure) "Try again" else "Ready"
            DictationPhase.Preparing -> "Preparing"
            DictationPhase.Listening ->
                if (state.finishingPhrase) "Finishing phrase" else "Listening"
            DictationPhase.Finishing -> "Finishing"
        },
        modifier.semantics { liveRegion = LiveRegionMode.Polite },
        style =
            if (prominent) MaterialTheme.typography.titleLarge
            else MaterialTheme.typography.titleMedium,
        color = MaterialTheme.colorScheme.onSurface,
    )
}

@Composable
private fun VoiceActions(
    state: DictationState,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        TextButton(
            onClick = onCancel,
            enabled = state.active,
            modifier =
                Modifier.heightIn(min = 48.dp).semantics {
                    stateDescription = "Keeps completed words; discards the unfinished phrase"
                },
            contentPadding = PaddingValues(horizontal = 12.dp),
        ) {
            Text("Cancel")
        }
        Button(
            onClick = if (state.active) onStop else onStart,
            modifier =
                Modifier.weight(1f).heightIn(min = 64.dp).semantics {
                    contentDescription = if (state.active) "Finish dictation" else "Start dictating"
                },
            enabled = state.phase != DictationPhase.Finishing,
            shapes =
                ButtonDefaults.shapes(
                    shape = RoundedCornerShape(28.dp),
                    pressedShape = RoundedCornerShape(18.dp),
                ),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 14.dp),
        ) {
            Icon(
                painterResource(if (state.active) R.drawable.ic_stop else R.drawable.ic_mic),
                null,
                Modifier.size(26.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                if (state.active) "Finish" else "Start",
                Modifier.clearAndSetSemantics {},
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

@Composable
private fun EditingRows(
    hasSelection: Boolean,
    action: Int,
    haptics: Boolean,
    onUndo: () -> Unit,
    onRedo: () -> Unit,
    onClear: () -> Unit,
    onDelete: () -> Unit,
    onLeft: () -> Unit,
    onRight: () -> Unit,
    onSpace: () -> Unit,
    onReturn: () -> Unit,
    onAction: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            EditKey("Undo", R.drawable.ic_undo, onUndo, repeat = true, haptics = haptics)
            EditKey("Redo", R.drawable.ic_redo, onRedo, repeat = true, haptics = haptics)
            TextButton(
                onClick = onClear,
                modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(4.dp),
            ) {
                Text(
                    if (hasSelection) "Keep selected text" else "Clear field",
                    style = MaterialTheme.typography.labelMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            EditKey("Backspace", R.drawable.ic_backspace, onDelete, repeat = true, haptics = haptics)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            EditKey(
                "Move cursor left",
                R.drawable.ic_arrow_back,
                onLeft,
                repeat = true,
                haptics = haptics,
            )
            EditKey(
                "Move cursor right",
                R.drawable.ic_arrow_forward,
                onRight,
                repeat = true,
                haptics = haptics,
            )
            EditKey("Space", R.drawable.ic_space_bar, onSpace, Modifier.weight(1f))
            EditKey("Insert newline", R.drawable.ic_keyboard_return, onReturn)
            EditorActionKey(action, onAction)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun EditorActionKey(action: Int, onAction: () -> Unit) {
    val enabled =
        action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED
    val label = if (enabled) VoiceInput.actionLabel(action) else "No editor action"
    val icon =
        when (action) {
            EditorInfo.IME_ACTION_SEND -> R.drawable.ic_send
            EditorInfo.IME_ACTION_SEARCH -> R.drawable.ic_search
            EditorInfo.IME_ACTION_GO -> R.drawable.ic_arrow_forward
            EditorInfo.IME_ACTION_NEXT -> R.drawable.ic_keyboard_tab
            EditorInfo.IME_ACTION_PREVIOUS -> R.drawable.ic_keyboard_tab_rtl
            else -> R.drawable.ic_check
        }
    TooltipBox(
        positionProvider =
            TooltipDefaults.rememberTooltipPositionProvider(TooltipAnchorPosition.Above),
        tooltip = { PlainTooltip { Text(label) } },
        state = rememberTooltipState(),
        modifier = Modifier.size(48.dp),
    ) {
        EditKey(label, icon, onAction, Modifier.fillMaxSize(), enabled = enabled)
    }
}

/** Normal accessible click semantics plus pointer hold-to-repeat for editing controls. */
@Composable
private fun EditKey(
    label: String,
    @DrawableRes icon: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    repeat: Boolean = false,
    haptics: Boolean = false,
    enabled: Boolean = true,
) {
    val view = LocalView.current
    val interactions = remember { MutableInteractionSource() }
    val latestClick by rememberUpdatedState(onClick)
    var repeated by remember { mutableStateOf(false) }
    LaunchedEffect(interactions, repeat, enabled, haptics, view) {
        if (!repeat || !enabled) return@LaunchedEffect
        var job: Job? = null
        try {
            interactions.interactions.collect { interaction ->
                when (interaction) {
                    is PressInteraction.Press -> {
                        job?.cancel()
                        repeated = false
                        job = launch {
                            delay(ViewConfiguration.getLongPressTimeout().toLong())
                            if (haptics)
                                view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                            while (true) {
                                repeated = true
                                latestClick()
                                delay(ViewConfiguration.getKeyRepeatDelay().toLong())
                            }
                        }
                    }
                    is PressInteraction.Release -> job?.cancel()
                    is PressInteraction.Cancel -> {
                        job?.cancel()
                        repeated = false
                    }
                }
            }
        } finally {
            job?.cancel()
        }
    }
    FilledTonalButton(
        onClick = {
            if (!repeated) latestClick()
            repeated = false
        },
        modifier =
            modifier.widthIn(min = 48.dp).heightIn(min = 48.dp).semantics {
                contentDescription = label
            },
        shape = RoundedCornerShape(14.dp),
        interactionSource = interactions,
        enabled = enabled,
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
    ) {
        Icon(painterResource(icon), null, Modifier.size(22.dp))
    }
}
