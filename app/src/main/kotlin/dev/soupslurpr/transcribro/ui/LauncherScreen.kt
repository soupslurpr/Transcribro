package dev.soupslurpr.transcribro.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.FileTranscription
import dev.soupslurpr.transcribro.InputReadiness
import dev.soupslurpr.transcribro.R

@Composable
internal fun LauncherScreen(
    readiness: InputReadiness,
    file: FileTranscription?,
    onTry: () -> Unit,
    onFile: () -> Unit,
    onSettings: () -> Unit,
    onAbout: () -> Unit,
    onEnableKeyboard: () -> Unit,
    onChooseKeyboard: () -> Unit,
    onMicrophone: () -> Unit,
) {
    val actions: @Composable (Modifier) -> Unit = { modifier ->
        Column(
            modifier.widthIn(max = 520.dp).fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            HomeActions(file, onTry, onFile)
            VoiceInputStatus(readiness, onEnableKeyboard, onChooseKeyboard, onMicrophone)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally)
            ) {
                TextButton(onClick = onSettings) { Text("Settings") }
                TextButton(onClick = onAbout) { Text("About") }
            }
        }
    }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val viewportHeight = maxHeight
        val wide =
            maxWidth >= 680.dp && maxWidth > maxHeight && LocalDensity.current.fontScale <= 1.3f
        if (wide) {
            Row(
                Modifier.align(Alignment.Center)
                    .widthIn(max = 960.dp)
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .heightIn(min = viewportHeight)
                    .padding(horizontal = 24.dp, vertical = 32.dp),
                horizontalArrangement = Arrangement.spacedBy(40.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                HomeIdentity(Modifier.weight(0.8f))
                actions(Modifier.weight(1.2f))
            }
        } else {
            Column(
                Modifier.fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .heightIn(min = viewportHeight)
                    .padding(horizontal = 24.dp)
                    .padding(top = 24.dp, bottom = 12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                HomeIdentity(Modifier.widthIn(max = 520.dp).fillMaxWidth())
                Spacer(Modifier.height(40.dp))
                actions(Modifier)
            }
        }
    }
}

@Composable
private fun HomeIdentity(modifier: Modifier = Modifier) {
    Column(
        modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        TranscribroAppIcon(80.dp)
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "Transcribro",
                Modifier.semantics { heading() },
                style = MaterialTheme.typography.displaySmall,
                textAlign = TextAlign.Center,
            )
            Text(
                "Private voice input & transcription",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

private data class HomeAction(
    val label: String,
    @DrawableRes val icon: Int,
    val onClick: () -> Unit,
    val primary: Boolean = false,
)

@Composable
private fun HomeActions(file: FileTranscription?, onTry: () -> Unit, onFile: () -> Unit) {
    val actions =
        listOf(
            HomeAction("Try dictation", R.drawable.ic_mic, onTry, primary = true),
            HomeAction(
                when {
                    file == null -> "Transcribe audio"
                    file.active -> "View transcription progress"
                    else -> "Open transcription"
                },
                R.drawable.ic_audio_file,
                onFile,
            ),
        )
    val interactions = remember { List(2) { MutableInteractionSource() } }
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = ButtonDefaults.textStyleFor(64.dp)
    val widestLabel = actions.maxOf {
        textMeasurer
            .measure(it.label, style = labelStyle, maxLines = 1, softWrap = false)
            .size
            .width
    }
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val labelSpace =
            with(LocalDensity.current) {
                // Account for the icon, spacing, padding, and the pressed-width animation.
                (constraints.maxWidth - 8.dp.roundToPx()) / 2 -
                    24.dp.roundToPx() -
                    12.dp.roundToPx() -
                    16.dp.roundToPx() -
                    8.dp.roundToPx()
            }
        if (widestLabel <= labelSpace) {
            ButtonGroup(
                overflowIndicator = { ButtonGroupDefaults.OverflowIndicator(menuState = it) },
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                actions.forEachIndexed { index, action ->
                    customItem(
                        buttonGroupContent = {
                            HomeActionButton(
                                action,
                                interactionSource = interactions[index],
                                modifier =
                                    Modifier.weight(1f)
                                        .animateWidth(interactions[index], compressionLimit = 8.dp)
                                        .heightIn(min = 64.dp),
                            )
                        },
                        menuContent = { menu ->
                            DropdownMenuItem(
                                text = { Text(action.label) },
                                onClick = {
                                    menu.dismiss()
                                    action.onClick()
                                },
                                leadingIcon = { Icon(painterResource(action.icon), null) },
                            )
                        },
                    )
                }
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                actions.forEach { action ->
                    HomeActionButton(action, Modifier.fillMaxWidth().heightIn(min = 64.dp))
                }
            }
        }
    }
}

@Composable
private fun HomeActionButton(
    action: HomeAction,
    modifier: Modifier = Modifier,
    interactionSource: MutableInteractionSource? = null,
) {
    Button(
        onClick = action.onClick,
        modifier = modifier,
        shapes =
            ButtonDefaults.shapes(
                shape = RoundedCornerShape(percent = 50),
                pressedShape = RoundedCornerShape(20.dp),
            ),
        colors =
            if (action.primary) ButtonDefaults.buttonColors()
            else ButtonDefaults.filledTonalButtonColors(),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 12.dp),
        interactionSource = interactionSource,
    ) {
        Icon(painterResource(action.icon), null, Modifier.size(24.dp))
        Spacer(Modifier.width(12.dp))
        Text(
            action.label,
            style = ButtonDefaults.textStyleFor(64.dp),
            fontWeight = if (action.primary) FontWeight.SemiBold else FontWeight.Medium,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun VoiceInputStatus(
    readiness: InputReadiness,
    onEnableKeyboard: () -> Unit,
    onChooseKeyboard: () -> Unit,
    onMicrophone: () -> Unit,
) {
    val ready = readiness.ready && !readiness.microphoneBlocked
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainer,
        shape = MaterialTheme.shapes.extraLarge,
    ) {
        Column(
            Modifier.fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = if (ready) 8.dp else 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (ready) {
                ReadinessAction(
                    R.drawable.ic_check_circle,
                    "Voice keyboard ready",
                    if (readiness.keyboardSelected) "Currently selected"
                    else "Switch to Transcribro",
                    onChooseKeyboard.takeUnless { readiness.keyboardSelected },
                )
            } else {
                Text(
                    "Set up voice input",
                    Modifier.padding(horizontal = 4.dp).semantics { heading() },
                    style = MaterialTheme.typography.titleMedium,
                )
                ReadinessAction(
                    R.drawable.ic_keyboard,
                    if (readiness.keyboardEnabled) "Voice keyboard enabled"
                    else "Enable the voice keyboard",
                    if (readiness.keyboardEnabled && readiness.keyboardSelected)
                        "Currently selected"
                    else if (readiness.keyboardEnabled) "Switch to Transcribro"
                    else "Add Transcribro in Android's keyboard settings",
                    if (readiness.keyboardEnabled) onChooseKeyboard else onEnableKeyboard,
                )
                ReadinessAction(
                    R.drawable.ic_mic,
                    when {
                        readiness.microphoneBlocked -> "Microphone is muted"
                        readiness.microphoneAllowed -> "Microphone ready"
                        else -> "Allow microphone access"
                    },
                    when {
                        readiness.microphoneBlocked ->
                            "Check Android's privacy settings and your device controls"
                        readiness.microphoneAllowed -> "Listening starts when you ask"
                        else -> "Needed when you dictate, not for audio files"
                    },
                    onMicrophone.takeIf {
                        !readiness.microphoneAllowed || readiness.microphoneBlocked
                    },
                )
            }
        }
    }
}

@Composable
private fun ReadinessAction(
    @DrawableRes icon: Int,
    title: String,
    detail: String,
    onClick: (() -> Unit)?,
) {
    val content: @Composable () -> Unit = {
        Row(
            Modifier.fillMaxWidth().padding(vertical = 10.dp, horizontal = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(painterResource(icon), null, Modifier.size(20.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(title, style = MaterialTheme.typography.labelLarge)
                Text(
                    detail,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (onClick != null)
                Icon(painterResource(R.drawable.ic_arrow_forward), null, Modifier.size(20.dp))
        }
    }
    if (onClick == null) Box(Modifier.semantics(mergeDescendants = true) {}) { content() }
    else
        Surface(
            onClick = onClick,
            color = MaterialTheme.colorScheme.surfaceContainer,
            shape = MaterialTheme.shapes.medium,
            content = content,
        )
}
