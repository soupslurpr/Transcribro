package dev.soupslurpr.transcribro.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.input.key.*
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.InputReadiness

@Composable
internal fun TryDictationScreen(
    input: TextFieldValue,
    readiness: InputReadiness,
    imeVisible: Boolean,
    chooseKeyboardOnEntry: Boolean,
    autoStart: Boolean,
    onInput: (TextFieldValue) -> Unit,
    onUndo: () -> Unit,
    onRedo: () -> Unit,
    onClear: () -> Unit,
    onEnableKeyboard: () -> Unit,
    onChooseKeyboard: () -> Unit,
) {
    val fieldFocus = remember { FocusRequester() }
    val keyboard = LocalSoftwareKeyboardController.current
    var choosingKeyboard by remember { mutableStateOf(chooseKeyboardOnEntry) }
    LaunchedEffect(choosingKeyboard) {
        if (choosingKeyboard) {
            fieldFocus.requestFocus()
            keyboard?.show()
        }
    }
    LaunchedEffect(choosingKeyboard, imeVisible) {
        // Android omits auxiliary voice keyboards from its picker while the IME is hidden.
        // Use the real practice field and wait for visibility before opening the picker.
        if (choosingKeyboard && imeVisible) {
            choosingKeyboard = false
            onChooseKeyboard()
        }
    }
    BoxWithConstraints(Modifier.widthIn(max = 800.dp).fillMaxSize()) {
        val compact = imeVisible && maxHeight < 200.dp
        val scroll = rememberScrollState()
        LaunchedEffect(imeVisible, compact) { scroll.scrollTo(0) }
        Column(
            Modifier.fillMaxSize()
                .verticalScroll(scroll)
                .padding(horizontal = 20.dp, vertical = if (compact) 8.dp else 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            if (!imeVisible) {
                Text("A little room to try it.", style = MaterialTheme.typography.headlineSmall)
                Text(
                    if (autoStart)
                        "Tap the field and choose Transcribro. Speak when it says Listening, then try a correction or two."
                    else
                        "Tap the field, switch to Transcribro, and tap Start. Try a correction or two.",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            // A single composition location preserves the input connection during IME resizing.
            OutlinedTextField(
                value = input,
                onValueChange = onInput,
                modifier =
                    Modifier.fillMaxWidth()
                        .heightIn(min = if (compact) 64.dp else if (imeVisible) 128.dp else 200.dp)
                        .focusRequester(fieldFocus)
                        .onPreviewKeyEvent { event ->
                            if (
                                event.type == KeyEventType.KeyDown &&
                                    event.isCtrlPressed &&
                                    event.key == Key.Z
                            ) {
                                if (event.isShiftPressed) onRedo() else onUndo()
                                true
                            } else false
                        },
                label = { Text("Try your voice keyboard") },
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.None),
                maxLines = if (compact) 2 else 6,
                shape = MaterialTheme.shapes.large,
            )
            if (!compact) {
                Text(
                    "Practice text disappears when you leave this test.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (!readiness.keyboardEnabled) {
                        FilledTonalButton(onClick = onEnableKeyboard) { Text("Enable Transcribro") }
                    } else if (!readiness.keyboardSelected) {
                        FilledTonalButton(onClick = { choosingKeyboard = true }) {
                            Text("Choose Transcribro")
                        }
                    }
                    TextButton(onClick = onClear, enabled = input.text.isNotEmpty()) {
                        Text("Clear test")
                    }
                }
            }
        }
    }
}
