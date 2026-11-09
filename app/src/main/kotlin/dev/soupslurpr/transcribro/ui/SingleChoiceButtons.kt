@file:OptIn(androidx.compose.material3.ExperimentalMaterial3ExpressiveApi::class)

package dev.soupslurpr.transcribro.ui

import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ButtonGroupDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.ToggleButton
import androidx.compose.material3.ToggleButtonDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.disabled
import androidx.compose.ui.semantics.focused
import androidx.compose.ui.semantics.onClick
import androidx.compose.ui.semantics.requestFocus
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.text
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import dev.soupslurpr.transcribro.R

/** Describes one option in a required, mutually exclusive choice. */
internal data class SingleChoiceOption(
    val label: String,
    val selected: Boolean,
    val onClick: () -> Unit,
    val enabled: Boolean = true,
)

/** Uses connected toggles when labels fit, otherwise full-width radio rows. */
@Composable
internal fun SingleChoiceButtons(
    choices: List<SingleChoiceOption>,
    modifier: Modifier = Modifier,
) {
    require(choices.isNotEmpty()) { "a choice group must not be empty" }
    require(choices.count { it.selected } == 1) { "exactly one option must be selected" }
    val density = LocalDensity.current
    val textMeasurer = rememberTextMeasurer()
    val buttonSize = ToggleButtonDefaults.size
    val labelStyle = ButtonDefaults.textStyleFor(buttonSize.height)
    val horizontalPadding = 12.dp
    val widestLabel = choices.maxOf { choice ->
        textMeasurer
            .measure(
                text = choice.label,
                style = labelStyle,
                maxLines = 1,
                softWrap = false,
            )
            .size
            .width
    }
    val spacing = ButtonGroupDefaults.ConnectedSpaceBetween
    BoxWithConstraints(modifier = modifier.fillMaxWidth()) {
        // Reserve padding and a selection icon even for currently unselected options,
        // so changing the selection cannot switch the layout or clip a label.
        // Round each part as the layout does, using the narrowest weighted button.
        val labelSpace =
            with(density) {
                val buttonWidth =
                    (constraints.maxWidth - spacing.roundToPx() * (choices.size - 1)) / choices.size
                buttonWidth -
                    horizontalPadding.roundToPx() * 2 -
                    ButtonDefaults.iconSizeFor(buttonSize.height).roundToPx() -
                    ButtonDefaults.iconSpacingFor(buttonSize.height).roundToPx()
            }
        if (widestLabel <= labelSpace) {
            Row(
                modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min).selectableGroup(),
                horizontalArrangement = Arrangement.spacedBy(spacing),
            ) {
                choices.forEachIndexed { index, choice ->
                    val interactionSource = remember { MutableInteractionSource() }
                    val isFocused by interactionSource.collectIsFocusedAsState()
                    val focusRequester = remember { FocusRequester() }
                    ToggleButton(
                        checked = choice.selected,
                        buttonSize = buttonSize,
                        onCheckedChange = { if (!choice.selected) choice.onClick() },
                        enabled = choice.enabled,
                        interactionSource = interactionSource,
                        shapes =
                            when {
                                choices.size == 1 -> ToggleButtonDefaults.shapesFor(buttonSize)
                                index == 0 -> ButtonGroupDefaults.connectedLeadingButtonShapes()
                                index == choices.lastIndex ->
                                    ButtonGroupDefaults.connectedTrailingButtonShapes()
                                else -> ButtonGroupDefaults.connectedMiddleButtonShapes()
                            },
                        modifier =
                            Modifier.weight(1f)
                                .fillMaxHeight()
                                .heightIn(min = 48.dp)
                                .focusRequester(focusRequester)
                                .clearAndSetSemantics {
                                    // Expose one radio option, rather than the toggle's checkbox
                                    // semantics and a second role node. Retain keyboard focus.
                                    role = Role.RadioButton
                                    selected = choice.selected
                                    text = AnnotatedString(choice.label)
                                    focused = isFocused
                                    if (!choice.enabled) disabled()
                                    requestFocus {
                                        if (choice.enabled) focusRequester.requestFocus() else false
                                    }
                                    onClick {
                                        if (choice.enabled && !choice.selected) choice.onClick()
                                        choice.enabled
                                    }
                                },
                        contentPadding =
                            PaddingValues(
                                horizontal = horizontalPadding,
                                vertical = 10.dp,
                            ),
                        icon =
                            if (choice.selected) {
                                {
                                    Icon(
                                        painterResource(R.drawable.ic_check),
                                        contentDescription = null,
                                    )
                                }
                            } else null,
                    ) {
                        Text(choice.label, maxLines = 1)
                    }
                }
            }
        } else {
            Column(
                modifier = Modifier.selectableGroup(),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                choices.forEach { choice ->
                    Surface(
                        shape = MaterialTheme.shapes.medium,
                        color =
                            if (choice.selected) {
                                MaterialTheme.colorScheme.secondaryContainer
                            } else Color.Transparent,
                    ) {
                        Row(
                            modifier =
                                Modifier.fillMaxWidth()
                                    .selectable(
                                        selected = choice.selected,
                                        enabled = choice.enabled,
                                        role = Role.RadioButton,
                                        onClick = choice.onClick,
                                    )
                                    .heightIn(min = 48.dp)
                                    .padding(horizontal = 12.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            RadioButton(
                                selected = choice.selected,
                                onClick = null,
                                enabled = choice.enabled,
                                modifier = Modifier.size(24.dp),
                            )
                            Text(
                                choice.label,
                                modifier = Modifier.weight(1f),
                                style = labelStyle,
                                color =
                                    if (choice.enabled) MaterialTheme.colorScheme.onSurface
                                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
                            )
                        }
                    }
                }
            }
        }
    }
}
