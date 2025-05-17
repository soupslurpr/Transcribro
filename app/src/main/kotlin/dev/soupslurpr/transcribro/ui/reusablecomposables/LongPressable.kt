package dev.soupslurpr.transcribro.ui.reusablecomposables

import android.view.ViewConfiguration.getKeyRepeatTimeout
import android.view.ViewConfiguration.getLongPressTimeout
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.PressInteraction
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/**
 * Configure component to receive long presses. Listens to provided [MutableInteractionSource]'s
 * [PressInteraction.Press] events to determine when a long press is being done.
 */
@Composable
fun Modifier.longPressable(
    interactionSource: MutableInteractionSource,
    onLongPress: suspend CoroutineScope.() -> Unit,
    longPressTimeout: Duration = getLongPressTimeout().milliseconds,
    hapticFeedbackEnabled: Boolean = true,
): Modifier {
    val hapticFeedback = LocalHapticFeedback.current
    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collectLatest { interaction ->
            when (interaction) {
                is PressInteraction.Press -> {
                    delay(longPressTimeout)
                    if (hapticFeedbackEnabled) {
                        hapticFeedback.performHapticFeedback(
                            HapticFeedbackType.LongPress
                        )
                    }
                    onLongPress()
                }
            }
        }
    }
    return this
}

/**
 * [longPressable] with defaults adjusted for keys.
 */
@Composable
fun Modifier.longPressableKey(
    interactionSource: MutableInteractionSource,
    onLongPress: suspend CoroutineScope.() -> Unit,
    longPressTimeout: Duration = getKeyRepeatTimeout().milliseconds,
    hapticFeedbackEnabled: Boolean = true,
): Modifier {
    return this.longPressable(
        interactionSource = interactionSource,
        onLongPress = onLongPress,
        longPressTimeout = longPressTimeout,
        hapticFeedbackEnabled = hapticFeedbackEnabled,
    )
}
