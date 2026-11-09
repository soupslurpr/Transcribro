package dev.soupslurpr.transcribro.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import dev.soupslurpr.transcribro.R

/** The same app identity on Home and About; listening has its own live indicator. */
@Composable
internal fun TranscribroAppIcon(size: Dp, modifier: Modifier = Modifier) {
    Box(
        modifier
            .size(size)
            .clip(RoundedCornerShape(percent = 30))
            .background(colorResource(R.color.ic_launcher_background))
    ) {
        Image(
            painter = painterResource(R.drawable.ic_launcher_foreground),
            contentDescription = null,
            // Keep the adaptive foreground's 108-to-72 viewport scale.
            modifier = Modifier.fillMaxSize().scale(1.5f),
        )
    }
}
