package dev.soupslurpr.transcribro.ui

import android.content.Intent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import dev.soupslurpr.transcribro.BuildConfig
import dev.soupslurpr.transcribro.R

private const val SOURCE_CODE_URL = "https://github.com/soupslurpr/Transcribro"

@Composable
internal fun AboutScreen(onLicenses: () -> Unit) {
    val context = LocalContext.current
    var linkUnavailable by rememberSaveable { mutableStateOf(false) }
    BoxWithConstraints(Modifier.widthIn(max = 680.dp).fillMaxSize()) {
        val compact = maxHeight < 480.dp
        Column(
            Modifier.fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp)
                .padding(top = if (compact) 12.dp else 24.dp, bottom = 32.dp),
            verticalArrangement = Arrangement.spacedBy(if (compact) 12.dp else 24.dp),
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(if (compact) 12.dp else 16.dp)) {
                AboutIdentity()
                Text(
                    "Use your voice in the apps you already use, or turn an audio file into text. Recognition runs on your device, in English."
                )
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FilledTonalButton(
                    shapes = ButtonDefaults.shapes(),
                    contentPadding = ButtonDefaults.ContentPadding,
                    onClick = {
                        linkUnavailable =
                            runCatching {
                                context.startActivity(
                                    Intent(Intent.ACTION_VIEW, SOURCE_CODE_URL.toUri())
                                )
                            }
                                .isFailure
                    },
                ) {
                    Text("Source code")
                    Spacer(Modifier.width(8.dp))
                    Icon(
                        painter = painterResource(R.drawable.ic_open_in_new),
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                }
                TextButton(onClick = onLicenses) { Text("Open-source licenses") }
            }
            if (linkUnavailable) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "No app is available to open the source code link. You can copy it below.",
                        Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                        color = MaterialTheme.colorScheme.error,
                    )
                    SelectionContainer { Text(SOURCE_CODE_URL) }
                }
            }
            AboutSection(
                "Where dictated text goes",
                "The app you're dictating into receives your words, including unfinished words when Live text is enabled. That app controls how it stores or shares them. Copying, sharing, or saving a file transcription sends text to the destination you choose.",
            )
            AboutSection(
                "Keeping your words",
                "Practice text and file transcriptions stay in memory. Copy, share, or save a transcription to keep it. Unsaved text is lost if Android ends the app. Transcribro keeps no recordings or transcript library, and requests no internet permission.",
            )
            AboutSection(
                "Recognition",
                "Review recognized words before sending. Listening stops after a minute without speech or five minutes total, keeping completed words. Selected audio files can be up to 30 minutes long. Supported file formats depend on Android's decoders.",
            )
            Text(
                "Transcribro ${BuildConfig.VERSION_NAME}\n© 2024–2026 soupslurpr",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun AboutIdentity() {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        TranscribroAppIcon(if (LocalDensity.current.fontScale >= 2f) 44.dp else 56.dp)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(
                "Transcribro",
                Modifier.semantics { heading() },
                style = MaterialTheme.typography.headlineMedium,
            )
            Text(
                "Your voice. Your words.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun AboutSection(title: String, body: String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            title,
            Modifier.semantics { heading() },
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        Text(body)
    }
}
