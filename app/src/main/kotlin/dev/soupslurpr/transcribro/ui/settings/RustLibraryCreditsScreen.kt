package dev.soupslurpr.transcribro.ui.settings

import android.webkit.WebView
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun RustLibraryCreditsScreen() {
    val colorScheme = MaterialTheme.colorScheme

    AndroidView(
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = false // disable JavaScript for security.
                settings.setSupportZoom(false)
                settings.builtInZoomControls = false
                settings.displayZoomControls = false
                setBackgroundColor(colorScheme.background.toArgb()) // set WebView background color to current colorScheme's background color.
            }
        },
        update = { view ->
            view.loadUrl("file:///android_asset/transcribro_rs-third-party-licenses.html")
        }
    )
}