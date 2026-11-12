package dev.soupslurpr.transcribro.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class LicenseNotice(val file: String, val title: String, val chunks: List<String>)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun LicensesScreen() {
    val context = LocalContext.current
    var loadAttempt by remember { mutableIntStateOf(0) }
    val loaded by
        produceState<Result<List<LicenseNotice>>?>(null, loadAttempt) {
            value = null
            value =
                withContext(Dispatchers.IO) {
                    runCatching {
                        val files = context.assets.list("licenses").orEmpty()
                        check(files.isNotEmpty())
                        files
                            .map { file ->
                                LicenseNotice(
                                    file,
                                    licenseTitle(file),
                                    context.assets.open("licenses/$file").bufferedReader().use {
                                        reader ->
                                        // Bound text layouts at large font sizes without changing
                                        // notice text.
                                        reader.readLines().chunked(40).map { it.joinToString("\n") }
                                    },
                                )
                            }
                            .sortedWith(
                                compareBy<LicenseNotice> { it.file != "NOTICE.txt" }
                                    .thenBy(String.CASE_INSENSITIVE_ORDER) { it.title }
                            )
                    }
                }
        }
    val notices = loaded?.getOrNull()
    if (notices == null) {
        Column(
            Modifier.fillMaxSize().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(if (loaded == null) "Loading licenses…" else "Licenses could not be loaded.")
            if (loaded != null) TextButton(onClick = { loadAttempt++ }) { Text("Try again") }
        }
        return
    }

    val list = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var showComponents by rememberSaveable { mutableStateOf(false) }
    val componentPositions =
        remember(notices) {
            var next = 0
            notices.map { notice -> next.also { next += 1 + notice.chunks.size } }
        }
    Column(Modifier.widthIn(max = 800.dp).fillMaxSize()) {
        OutlinedButton(
            onClick = { showComponents = true },
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp),
        ) {
            Text("Jump to component")
        }
        HorizontalDivider()
        LazyColumn(
            Modifier.weight(1f).fillMaxWidth(),
            state = list,
            contentPadding = PaddingValues(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            notices.forEach { notice ->
                item(key = notice.file, contentType = "heading") {
                    Text(
                        notice.title,
                        Modifier.semantics { heading() },
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                itemsIndexed(
                    notice.chunks,
                    key = { index, _ -> "${notice.file}:$index" },
                    contentType = { _, _ -> "notice" },
                ) { _, chunk ->
                    SelectionContainer { Text(chunk, style = MaterialTheme.typography.bodySmall) }
                }
            }
        }
    }
    if (showComponents) {
        val sheet = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ModalBottomSheet(onDismissRequest = { showComponents = false }, sheetState = sheet) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 24.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Components",
                    Modifier.weight(1f).semantics { heading() },
                    style = MaterialTheme.typography.titleLarge,
                )
                TextButton(onClick = { showComponents = false }) { Text("Close") }
            }
            LazyColumn(Modifier.fillMaxWidth().weight(1f, fill = false)) {
                itemsIndexed(notices, key = { _, notice -> notice.file }) { index, notice ->
                    ListItem(
                        headlineContent = { Text(notice.title) },
                        modifier =
                            Modifier.clickable(role = Role.Button) {
                                scope.launch {
                                    list.scrollToItem(componentPositions[index])
                                    sheet.hide()
                                    showComponents = false
                                }
                            },
                    )
                }
            }
        }
    }
}

/** Presentation labels only. The packaged files and their license terms remain untouched. */
private fun licenseTitle(file: String): String =
    when (file) {
        "NOTICE.txt" -> "Transcribro and core libraries"
        "SOURCES.txt" -> "Recognition sources and attribution"
        "Material-Symbols-NOTICE.txt" -> "Material Symbols"
        "Protobuf-LICENSE.txt" -> "Protocol Buffers"
        "Rust-Dependency-NOTICES.txt" -> "Rust library dependencies"
        "Android-Dependency-NOTICES.txt" -> "Android library dependencies"
        "Moonshine-LICENSE.txt" -> "Moonshine"
        "Moonshine-MODIFICATIONS.txt" -> "Moonshine modifications"
        else -> file.substringBeforeLast('.').replace('-', ' ').replace('_', ' ')
    }
