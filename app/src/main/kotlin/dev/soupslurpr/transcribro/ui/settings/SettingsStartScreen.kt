package dev.soupslurpr.transcribro.ui.settings

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material.icons.filled.Link
import androidx.compose.material3.HorizontalDivider
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperModelDownload
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.dataStore
import dev.soupslurpr.transcribro.preferences.PreferencesViewModel
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperLanguage
import dev.soupslurpr.transcribro.recognitionservice.whisper.WhisperModel
import dev.soupslurpr.transcribro.ui.reusablecomposables.ScreenLazyColumn

@Composable
fun SettingsStartScreen(
    onClickLicense: () -> Unit,
    onClickPrivacyPolicy: () -> Unit,
    onClickCredits: () -> Unit,
) {
    val preferencesViewModel: PreferencesViewModel = viewModel(
        factory = PreferencesViewModel.PreferencesViewModelFactory(LocalContext.current.dataStore)
    )

    val preferencesUiState by preferencesViewModel.uiState.collectAsState()

    val localUriHandler = LocalUriHandler.current

    ScreenLazyColumn(
        modifier = Modifier
            .fillMaxSize()
    ) {
        item {
            SettingsCategory(
                stringResource(R.string.theme)
            )
        }
        item {
            val preference = preferencesUiState.pitchBlackBackground
            SettingsSwitchItem(
                name = stringResource(id = R.string.pitch_black_background_setting_name),
                description = stringResource(id = R.string.pitch_black_background_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            SettingsCategory(
                stringResource(R.string.language_setting_category)
            )
        }
        item {
            val preference = preferencesUiState.languageOverride
            SettingsLanguageItem(
                value = preference.second.value,
                onValueChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            SettingsCategory(
                stringResource(R.string.model_setting_category)
            )
        }
        item {
            SettingsModelItem(
                selected = preferencesUiState.model.second.value,
                customPath = preferencesUiState.customModelPath.second.value,
                customName = preferencesUiState.customModelName.second.value,
                onSelectBundled = {
                    preferencesViewModel.setPreference(
                        preferencesUiState.model.first,
                        WhisperModel.DEFAULT.assetPath
                    )
                },
                onSelectCustom = {
                    val path = preferencesUiState.customModelPath.second.value
                    if (path.isNotBlank()) {
                        preferencesViewModel.setPreference(preferencesUiState.model.first, path)
                    }
                },
                onPickCustom = { path, name ->
                    preferencesViewModel.setPreference(preferencesUiState.customModelPath.first, path)
                    preferencesViewModel.setPreference(preferencesUiState.customModelName.first, name)
                    preferencesViewModel.setPreference(preferencesUiState.model.first, path)
                },
                onDeleteCustom = {
                    val path = preferencesUiState.customModelPath.second.value
                    if (path.isNotBlank()) {
                        runCatching { File(path).delete() }
                    }
                    // If the deleted model was selected, fall back to the bundled model.
                    if (preferencesUiState.model.second.value == path) {
                        preferencesViewModel.setPreference(
                            preferencesUiState.model.first,
                            WhisperModel.DEFAULT.assetPath
                        )
                    }
                    preferencesViewModel.setPreference(preferencesUiState.customModelPath.first, "")
                    preferencesViewModel.setPreference(preferencesUiState.customModelName.first, "")
                }
            )
        }
        item {
            SettingsModelLinkItem()
        }
        item {
            SettingsCategory(
                stringResource(R.string.voice_input_keyboard_setting_category)
            )
        }
        item {
            val preference = preferencesUiState.autoSwitchToPreviousInputMethod
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_switch_to_previous_input_method_setting_name),
                description = stringResource(id = R.string.auto_switch_to_previous_input_method_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            val preference = preferencesUiState.autoStopRecognition
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_stop_recognition_setting_name),
                description = stringResource(id = R.string.auto_stop_recognition_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            val preference = preferencesUiState.autoStartRecognition
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_start_recognition_setting_name),
                description = stringResource(id = R.string.auto_start_recognition_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            val preference = preferencesUiState.autoSendTranscription
            SettingsSwitchItem(
                name = stringResource(id = R.string.auto_send_transcription_setting_name),
                description = stringResource(id = R.string.auto_send_transcription_setting_description),
                checked = preference.second.value,
                onCheckedChange = {
                    preferencesViewModel.setPreference(
                        preference.first,
                        it
                    )
                }
            )
        }
        item {
            SettingsCategory(
                stringResource(R.string.about_setting_category)
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.view_source_code_setting_name),
                description = stringResource(id = R.string.view_source_code_setting_description),
                icon = Icons.AutoMirrored.Filled.ExitToApp,
                onClick = {
                    localUriHandler.openUri("https://github.com/soupslurpr/Transcribro")
                }
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.license_setting_name),
                description = stringResource(id = R.string.license_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickLicense
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.privacy_policy_setting_name),
                description = stringResource(id = R.string.privacy_policy_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickPrivacyPolicy
            )
        }
        item {
            SettingsIconItem(
                name = stringResource(id = R.string.credits_setting_name),
                description = stringResource(id = R.string.credits_setting_description),
                icon = Icons.Filled.Info,
                onClick = onClickCredits
            )
        }
    }
}

@Composable
fun SettingsCategory(category: String) {
    Text(
        text = category,
        modifier = Modifier.padding(top = 8.dp),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colorScheme.primary,
        style = MaterialTheme.typography.bodyMedium,
        fontWeight = FontWeight.Bold
    )
}

@Composable
fun SettingsSwitchItem(
    modifier: Modifier = Modifier,
    name: String,
    description: String? = null,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    ListItem(
        modifier = modifier
            .toggleable(
                value = checked,
                onValueChange = { onCheckedChange(it) }
            ),
        headlineContent = {
            Text(
                name,
                fontWeight = FontWeight.SemiBold
            )
        },
        supportingContent = run {
            if (description != null) {
                { Text(description) }
            } else {
                null
            }
        },
        trailingContent = {
            Switch(
                checked = checked,
                onCheckedChange = null
            )
        }
    )
}

@Composable
fun SettingsIconItem(
    modifier: Modifier = Modifier,
    name: String,
    description: String? = null,
    icon: ImageVector,
    onClick: () -> Unit
) {
    ListItem(
        modifier = modifier
            .clickable(
                onClick = onClick
            ),
        headlineContent = {
            Text(
                name,
                fontWeight = FontWeight.SemiBold
            )
        },
        supportingContent = run {
            if (description != null) {
                { Text(description) }
            } else {
                null
            }
        },
        trailingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null
            )
        }
    )
}
@Composable
fun SettingsLanguageItem(
    value: String,
    onValueChange: (String) -> Unit,
) {
    var showDialog by remember { mutableStateOf(false) }

    // Ordered options: auto-detect, then each forceable language.
    val autoDetectLabel = stringResource(R.string.language_auto_detect)
    val options = remember(autoDetectLabel) {
        buildList {
            add(WhisperLanguage.AUTO to autoDetectLabel)
            WhisperLanguage.SELECTABLE.forEach { code ->
                add(code to WhisperLanguage.displayName(code))
            }
        }
    }

    val currentLabel = options.firstOrNull { it.first == value }?.second
        ?: WhisperLanguage.displayName(value)

    SettingsIconItem(
        name = stringResource(R.string.language_setting_name),
        description = stringResource(R.string.language_setting_description, currentLabel),
        icon = Icons.Filled.Language,
        onClick = { showDialog = true }
    )

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(stringResource(R.string.language_setting_name)) },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 360.dp)
                        .verticalScroll(rememberScrollState())
                        .selectableGroup()
                ) {
                    options.forEach { (optionValue, optionLabel) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .selectable(
                                    selected = (optionValue == value),
                                    onClick = {
                                        onValueChange(optionValue)
                                        showDialog = false
                                    },
                                    role = Role.RadioButton
                                )
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = (optionValue == value),
                                onClick = null
                            )
                            Text(
                                text = optionLabel,
                                modifier = Modifier.padding(start = 12.dp)
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text(stringResource(R.string.close))
                }
            }
        )
    }
}

@Composable
fun SettingsModelItem(
    selected: String,
    customPath: String,
    customName: String,
    onSelectBundled: () -> Unit,
    onSelectCustom: () -> Unit,
    onPickCustom: (path: String, name: String) -> Unit,
    onDeleteCustom: () -> Unit,
) {
    var showDialog by remember { mutableStateOf(false) }
    var importing by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Storage Access Framework picker. This is the only point where file-system access is
    // requested, and only when the user actually imports a model. No storage permission is
    // needed; the picked file is copied into app-internal storage (off the main thread) and
    // afterwards referenced by its own file path.
    val pickModelLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            val name = queryDisplayName(context, uri) ?: uri.lastPathSegment ?: "model.bin"
            importing = true
            scope.launch {
                val path = withContext(Dispatchers.IO) {
                    copyModelToInternalStorage(context, uri)
                }
                importing = false
                if (path != null) {
                    onPickCustom(path, name)
                }
                showDialog = false
            }
        }
    }

    val isCustomSelected = WhisperModel.isCustom(selected)
    val description = if (isCustomSelected) {
        customName.ifBlank { stringResource(R.string.model_custom_unnamed) }
    } else {
        WhisperModel.DEFAULT.displayName
    }

    SettingsIconItem(
        name = stringResource(R.string.model_setting_name),
        description = stringResource(R.string.model_setting_description, description),
        icon = Icons.Filled.Memory,
        onClick = { showDialog = true }
    )

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { if (!importing) showDialog = false },
            title = { Text(stringResource(R.string.model_setting_name)) },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectableGroup()
                ) {
                    ModelRadioRow(
                        label = WhisperModel.DEFAULT.displayName,
                        selected = !isCustomSelected,
                        onClick = {
                            onSelectBundled()
                            showDialog = false
                        }
                    )
                    if (customPath.isNotBlank()) {
                        ModelRadioRow(
                            label = customName.ifBlank { stringResource(R.string.model_custom_unnamed) },
                            selected = isCustomSelected,
                            onClick = {
                                onSelectCustom()
                                showDialog = false
                            }
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    if (importing) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(modifier = Modifier.height(24.dp))
                            Text(
                                text = stringResource(R.string.model_importing),
                                modifier = Modifier.padding(start = 12.dp)
                            )
                        }
                    } else {
                        TextButton(onClick = { pickModelLauncher.launch(arrayOf("*/*")) }) {
                            Text(stringResource(R.string.model_choose_file))
                        }
                        if (customPath.isNotBlank()) {
                            TextButton(onClick = { onDeleteCustom() }) {
                                Text(stringResource(R.string.model_delete_imported))
                            }
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    enabled = !importing,
                    onClick = { showDialog = false }
                ) {
                    Text(stringResource(R.string.close))
                }
            }
        )
    }
}

@Composable
private fun ModelRadioRow(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .selectable(
                selected = selected,
                onClick = onClick,
                role = Role.RadioButton
            )
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = null)
        Text(text = label, modifier = Modifier.padding(start = 12.dp))
    }
}

@Composable
private fun ModelDownloadRow(
    model: String,
    size: WhisperModelDownload.ModelSize,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .selectable(
                selected = selected,
                onClick = onClick,
                role = Role.RadioButton
            )
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(selected = selected, onClick = null)
        Column(modifier = Modifier.padding(start = 12.dp)) {
            Text(text = model)
            Text(
                text = modelSizeText(size),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

/** Builds a "<disk> on disk · <mem> RAM" line, omitting whichever part is unknown. */
@Composable
private fun modelSizeText(size: WhisperModelDownload.ModelSize): String {
    val parts = listOfNotNull(
        size.disk?.let { stringResource(R.string.model_disk, it) },
        size.memory?.let { stringResource(R.string.model_mem, it) }
    )
    return if (parts.isEmpty()) stringResource(R.string.model_size_unknown) else parts.joinToString(" · ")
}

@Composable
fun SettingsModelLinkItem() {
    var showDialog by remember { mutableStateOf(false) }
    var selectedModel by remember { mutableStateOf(WhisperModelDownload.MODELS.first()) }
    val url = WhisperModelDownload.urlFor(selectedModel)
    val context = LocalContext.current

    SettingsIconItem(
        name = stringResource(R.string.model_link_setting_name),
        description = stringResource(R.string.model_link_setting_description),
        icon = Icons.Filled.Link,
        onClick = { showDialog = true }
    )

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(stringResource(R.string.model_link_setting_name)) },
            text = {
                Column(modifier = Modifier.fillMaxWidth()) {
                    val selectedSize = WhisperModelDownload.sizeFor(selectedModel)
                    Text(text = selectedModel, fontWeight = FontWeight.Bold)
                    Text(text = url, style = MaterialTheme.typography.bodySmall)
                    Text(
                        text = modelSizeText(selectedSize),
                        style = MaterialTheme.typography.bodySmall
                    )
                    Row {
                        TextButton(onClick = {
                            val clipboard =
                                context.getSystemService(ClipboardManager::class.java)
                            clipboard?.setPrimaryClip(
                                ClipData.newPlainText("Whisper model URL", url)
                            )
                        }) {
                            Text(stringResource(R.string.copy))
                        }
                        TextButton(onClick = {
                            runCatching {
                                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
                            }
                        }) {
                            Text(stringResource(R.string.open_in_browser))
                        }
                    }
                    HorizontalDivider()
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 280.dp)
                            .verticalScroll(rememberScrollState())
                            .selectableGroup()
                    ) {
                        WhisperModelDownload.MODELS.forEach { model ->
                            ModelDownloadRow(
                                model = model,
                                size = WhisperModelDownload.sizeFor(model),
                                selected = model == selectedModel,
                                onClick = { selectedModel = model }
                            )
                        }
                    }
                    Text(
                        text = stringResource(R.string.model_size_note),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text(stringResource(R.string.close))
                }
            }
        )
    }
}

private fun queryDisplayName(context: Context, uri: Uri): String? {
    return try {
        context.contentResolver
            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0) cursor.getString(index) else null
                } else {
                    null
                }
            }
    } catch (e: Exception) {
        null
    }
}

/**
 * Copies the model at [uri] into app-internal storage and returns its absolute path, or null
 * on failure. A single fixed destination is used, so importing a new model replaces the
 * previous one rather than accumulating files. Must be called off the main thread.
 */
private fun copyModelToInternalStorage(context: Context, uri: Uri): String? {
    val dir = File(context.filesDir, "models").apply { mkdirs() }
    val dest = File(dir, "custom-model.bin")
    val tmp = File(dir, "custom-model.bin.tmp")
    return try {
        val copied = context.contentResolver.openInputStream(uri)?.use { input ->
            tmp.outputStream().use { output ->
                input.copyTo(output)
            }
            true
        } ?: false
        // Rename only on success so an interrupted copy can't corrupt the model already in use.
        if (copied && tmp.renameTo(dest)) {
            dest.absolutePath
        } else {
            tmp.delete()
            null
        }
    } catch (e: Exception) {
        tmp.delete()
        null
    }
}
