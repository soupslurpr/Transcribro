package dev.soupslurpr.transcribro.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContentTransitionScope.SlideDirection
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation3.runtime.NavEntry
import androidx.navigation3.runtime.NavEntryDecorator
import androidx.navigation3.runtime.rememberDecoratedNavEntries
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.scene.NavigationBackHandler
import androidx.navigation3.scene.SinglePaneSceneStrategy
import androidx.navigation3.scene.rememberNavigationEventState
import androidx.navigation3.scene.rememberSceneState
import androidx.navigation3.ui.NavDisplay
import androidx.navigationevent.NavigationEvent
import androidx.navigationevent.NavigationEventTransitionState.InProgress
import dev.soupslurpr.transcribro.MainViewModel
import dev.soupslurpr.transcribro.R
import dev.soupslurpr.transcribro.preferences.PauseLength
import dev.soupslurpr.transcribro.preferences.Preference
import dev.soupslurpr.transcribro.preferences.SendTiming

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScreen(
    model: MainViewModel,
    settingsRequest: Int,
    fileRequest: Int,
    onOpenAudio: () -> Unit,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onSave: () -> Unit,
    onEnableKeyboard: () -> Unit,
    onChooseKeyboard: () -> Unit,
    onMicrophone: () -> Unit,
    onVoiceSettings: () -> Unit,
    setPreference: (Preference, Boolean) -> Unit,
    setSendTiming: (SendTiming) -> Unit,
    setPauseLength: (PauseLength) -> Unit,
) {
    var backStack by rememberSaveable { mutableStateOf(listOf("home")) }
    val popDirections = remember { mutableStateMapOf<Any, SlideDirection>() }
    val page = backStack.last()
    var chooseKeyboardOnEntry by remember { mutableStateOf(false) }
    var discardAction by remember { mutableStateOf<String?>(null) }
    val context = LocalContext.current
    val focus = LocalFocusManager.current
    val keyboard = LocalSoftwareKeyboardController.current
    val snackbar = remember { SnackbarHostState() }
    val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0
    val preferences by model.preferences.collectAsStateWithLifecycle()
    val playback by model.playback.state.collectAsStateWithLifecycle()
    val file = model.file
    fun leavePage() {
        focus.clearFocus()
        keyboard?.hide()
        if (page == "test") model.clearTestInput()
        chooseKeyboardOnEntry = false
    }

    fun navigate(destination: String) {
        if (page == destination) return
        leavePage()
        backStack =
            when (destination) {
                "home" -> listOf("home")
                "about",
                "licenses" -> backStack + destination
                else -> listOf("home", destination)
            }
    }

    fun goBack() {
        if (backStack.size > 1) {
            // Clear practice only on committed navigation, never during a Back preview.
            popDirections[backStack.last()] = SlideDirection.End
            leavePage()
            backStack = backStack.dropLast(1)
        }
    }
    fun chooseFile() {
        if (file?.unpreserved == true || file?.active == true) discardAction = "replace"
        else onOpenAudio()
    }
    LaunchedEffect(fileRequest) {
        if (fileRequest > 0) navigate("file")
    }
    LaunchedEffect(settingsRequest) {
        if (settingsRequest > 0) navigate("settings")
    }
    LaunchedEffect(model.notice) {
        model.notice?.let {
            snackbar.showSnackbar(it)
            model.notice = null
        }
    }
    DisposableEffect(page, file?.id) {
        if (page == "file" && file != null) model.playback.load(file.uri)
        onDispose { model.playback.close() }
    }
    // Navigation entries outlive individual UI updates. Keep their rendered state and
    // callbacks current even when the back stack itself has not changed.
    val pageContent by
        rememberUpdatedState<@Composable (String) -> Unit> { page ->
            Scaffold(
                topBar = {
                    if (page != "home")
                        TopAppBar(
                            title = {
                                Text(
                                    when (page) {
                                        "test" -> "Try dictation"
                                        "file" -> "Audio transcription"
                                        "settings" -> "Settings"
                                        "about" -> "About Transcribro"
                                        "licenses" -> "Open-source licenses"
                                        else -> "Transcribro"
                                    },
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                            },
                            navigationIcon = {
                                IconButton(onClick = ::goBack) {
                                    Icon(painterResource(R.drawable.ic_arrow_back), "Back")
                                }
                            },
                            actions = {
                                if (page == "settings")
                                    IconButton(onClick = { navigate("about") }) {
                                        Icon(painterResource(R.drawable.ic_info), "About")
                                    }
                            },
                        )
                },
                snackbarHost = { if (page == backStack.last()) SnackbarHost(snackbar) },
            ) { insets ->
                Box(
                    Modifier.fillMaxSize().padding(insets).consumeWindowInsets(insets).imePadding(),
                    contentAlignment = Alignment.TopCenter,
                ) {
                    when (page) {
                        "home" ->
                            LauncherScreen(
                                model.readiness,
                                file,
                                { navigate("test") },
                                { navigate("file") },
                                { navigate("settings") },
                                { navigate("about") },
                                onEnableKeyboard,
                                {
                                    navigate("test")
                                    chooseKeyboardOnEntry = true
                                },
                                onMicrophone,
                            )
                        "test" -> {
                            // Keep the outgoing entry's last rendered words until its exit
                            // finishes. The model and undo history still clear immediately
                            // on leaving; this unsaved display state dies with the entry.
                            val displayedInput = remember { mutableStateOf(model.testInput) }
                            if (page == backStack.last()) displayedInput.value = model.testInput
                            TryDictationScreen(
                                displayedInput.value,
                                model.readiness,
                                imeVisible,
                                chooseKeyboardOnEntry,
                                preferences.autoStart,
                                model::updateTestInput,
                                model::undoTestInput,
                                model::redoTestInput,
                                model::clearTestInput,
                                onEnableKeyboard,
                                onChooseKeyboard,
                            )
                        }
                        "file" ->
                            FileTranscriptionScreen(
                                file,
                                playback,
                                ::chooseFile,
                                model::stopImport,
                                { discardAction = "discard" },
                                model.playback::toggle,
                                model.playback::seek,
                                onCopy,
                                onShare,
                                onSave,
                                saveFailed = model.saveFailed,
                                saving = model.saving,
                                onDismissSaveFailure = model::dismissSaveFailure,
                            )
                        "settings" ->
                            SettingsScreen(
                                preferences,
                                setPreference,
                                onEnableKeyboard,
                                onVoiceSettings,
                                setSendTiming,
                                setPauseLength,
                            )
                        "about" -> AboutScreen { navigate("licenses") }
                        "licenses" -> LicensesScreen()
                    }
                }
            }
        }
    val entries = rememberDecoratedNavEntries(
        backStack = backStack,
        entryDecorators = listOf(
            rememberSaveableStateHolderNavEntryDecorator(),
            remember {
                NavEntryDecorator<String>(
                    onPop = { key -> popDirections.remove(key) },
                    decorate = { entry -> entry.Content() },
                )
            },
        ),
    ) { destination ->
        NavEntry(destination, contentKey = destination) { pageContent(destination) }
    }
    val sceneState = rememberSceneState(
        entries = entries,
        sceneStrategies = remember { listOf(SinglePaneSceneStrategy<String>()) },
        onBack = ::goBack,
    )
    val navigationState = rememberNavigationEventState(sceneState)
    // Completion becomes Idle before invoking its callback. Capture the physical edge
    // while the preview is composed, and retain that direction for each outgoing entry.
    val gestureDirection = when (
        (navigationState.transitionState as? InProgress)?.latestEvent?.swipeEdge
    ) {
        NavigationEvent.EDGE_LEFT -> SlideDirection.Right
        NavigationEvent.EDGE_RIGHT -> SlideDirection.Left
        else -> SlideDirection.End
    }
    NavigationBackHandler(
        sceneState = sceneState,
        state = navigationState,
        onBackCompleted = {
            val outgoing = backStack.last()
            goBack()
            if (backStack.last() != outgoing) popDirections[outgoing] = gestureDirection
        },
    )
    val pageMovement = MaterialTheme.motionScheme.defaultSpatialSpec<IntOffset>()
    val pageShading = MaterialTheme.motionScheme.defaultEffectsSpec<Float>()
    // Placement and shading follow predictive gesture progress on the same linear timeline.
    val gestureMovement = tween<IntOffset>(300, easing = LinearEasing)
    val gestureShading = tween<Float>(300, easing = LinearEasing)
    NavDisplay(
        sceneState = sceneState,
        navigationEventState = navigationState,
        // The lower page moves one-third as far and darkens by up to 10% against black.
        modifier = Modifier.fillMaxSize().background(Color.Black),
        transitionSpec = {
            slideIntoContainer(SlideDirection.Start, pageMovement) togetherWith
                (slideOutOfContainer(SlideDirection.Start, pageMovement) { it / 3 } +
                    fadeOut(pageShading, targetAlpha = .9f))
        },
        popTransitionSpec = {
            val direction = popDirections[initialState.entries.last().contentKey]
                ?: SlideDirection.End
            (slideIntoContainer(direction, pageMovement) { it / 3 } +
                fadeIn(pageShading, initialAlpha = .9f)) togetherWith
                slideOutOfContainer(direction, pageMovement)
        },
        predictivePopTransitionSpec = { edge ->
            val direction = if (edge == NavigationEvent.EDGE_RIGHT) {
                SlideDirection.Left
            } else SlideDirection.Right
            (slideIntoContainer(direction, gestureMovement) { it / 3 } +
                fadeIn(gestureShading, initialAlpha = .9f)) togetherWith
                slideOutOfContainer(direction, gestureMovement)
        },
    )
    // Reserve Back for dismissing the IME or protecting a result at the root. All other
    // in-app Back gestures, including cancellation and saved page state, belong to NavDisplay.
    val protectResult = page == "home" && file?.unpreserved == true && !file.active
    BackHandler(enabled = imeVisible || protectResult) {
        if (imeVisible) {
            keyboard?.hide()
            focus.clearFocus()
        } else if (protectResult) discardAction = "leave"
        // The IME can finish hiding during a gesture that this handler already owns.
        // Consume that gesture without opening a discard prompt for a nonexistent result.
    }
    discardAction?.let { action ->
        AlertDialog(
            onDismissRequest = { discardAction = null },
            title = {
                Text(
                    if (action == "replace") "Replace this transcription?"
                    else "Discard this transcription?"
                )
            },
            text = {
                Text(
                    when {
                        action == "replace" ->
                            "Choosing another file will replace this result. Copy or save any completed words you want to keep."
                        file?.active == true ->
                            "Transcription will stop. Copy or save any completed words you want to keep before leaving."
                        else -> "This result is temporary. Copy or save any words you want to keep."
                    },
                    modifier = Modifier.verticalScroll(rememberScrollState()),
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        discardAction = null
                        if (action == "replace") onOpenAudio()
                        else {
                            model.clearFile()
                            if (action == "leave") (context as? android.app.Activity)?.finish()
                        }
                    }
                ) {
                    Text(if (action == "replace") "Choose file" else "Discard")
                }
            },
            dismissButton = {
                TextButton(onClick = { discardAction = null }) { Text("Keep transcription") }
            },
        )
    }
}
