package dev.soupslurpr.transcribro.ui.voiceinput

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.icu.text.BreakIterator
import android.inputmethodservice.InputMethodService
import android.os.SystemClock
import android.text.InputType
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.CursorAnchorInfo
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.*
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.*
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import dev.soupslurpr.transcribro.MainActivity
import dev.soupslurpr.transcribro.MicrophonePermissionActivity
import dev.soupslurpr.transcribro.dictation.AutomaticSending
import dev.soupslurpr.transcribro.dictation.DictationController
import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.PreferencesRepository
import dev.soupslurpr.transcribro.ui.TranscribroTheme
import kotlinx.coroutines.*

class VoiceInput :
    InputMethodService(), LifecycleOwner, ViewModelStoreOwner, SavedStateRegistryOwner {
    private val registry = LifecycleRegistry(this)
    private val savedState = SavedStateRegistryController.create(this)
    override val lifecycle: Lifecycle
        get() = registry

    override val viewModelStore = ViewModelStore()
    override val savedStateRegistry: SavedStateRegistry
        get() = savedState.savedStateRegistry

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var controller: DictationController
    private var preferences by mutableStateOf(Preferences())
    private var editor by mutableStateOf<EditorInfo?>(null)
    private var shown = false
    private var editorGeneration = 0L
    private var expectedCursor = -1
    private var reportedSelection = -1 to -1
    private var hasSelection by mutableStateOf(false)
    private val expectedEdits = ArrayDeque<Pair<Int, Int>>()
    private var composition: EditorDictation? = null
    private var dictationTarget: InputConnection? = null
    private val automaticSending = AutomaticSending()
    private var awaitingSendReset = false
    private var allowSendRestart = false
    private val queuedEdits = ArrayDeque<() -> Unit>()
    private val pendingEditorEvents = ArrayDeque<() -> Unit>()
    private var awaitingEditorSelection = false
    private var keyEditSelection: Pair<Int, Int>? = null
    private var editorSelectionTimeout: Job? = null
    private var preferencesReady = false
    private var pendingAutoStart = false
    private var pendingManualStart = false
    private var composeView: ComposeView? = null

    override fun onCreate() {
        super.onCreate()
        savedState.performRestore(null)
        registry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
        controller = DictationController(this)
        scope.launch {
            PreferencesRepository(this@VoiceInput).values.collect {
                preferences = it
                preferencesReady = true
                if (pendingManualStart) startDictating() else maybeAutoStart()
            }
        }
    }

    override fun onCreateInputView(): View {
        composeView?.disposeComposition()
        window?.window?.decorView?.let {
            it.setViewTreeLifecycleOwner(this)
            it.setViewTreeViewModelStoreOwner(this)
            it.setViewTreeSavedStateRegistryOwner(this)
        }
        return ComposeView(this).apply {
            composeView = this
            setViewTreeLifecycleOwner(this@VoiceInput)
            setViewTreeViewModelStoreOwner(this@VoiceInput)
            setViewTreeSavedStateRegistryOwner(this@VoiceInput)
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed)
            setContent {
                val state by controller.state.collectAsState()
                val dark = isSystemInDarkTheme()
                SideEffect {
                    window?.window?.let {
                        WindowCompat.getInsetsController(
                                it,
                                it.decorView,
                            )
                            .isAppearanceLightNavigationBars = !dark
                    }
                }
                TranscribroTheme {
                    VoiceKeyboard(
                        state = state,
                        preferences = preferences,
                        passwordField = isPassword(editor?.inputType ?: 0),
                        hasSelection = hasSelection,
                        action = action(),
                        onStart = ::startDictating,
                        onStop = controller::stop,
                        onCancel = ::cancelDictation,
                        onUndo = { edit { sendKey(KeyEvent.KEYCODE_Z, KeyEvent.META_CTRL_ON) } },
                        onRedo = {
                            edit {
                                sendKey(
                                    KeyEvent.KEYCODE_Z,
                                    KeyEvent.META_CTRL_ON or KeyEvent.META_SHIFT_ON,
                                )
                            }
                        },
                        onClear = {
                            edit {
                                currentInputConnection?.deleteSurroundingText(
                                    Int.MAX_VALUE,
                                    Int.MAX_VALUE,
                                )
                            }
                        },
                        onLeft = { edit { moveCursor(forward = false) } },
                        onRight = { edit { moveCursor(forward = true) } },
                        onSwitch = {
                            cancelDictation()
                            if (!switchToPreviousInputMethod())
                                getSystemService(InputMethodManager::class.java)
                                    .showInputMethodPicker()
                        },
                        onSettings = {
                            cancelDictation()
                            startActivity(
                                Intent(this@VoiceInput, MainActivity::class.java)
                                    .setAction(Intent.ACTION_APPLICATION_PREFERENCES)
                                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            )
                        },
                        onSpace = { edit { currentInputConnection?.commitText(" ", 1) } },
                        onDelete = { edit { backspace() } },
                        onReturn = { edit { insertNewline() } },
                        onAction = { edit { performEditorAction() } },
                    )
                }
            }
        }
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {

        super.onStartInput(attribute, restarting)
        val previous = editor
        val replacement = currentInputConnection
        // Host undo/redo may recreate a Compose connection while applying our own edit.
        // Rebase only that acknowledged edit in the same field; new destinations still stop.
        if (
            restarting &&
                controller.state.value.active &&
                awaitingEditorSelection &&
                attribute != null &&
                previous != null &&
                replacement != null &&
                attribute.packageName == previous.packageName &&
                attribute.fieldId == previous.fieldId &&
                attribute.inputType == previous.inputType &&
                attribute.imeOptions == previous.imeOptions
        ) {
            editor = attribute
            dictationTarget = replacement
            composition = newComposition(replacement)
            completeEditorEdit(attribute.initialSelStart, attribute.initialSelEnd)
            return
        }
        // Compose may replace its connection when Send clears the same focused field.
        // Android marks that restart explicitly; a new field always cancels dictation.
        if (
            restarting &&
                controller.state.value.active &&
                allowSendRestart &&
                attribute != null &&
                previous != null &&
                attribute.packageName == previous.packageName &&
                attribute.fieldId == previous.fieldId &&
                attribute.inputType == previous.inputType &&
                attribute.imeOptions == previous.imeOptions &&
                replacement?.getTextBeforeCursor(1, 0)?.isEmpty() == true &&
                replacement.getTextAfterCursor(1, 0)?.isEmpty() == true &&
                replacement.getSelectedText(0).isNullOrEmpty()
        ) {
            editor = attribute
            reportedSelection = 0 to 0
            expectedCursor = 0
            expectedEdits.clear()
            awaitingSendReset = false
            dictationTarget = replacement
            composition = newComposition(replacement)
            scope.launch { applyQueuedEdits() }
            return
        }
        cancelDictation()
        // Recognition feedback belongs to the field where that attempt happened.
        // Retain it across a same-field restart, but not in a new destination.
        if (!restarting) controller.clearMessage()
        editorGeneration++
        editor = attribute
        expectedCursor = attribute?.let { minOf(it.initialSelStart, it.initialSelEnd) } ?: -1
        reportedSelection = (attribute?.initialSelStart ?: -1) to (attribute?.initialSelEnd ?: -1)
        hasSelection = attribute?.let { it.initialSelStart != it.initialSelEnd } ?: false
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        shown = true
        registry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
        pendingAutoStart = !restarting && !isPassword(info?.inputType ?: 0)
        maybeAutoStart()
    }

    private fun maybeAutoStart() {
        if (!shown || !preferencesReady || !pendingAutoStart) return
        pendingAutoStart = false
        if (preferences.autoStart && hasMicrophone()) startDictating()
    }

    private fun hasMicrophone() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun startDictating() {
        if (!shown) return
        pendingAutoStart = false
        if (!preferencesReady) {
            pendingManualStart = true
            return
        }
        pendingManualStart = false
        if (!hasMicrophone()) {
            startActivity(
                Intent(
                        this,
                        MicrophonePermissionActivity::class.java,
                    )
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
            return
        }
        if (controller.state.value.active) return
        queuedEdits.clear()
        val target = currentInputConnection ?: return
        dictationTarget = target
        composition = newComposition(target)
        val generation = editorGeneration
        automaticSending.cancel()
        awaitingSendReset = false
        allowSendRestart = false
        expectedEdits.clear()
        controller.onText = { text ->
            withEditorSelection {
                val target = dictationTarget
                if (target != null && shown && generation == editorGeneration) {
                    // Some editors report selection changes after delivering the next result.
                    if (
                        awaitingSendReset &&
                            target.getTextBeforeCursor(1, 0)?.isEmpty() == true &&
                            target.getTextAfterCursor(1, 0)?.isEmpty() == true &&
                            target.getSelectedText(0).isNullOrEmpty()
                    ) {
                        awaitingSendReset = false
                        expectedCursor = 0
                        expectedEdits.clear()
                        applyQueuedEdits()
                    }
                    if (composition?.commit(text) != true) {
                        controller.interrupt(
                            "The text field changed. Check your words before continuing."
                        )
                        automaticSending.cancel()
                    } else {
                        allowSendRestart = false
                        automaticSending.committedSpeech()
                        if (awaitingSendReset) {
                            // Preserve the new words, but stop sending if the host did not clear
                            // the previous message. An accepted action is not a delivery receipt.
                            controller.interrupt(
                                "This app didn't clear its message field after Send. Your words are kept; check the message before continuing."
                            )
                            automaticSending.cancel()
                            awaitingSendReset = false
                        }
                    }
                }
            }
        }
        controller.onHypothesis = { text ->
            withEditorSelection {
                if (!controller.state.value.active) queuedEdits.clear()
                val target = dictationTarget
                if (target != null && shown && generation == editorGeneration) {
                    // Wait for Send to clear its field before showing the next tentative phrase.
                    if (
                        awaitingSendReset &&
                            target.getTextBeforeCursor(1, 0)?.isEmpty() == true &&
                            target.getTextAfterCursor(1, 0)?.isEmpty() == true &&
                            target.getSelectedText(0).isNullOrEmpty()
                    ) {
                        awaitingSendReset = false
                        expectedCursor = 0
                        expectedEdits.clear()
                        applyQueuedEdits()
                    }
                    if (!awaitingSendReset && composition?.preview(text) == false) {
                        controller.interrupt(
                            "The text field changed. Check your words before continuing."
                        )
                        automaticSending.cancel()
                    }
                    if (!awaitingSendReset && text.isNotEmpty()) allowSendRestart = false
                }
            }
        }
        controller.onPhraseEnd = {
            withEditorSelection {
                if (shown && generation == editorGeneration) sendAutomatically(phraseEnd = true)
            }
        }
        controller.onFinished = {
            withEditorSelection {
                if (shown && generation == editorGeneration) {
                    sendAutomatically()
                    if (!controller.state.value.failure && preferences.switchBack)
                        switchToPreviousInputMethod()
                }
            }
        }
        controller.start(
            continuous = !preferences.stopAtPause,
            liveText = preferences.liveText && !isPassword(editor?.inputType ?: 0),
            pauseMillis = preferences.pauseLength.millis,
        )
    }

    override fun onUpdateSelection(
        oldSelStart: Int,
        oldSelEnd: Int,
        newSelStart: Int,
        newSelEnd: Int,
        candidatesStart: Int,
        candidatesEnd: Int,
    ) {
        super.onUpdateSelection(
            oldSelStart,
            oldSelEnd,
            newSelStart,
            newSelEnd,
            candidatesStart,
            candidatesEnd,
        )
        hasSelection = newSelStart != newSelEnd
        reportedSelection = newSelStart to newSelEnd
        if (awaitingEditorSelection) {
            // Key shortcuts travel through Android's input-event queue. An immediate cursor
            // request can precede them, so only their changed selection acknowledges the key.
            if (keyEditSelection != null && keyEditSelection != reportedSelection)
                completeEditorEdit(newSelStart, newSelEnd)
            return
        }
        val expected = expectedEdits.indexOf(newSelStart to newSelEnd)
        if (expected >= 0) repeat(expected + 1) { expectedEdits.removeFirst() }
        val sendReset = awaitingSendReset && newSelStart == 0 && newSelEnd == 0
        if (sendReset) {
            awaitingSendReset = false
            scope.launch { applyQueuedEdits() }
        }
        if (
            controller.state.value.active &&
                (oldSelStart != newSelStart || oldSelEnd != newSelEnd) &&
                expected < 0 &&
                !sendReset &&
                (newSelStart != expectedCursor || newSelEnd != expectedCursor)
        ) {

            composition?.abandon()
            cancelDictation()
        } else {
            if (
                composition?.selectionChanged(
                    newSelStart,
                    newSelEnd,
                    candidatesStart,
                    candidatesEnd,
                ) == false
            ) {
                controller.interrupt("The text field changed. Check your words before continuing.")
                automaticSending.cancel()
            }
        }
        expectedCursor = expectedEdits.lastOrNull()?.first ?: minOf(newSelStart, newSelEnd)
    }

    private fun insertNewline() {
        val input = currentInputConnection ?: return
        val surrounding = input.getSurroundingText(0, 0, 0)
        val selection =
            if (surrounding != null && surrounding.offset >= 0)
                (surrounding.offset + surrounding.selectionStart) to
                    (surrounding.offset + surrounding.selectionEnd)
            else reportedSelection
        // Chromium turns commitText with a lone newline into Enter, which a page may use
        // for Send. Replace the selection as text without requiring composition support.
        if (selection.first >= 0 && selection.second >= 0)
            input.replaceText(
                minOf(selection.first, selection.second),
                maxOf(selection.first, selection.second),
                "\n",
                1,
                null,
            )
        else input.commitText("\n", 1)
    }

    private fun moveCursor(forward: Boolean) {
        val input = currentInputConnection ?: return
        val surrounding = input.getSurroundingText(1024, 1024, 0)
        if (surrounding == null) {
            sendKey(if (forward) KeyEvent.KEYCODE_DPAD_RIGHT else KeyEvent.KEYCODE_DPAD_LEFT)
            return
        }
        val start = minOf(surrounding.selectionStart, surrounding.selectionEnd)
        val end = maxOf(surrounding.selectionStart, surrounding.selectionEnd)
        val boundary =
            BreakIterator.getCharacterInstance().apply {
                setText(surrounding.text.toString())
            }
        val next =
            if (start != end) {
                if (forward) end else start
            } else if (forward) boundary.following(end) else boundary.preceding(start)
        if (next != BreakIterator.DONE) {
            val reported = expectedEdits.lastOrNull() ?: reportedSelection
            val offset =
                if (surrounding.offset >= 0) surrounding.offset
                else minOf(reported.first, reported.second) - start
            if (offset >= 0) input.setSelection(offset + next, offset + next)
            else sendKey(if (forward) KeyEvent.KEYCODE_DPAD_RIGHT else KeyEvent.KEYCODE_DPAD_LEFT)
        }
    }

    private fun backspace() {
        val input = currentInputConnection ?: return
        if (!input.getSelectedText(0).isNullOrEmpty()) {
            input.commitText("", 1)
        } else {
            val before = input.getTextBeforeCursor(1024, 0)?.toString().orEmpty()
            if (before.isEmpty()) return
            val boundary = BreakIterator.getCharacterInstance().apply { setText(before) }
            val end = boundary.last()
            val start = boundary.previous()
            if (start == 0 && before.length == 1024) {
                sendKey(KeyEvent.KEYCODE_DEL)
            } else if (start != BreakIterator.DONE) {
                input.deleteSurroundingText(end - start, 0)
            }
        }
    }

    private fun expectSelection(start: Int, end: Int) {
        expectedCursor = minOf(start, end)
        if (expectedEdits.size == 64) expectedEdits.removeFirst()
        expectedEdits.addLast(start to end)
    }

    private fun newComposition(target: InputConnection) =
        EditorDictation(
            target,
            preferences.liveText && !isPassword(editor?.inputType ?: 0),
            selection = { expectedEdits.lastOrNull() ?: reportedSelection },
            expectSelection = ::expectSelection,
        )

    private fun cancelDictation() {
        pendingAutoStart = false
        pendingManualStart = false
        queuedEdits.clear()
        pendingEditorEvents.clear()
        awaitingEditorSelection = false
        keyEditSelection = null
        editorSelectionTimeout?.cancel()
        editorSelectionTimeout = null
        composition?.cancel()
        composition = null
        dictationTarget = null
        controller.cancel()
        automaticSending.cancel()
        awaitingSendReset = false
        allowSendRestart = false
        expectedEdits.clear()
    }

    private fun edit(action: () -> Unit) {
        if (!controller.state.value.active) {
            cancelDictation()
            action()
            return
        }
        if (queuedEdits.size >= 64) return
        queuedEdits.addLast(action)
        if (queuedEdits.size == 1) {
            automaticSending.cancel()
            controller.finishPhrase { applyQueuedEdits(allowInactive = true) }
        }
    }

    private fun applyQueuedEdits(allowInactive: Boolean = false) {
        if (!allowInactive && !controller.state.value.active) return
        if (awaitingEditorSelection || awaitingSendReset || queuedEdits.isEmpty()) return
        val input = dictationTarget ?: return
        if (!shown || input !== currentInputConnection) return
        val action = queuedEdits.removeFirst()
        automaticSending.cancel()
        awaitingEditorSelection = true
        input.beginBatchEdit()
        try {
            action()
        } finally {
            input.endBatchEdit()
        }
        composition = newComposition(input)
        if (awaitingSendReset) {
            awaitingEditorSelection = false
            return
        }
        // The request follows the edit on the same connection. Some Compose editors return
        // surrounding text with offset -1; its relative selection cannot locate our next phrase.
        val surrounding = input.getSurroundingText(0, 0, 0)
        if (keyEditSelection == null && surrounding != null && surrounding.offset >= 0) {
            completeEditorEdit(
                surrounding.offset + surrounding.selectionStart,
                surrounding.offset + surrounding.selectionEnd,
            )
        } else if (input.requestCursorUpdates(InputConnection.CURSOR_UPDATE_IMMEDIATE)) {
            editorSelectionTimeout = scope.launch {
                delay(1000)
                if (awaitingEditorSelection)
                    completeEditorEdit(reportedSelection.first, reportedSelection.second)
            }
        } else {
            // The synchronous read above orders the edit and its selection notifications.
            // Give those already queued main-thread callbacks a turn before using their result.
            scope.launch {
                yield()
                if (awaitingEditorSelection)
                    completeEditorEdit(reportedSelection.first, reportedSelection.second)
            }
        }
    }

    override fun onUpdateCursorAnchorInfo(info: CursorAnchorInfo?) {

        super.onUpdateCursorAnchorInfo(info)
        if (
            awaitingEditorSelection &&
                info != null &&
                (keyEditSelection == null ||
                    keyEditSelection != (info.selectionStart to info.selectionEnd))
        ) {
            completeEditorEdit(info.selectionStart, info.selectionEnd)
        }
    }

    private fun completeEditorEdit(start: Int, end: Int) {
        editorSelectionTimeout?.cancel()
        editorSelectionTimeout = null
        awaitingEditorSelection = false
        keyEditSelection = null
        expectedEdits.clear()
        if (start >= 0 && end >= 0) {
            reportedSelection = start to end
            hasSelection = start != end
            expectSelection(start, end)
        }
        while (pendingEditorEvents.isNotEmpty() && !awaitingEditorSelection) {
            pendingEditorEvents.removeFirst().invoke()
        }
        applyQueuedEdits(allowInactive = true)
    }

    private fun withEditorSelection(event: () -> Unit) {
        if (!awaitingEditorSelection) {
            event()
        } else if (pendingEditorEvents.size < 64) {
            pendingEditorEvents.addLast(event)
        } else {
            cancelDictation()
            controller.interrupt("The text field isn't responding. Your completed words are kept.")
        }
    }

    private fun sendAutomatically(phraseEnd: Boolean = false) {
        if (
            awaitingSendReset ||
                awaitingEditorSelection ||
                queuedEdits.isNotEmpty() ||
                controller.state.value.finishingPhrase
        )
            return
        val input = currentInputConnection ?: return
        if (
            !automaticSending.consume(
                preferences,
                phraseEnd,
                action() == EditorInfo.IME_ACTION_SEND,
            )
        )
            return
        awaitingSendReset = true
        allowSendRestart = true
        expectSelection(0, 0)
        if (!input.performEditorAction(EditorInfo.IME_ACTION_SEND)) {
            awaitingSendReset = false
            allowSendRestart = false
            expectedEdits.clear()
            controller.interrupt(
                "This field doesn't support automatic sending. Your words are kept."
            )
        }
    }

    private fun sendKey(code: Int, meta: Int = 0) {
        val input = currentInputConnection ?: return
        if (awaitingEditorSelection)
            keyEditSelection = expectedEdits.lastOrNull() ?: reportedSelection
        val time = SystemClock.uptimeMillis()
        input.sendKeyEvent(KeyEvent(time, time, KeyEvent.ACTION_DOWN, code, 0, meta))
        input.sendKeyEvent(KeyEvent(time, time, KeyEvent.ACTION_UP, code, 0, meta))
    }

    private fun action(): Int {
        val options = editor?.imeOptions ?: 0
        return if (options and EditorInfo.IME_FLAG_NO_ENTER_ACTION != 0) {
            EditorInfo.IME_ACTION_NONE
        } else {
            options and EditorInfo.IME_MASK_ACTION
        }
    }

    private fun performEditorAction() {
        val action = action()
        if (action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED) {
            if (action == EditorInfo.IME_ACTION_SEND && controller.state.value.active) {
                if (awaitingSendReset) return
                awaitingSendReset = true
                allowSendRestart = true
                expectSelection(0, 0)
                if (currentInputConnection?.performEditorAction(action) != true) {
                    awaitingSendReset = false
                    allowSendRestart = false
                    controller.interrupt("This field doesn't support Send. Your words are kept.")
                }
            } else currentInputConnection?.performEditorAction(action)
        }
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        shown = false
        cancelDictation()
        registry.handleLifecycleEvent(Lifecycle.Event.ON_STOP)
        super.onFinishInputView(finishingInput)
    }

    override fun onFinishInput() {
        shown = false
        editorGeneration++
        cancelDictation()
        super.onFinishInput()
    }

    override fun onEvaluateFullscreenMode() = false

    override fun onDestroy() {
        // InputMethodService finishes the active input view during its teardown.
        // Keep our lifecycle alive until those final callbacks have completed.
        super.onDestroy()
        controller.close()
        scope.cancel()
        composeView?.disposeComposition()
        registry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        viewModelStore.clear()
    }

    companion object {
        fun isPassword(type: Int): Boolean {
            val variation = type and InputType.TYPE_MASK_VARIATION
            return when (type and InputType.TYPE_MASK_CLASS) {
                InputType.TYPE_CLASS_TEXT ->
                    variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                        variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD ||
                        variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
                InputType.TYPE_CLASS_NUMBER -> variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD
                else -> false
            }
        }

        fun actionLabel(action: Int) =
            when (action) {
                EditorInfo.IME_ACTION_SEND -> "Send"
                EditorInfo.IME_ACTION_SEARCH -> "Search"
                EditorInfo.IME_ACTION_GO -> "Go"
                EditorInfo.IME_ACTION_NEXT -> "Next"
                EditorInfo.IME_ACTION_DONE -> "Done"
                EditorInfo.IME_ACTION_PREVIOUS -> "Previous"
                else -> "Return"
            }
    }
}
