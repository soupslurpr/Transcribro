package dev.soupslurpr.transcribro

import android.os.Bundle
import android.text.InputType
import android.view.inputmethod.BaseInputConnection
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputConnectionWrapper
import android.widget.EditText
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.InterceptPlatformTextInput
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp

/** Debug-only, local editor fixtures. Send records text in RAM and contacts nobody. */
class EditorCompatibilityActivity : ComponentActivity() {
    var input: InputConnection? = null
    private var native: EditText? = null
    private var web: WebEditorView? = null
    private var value by mutableStateOf(TextFieldValue())
    val sent = mutableListOf<String>()
    val performedActions = mutableListOf<Int>()
    var clearOnSend = true
    var sendDelay = 0L

    fun field(): TextFieldValue =
        native?.let {
            TextFieldValue(
                it.text.toString(),
                TextRange(it.selectionStart, it.selectionEnd),
                BaseInputConnection.getComposingSpanStart(it.text)
                    .takeIf { start -> start >= 0 }
                    ?.let { start ->
                        TextRange(start, BaseInputConnection.getComposingSpanEnd(it.text))
                    },
            )
        } ?: value

    private fun send(text: String = field().text) {
        sent.add(text)
        if (clearOnSend)
            window.decorView.postDelayed(
                {
                    when {
                        web != null -> web!!.setContent("")
                        native != null -> native!!.setText("")
                        else -> value = TextFieldValue()
                    }
                },
                sendDelay,
            )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val host = intent.getStringExtra("host")
        if (host?.startsWith("web-") == true) {
            web =
                WebEditorView(
                    this,
                    host.removePrefix("web-"),
                    onInput = { input = it },
                    onField = { value = it },
                    onSend = ::send,
                )
            setContentView(web)
        } else if (host == "compose" || intent.getBooleanExtra("compose", false)) {
            setContent {
                val focus = remember { FocusRequester() }
                InterceptPlatformTextInput(
                    interceptor =
                        remember {
                            { request, handler ->
                                handler.startInputMethod { attributes ->
                                    request.createInputConnection(attributes).also { input = it }
                                }
                            }
                        }
                ) {
                    BasicTextField(
                        value = value,
                        onValueChange = { value = it },
                        modifier = Modifier.fillMaxSize().padding(32.dp).focusRequester(focus),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = { send() }),
                    )
                    LaunchedEffect(Unit) { focus.requestFocus() }
                }
            }
        } else {
            val field =
                object : EditText(this) {
                        override fun onCreateInputConnection(
                            outAttrs: EditorInfo
                        ): InputConnection? =
                            super.onCreateInputConnection(outAttrs)
                                ?.let { connection ->
                                    if (intent.getBooleanExtra("rejectComposing", false))
                                        object : InputConnectionWrapper(connection, false) {
                                            override fun setComposingText(
                                                text: CharSequence?,
                                                newCursorPosition: Int,
                                            ) = false
                                        }
                                    else connection
                                }
                                .also {
                                    // This local chat fixture deliberately advertises a separate
                                    // Send
                                    // action while retaining multiline Return behavior.
                                    outAttrs.imeOptions =
                                        outAttrs.imeOptions and
                                            EditorInfo.IME_FLAG_NO_ENTER_ACTION.inv()
                                    // TextView normally turns Unspecified into Done/Next.
                                    // Explicit test cases must advertise the requested action.
                                    if (intent.hasExtra("action"))
                                        outAttrs.imeOptions =
                                            (outAttrs.imeOptions and
                                                EditorInfo.IME_MASK_ACTION.inv()) or
                                                intent.getIntExtra(
                                                    "action",
                                                    EditorInfo.IME_ACTION_SEND,
                                                )
                                    input = it
                                }
                    }
                    .apply {
                        inputType =
                            if (intent.getBooleanExtra("password", false))
                                InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
                            else InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                        imeOptions = intent.getIntExtra("action", EditorInfo.IME_ACTION_SEND)
                        setOnEditorActionListener { _, action, _ ->
                            performedActions.add(action)
                            if (action == EditorInfo.IME_ACTION_SEND) {
                                send()
                            }
                            true
                        }
                    }
            native = field
            setContentView(field)
            field.requestFocus()
            window.setSoftInputMode(
                android.view.WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE
            )
        }
    }

    fun setWebContent(html: String) = web!!.setContent(html)

    fun webMarkup(callback: (String) -> Unit) = web!!.markup(callback)

    override fun onDestroy() {
        web?.destroy()
        super.onDestroy()
    }
}
