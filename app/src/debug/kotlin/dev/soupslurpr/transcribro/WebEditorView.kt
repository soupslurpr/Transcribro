package dev.soupslurpr.transcribro

import android.annotation.SuppressLint
import android.app.Activity
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputConnectionWrapper
import android.view.inputmethod.InputMethodManager
import android.view.inputmethod.TextAttribute
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import org.json.JSONObject
import org.json.JSONTokener

/**
 * A local Chromium editor. Its DOM reports observations; Android still owns the input connection.
 */
// This debug fixture is constructed in code, never inflated from a layout.
@SuppressLint("SetJavaScriptEnabled", "ViewConstructor")
internal class WebEditorView(
    private val activity: Activity,
    kind: String,
    private val onInput: (InputConnection?) -> Unit,
    onField: (TextFieldValue) -> Unit,
    onSend: (String) -> Unit,
) : WebView(activity) {
    private val traceEnabled = activity.intent.getBooleanExtra("editorTrace", false)
    private val events = ArrayDeque<String>()

    private fun record(event: String) {
        if (!traceEnabled) return
        synchronized(events) {
            if (events.size == 200) events.removeFirst()
            events.addLast(event)
        }
    }

    fun trace(): String = synchronized(events) { events.joinToString("\n") }

    init {
        require(kind in setOf("input", "textarea", "rich"))
        settings.javaScriptEnabled = true
        settings.allowFileAccess = false
        settings.allowContentAccess = false
        addJavascriptInterface(
            object {
                @JavascriptInterface
                fun state(json: String) {
                    record("DOM $json")
                    val data = JSONObject(json)
                    val composing = data.getInt("composingStart")
                    val field =
                        TextFieldValue(
                            data.getString("text"),
                            TextRange(data.getInt("start"), data.getInt("end")),
                            if (composing >= 0) TextRange(composing, data.getInt("composingEnd"))
                            else null,
                        )
                    activity.runOnUiThread { onField(field) }
                }

                @JavascriptInterface
                fun send(text: String) = activity.runOnUiThread { onSend(text) }
            },
            "fixture",
        )
        webViewClient =
            object : WebViewClient() {
                override fun onPageFinished(view: WebView, url: String) {
                    evaluateJavascript("init(${JSONObject.quote(kind)})") {
                        requestFocus()
                        activity
                            .getSystemService(InputMethodManager::class.java)
                            .showSoftInput(this@WebEditorView, InputMethodManager.SHOW_IMPLICIT)
                    }
                }
            }
        loadDataWithBaseURL(
            "https://editor.transcribro.invalid/",
            activity.assets.open("editor-compatibility.html").bufferedReader().use {
                it.readText()
            },
            "text/html",
            "UTF-8",
            null,
        )
    }

    override fun onCreateInputConnection(outAttrs: EditorInfo): InputConnection? =
        super.onCreateInputConnection(outAttrs)
            ?.let { delegate ->
                if (!traceEnabled) return@let delegate
                record("connect ${outAttrs.initialSelStart}:${outAttrs.initialSelEnd}")
                object : InputConnectionWrapper(delegate, false) {
                    override fun setComposingText(
                        text: CharSequence?,
                        newCursorPosition: Int,
                    ): Boolean {
                        record("preview ${JSONObject.quote(text?.toString())}")
                        return super.setComposingText(text, newCursorPosition)
                    }

                    override fun commitText(text: CharSequence?, newCursorPosition: Int): Boolean {
                        record("commit ${JSONObject.quote(text?.toString())}")
                        return super.commitText(text, newCursorPosition)
                    }

                    override fun replaceText(
                        start: Int,
                        end: Int,
                        text: CharSequence,
                        newCursorPosition: Int,
                        textAttribute: TextAttribute?,
                    ): Boolean {
                        record("replace $start:$end ${JSONObject.quote(text.toString())}")
                        return super.replaceText(start, end, text, newCursorPosition, textAttribute)
                    }

                    override fun finishComposingText(): Boolean {
                        record("finish composition")
                        return super.finishComposingText()
                    }

                    override fun performEditorAction(editorAction: Int): Boolean {
                        record("action $editorAction")
                        return super.performEditorAction(editorAction)
                    }
                }
            }
            .also(onInput)

    fun setContent(html: String) {
        evaluateJavascript("setContent(${JSONObject.quote(html)})", null)
    }

    fun markup(callback: (String) -> Unit) {
        evaluateJavascript("field.innerHTML") { callback(JSONTokener(it).nextValue() as String) }
    }
}
