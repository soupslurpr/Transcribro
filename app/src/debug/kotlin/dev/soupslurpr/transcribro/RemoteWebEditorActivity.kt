package dev.soupslurpr.transcribro

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.ResultReceiver
import android.view.ViewGroup
import android.view.inputmethod.InputConnection
import android.webkit.WebView
import androidx.compose.ui.text.input.TextFieldValue

/** Browser and IME must have independent UI threads, just as an actual browser app does. */
class RemoteWebEditorActivity : Activity() {
    private lateinit var web: WebEditorView
    private var input: InputConnection? = null
    private var field = TextFieldValue()
    private var loaded = false
    private val sent = arrayListOf<String>()
    private var sendDelay = 150L
    private var registered = false
    private val commands =
        object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val reply = intent.getParcelableExtra("reply", ResultReceiver::class.java) ?: return
                when (intent.getStringExtra("command")) {
                    "snapshot" ->
                        reply.send(
                            0,
                            Bundle().apply {
                                putBoolean("ready", loaded)
                                putString("text", field.text)
                                putStringArrayList("sent", ArrayList(sent))
                                putString("trace", web.trace())
                            },
                        )
                    "delay" -> {
                        sendDelay = intent.getLongExtra("delay", 150)
                        reply.send(0, Bundle())
                    }
                    "content" -> {
                        web.setContent(intent.getStringExtra("text").orEmpty())
                        reply.send(0, Bundle())
                    }
                    "close" -> {
                        unregisterReceiver(this)
                        registered = false
                        reply.send(0, Bundle())
                        finish()
                    }
                    "read",
                    "append" -> {
                        val connection = input ?: return reply.send(1, Bundle())
                        Handler(connection.handler?.looper ?: Looper.getMainLooper()).post {
                            val result = Bundle()
                            try {
                                if (intent.getStringExtra("command") == "read") {
                                    result.putStringArrayList(
                                        "parts",
                                        arrayListOf(
                                            connection
                                                .getTextBeforeCursor(1000, 0)
                                                ?.toString()
                                                .orEmpty(),
                                            connection.getSelectedText(0)?.toString().orEmpty(),
                                            connection
                                                .getTextAfterCursor(1000, 0)
                                                ?.toString()
                                                .orEmpty(),
                                        ),
                                    )
                                } else {
                                    result.putBoolean(
                                        "accepted",
                                        connection.commitText(intent.getStringExtra("text"), 1),
                                    )
                                }
                                reply.send(0, result)
                            } catch (error: Exception) {
                                result.putString("error", error.toString())
                                reply.send(1, result)
                            }
                        }
                    }
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!directorySet) {
            WebView.setDataDirectorySuffix("editor-host")
            directorySet = true
        }
        web =
            WebEditorView(
                this,
                intent.getStringExtra("host")!!.removePrefix("web-"),
                onInput = { input = it },
                onField = {
                    field = it
                    loaded = true
                },
                onSend = {
                    sent.add(it)
                    web.postDelayed({ if (!isFinishing) web.setContent("") }, sendDelay)
                },
            )
        setContentView(web)
        registerReceiver(commands, IntentFilter(ACTION), Context.RECEIVER_NOT_EXPORTED)
        registered = true
    }

    override fun onDestroy() {
        if (registered) unregisterReceiver(commands)
        (web.parent as? ViewGroup)?.removeView(web)
        web.destroy()
        super.onDestroy()
    }

    companion object {
        const val ACTION = "dev.soupslurpr.transcribro.TEST_EDITOR_COMMAND"
        private var directorySet = false
    }
}
