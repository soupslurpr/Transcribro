package dev.soupslurpr.transcribro.dictation

/**
 * Main-thread control for our own recognizer client. External recognition clients cannot join it.
 */
internal object PhraseCommands {
    private val sessions = mutableMapOf<String, (Long) -> Unit>()

    fun register(token: String, finish: (Long) -> Unit) {
        sessions[token] = finish
    }

    fun unregister(token: String) {
        sessions.remove(token)
    }

    fun finish(token: String, request: Long): Boolean {
        val session = sessions[token] ?: return false
        session(request)
        return true
    }
}
