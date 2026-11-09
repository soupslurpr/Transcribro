package dev.soupslurpr.transcribro.dictation

import dev.soupslurpr.transcribro.preferences.Preferences
import dev.soupslurpr.transcribro.preferences.SendTiming

/** Tracks successfully inserted speech, never tentative results or editor contents alone. */
internal class AutomaticSending {
    private var pending = false

    fun committedSpeech() {
        pending = true
    }

    fun cancel() {
        pending = false
    }

    fun consume(preferences: Preferences, phraseEnd: Boolean, canSend: Boolean): Boolean {
        if (!pending || !preferences.autoSend || !canSend) return false
        if (phraseEnd && preferences.sendTiming != SendTiming.Phrase) return false
        pending = false
        return true
    }
}
