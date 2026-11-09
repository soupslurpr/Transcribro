package dev.soupslurpr.transcribro.engine

/** Audio-time bounds: ordinary pauses keep listening; a minute of inactivity does not. */
internal class ListeningLimits(private val realtime: Boolean) {
    private var samples = 0L
    private var lastSpeech = 0L
    private var heardSpeech = false

    fun accept(count: Int, speaking: Boolean): Int {
        samples += count
        if (speaking) {
            heardSpeech = true
            lastSpeech = samples
        }
        return when {
            samples >= 5 * 60 * 16_000 -> DURATION
            !realtime -> COMPLETE
            !heardSpeech && samples >= 12 * 16_000 -> NO_SPEECH
            heardSpeech && samples - lastSpeech >= 60 * 16_000 -> INACTIVITY
            else -> COMPLETE
        }
    }

    companion object {
        const val COMPLETE = 0
        const val INACTIVITY = 1
        const val DURATION = 2
        const val NO_SPEECH = 3
        const val OVERLOAD = 4
    }
}
