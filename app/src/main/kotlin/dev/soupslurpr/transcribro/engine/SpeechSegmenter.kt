package dev.soupslurpr.transcribro.engine

/** Speech-pause detection independent of the recognizer's internal line boundaries. */
class SpeechSegmenter(private val silenceFrames: Int = 30) {
    private var quietFrames = 0
    private var voicedFrames = 0
    var speaking = false
        private set

    var completedPhrase = false
        private set

    init {
        require(silenceFrames > 0)
    }

    /** One probability per 512-sample / 32 ms Silero frame. No audio is retained here. */
    fun accept(probability: Float) {
        require(probability.isFinite() && probability in 0f..1f)
        completedPhrase = false
        val voiced = probability >= if (speaking) 0.35f else 0.6f
        if (!speaking && !voiced) return
        speaking = true
        if (voiced) {
            voicedFrames = (voicedFrames + 1).coerceAtMost(4)
            quietFrames = 0
        } else quietFrames++
        if (quietFrames >= silenceFrames) {
            completedPhrase = voicedFrames >= 4
            speaking = false
            voicedFrames = 0
            quietFrames = 0
        }
    }
}
