package dev.soupslurpr.transcribro.recognitionservice.whisper

/**
 * Builds Hugging Face download URLs for the pre-converted ggml Whisper models, mirroring the
 * logic of whisper.cpp's `download-ggml-model.sh`.
 *
 * The app never downloads anything itself (no network permission); it only produces the URL so
 * the user can fetch a model with a browser and then import it from the file system.
 */
object WhisperModelDownload {
    private const val DEFAULT_SRC = "https://huggingface.co/ggerganov/whisper.cpp"
    private const val TDRZ_SRC = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
    private const val PFX = "resolve/main/ggml"

    /** Model identifiers, in the same order/grouping as download-ggml-model.sh. */
    val MODELS: List<String> = listOf(
        "tiny", "tiny.en", "tiny-q5_1", "tiny.en-q5_1", "tiny-q8_0",
        "base", "base.en", "base-q5_1", "base.en-q5_1", "base-q8_0",
        "small", "small.en", "small.en-tdrz", "small-q5_1", "small.en-q5_1", "small-q8_0",
        "medium", "medium.en", "medium-q5_0", "medium.en-q5_0", "medium-q8_0",
        "large-v1", "large-v2", "large-v2-q5_0", "large-v2-q8_0",
        "large-v3", "large-v3-q5_0", "large-v3-turbo", "large-v3-turbo-q5_0",
        "large-v3-turbo-q8_0"
    )

    /**
     * Returns the ggml download URL for [model], e.g.
     * "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin".
     * tinydiarize ("-tdrz") models come from a different repository.
     */
    fun urlFor(model: String): String {
        val src = if (model.contains("tdrz")) TDRZ_SRC else DEFAULT_SRC
        return "$src/$PFX-$model.bin"
    }

    /** Approximate on-disk download size and required memory for a model (either may be null). */
    data class ModelSize(val disk: String?, val memory: String?)

    /**
     * Actual download size of each model's `.bin`, as reported by the Hugging Face repository
     * file listing (https://huggingface.co/ggerganov/whisper.cpp and the tinydiarize repo for
     * `-tdrz`). These were probed once and bundled with the app; no network access is performed
     * at runtime.
     */
    private val DISK_BY_MODEL: Map<String, String> = mapOf(
        "tiny" to "77.7 MB",
        "tiny.en" to "77.7 MB",
        "tiny-q5_1" to "32.2 MB",
        "tiny.en-q5_1" to "32.2 MB",
        "tiny-q8_0" to "43.5 MB",
        "base" to "148 MB",
        "base.en" to "148 MB",
        "base-q5_1" to "59.7 MB",
        "base.en-q5_1" to "59.7 MB",
        "base-q8_0" to "81.8 MB",
        "small" to "488 MB",
        "small.en" to "488 MB",
        "small.en-tdrz" to "488 MB",
        "small-q5_1" to "190 MB",
        "small.en-q5_1" to "190 MB",
        "small-q8_0" to "264 MB",
        "medium" to "1.53 GB",
        "medium.en" to "1.53 GB",
        "medium-q5_0" to "539 MB",
        "medium.en-q5_0" to "539 MB",
        "medium-q8_0" to "823 MB",
        "large-v1" to "3.09 GB",
        "large-v2" to "3.09 GB",
        "large-v2-q5_0" to "1.08 GB",
        "large-v2-q8_0" to "1.66 GB",
        "large-v3" to "3.1 GB",
        "large-v3-q5_0" to "1.08 GB",
        "large-v3-turbo" to "1.62 GB",
        "large-v3-turbo-q5_0" to "574 MB",
        "large-v3-turbo-q8_0" to "874 MB",
    )

    /**
     * Required memory per size class, from the whisper.cpp "Memory usage" table. Bundled with
     * the app; no network access.
     */
    private val MEM_BY_CLASS: Map<String, String> = mapOf(
        "tiny" to "~273 MB",
        "base" to "~388 MB",
        "small" to "~852 MB",
        "medium" to "~2.1 GB",
        "large" to "~3.9 GB",
    )

    /**
     * Size info for [model]: exact disk size if published, plus the required memory for its
     * size class (e.g. "base.en-q5_1" -> "base").
     */
    fun sizeFor(model: String): ModelSize {
        val sizeClass = model.substringBefore('.').substringBefore('-')
        return ModelSize(disk = DISK_BY_MODEL[model], memory = MEM_BY_CLASS[sizeClass])
    }
}
