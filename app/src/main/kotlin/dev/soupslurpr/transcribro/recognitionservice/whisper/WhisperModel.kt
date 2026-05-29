package dev.soupslurpr.transcribro.recognitionservice.whisper

/**
 * Whisper models bundled as app assets.
 *
 * Only the base model is bundled. Other models can be used by downloading them (see
 * [WhisperModelDownload]) and importing them from the local file system; in that case the
 * model is referenced by a content URI rather than a [WhisperModel] entry.
 *
 * Both bundled and imported models must be multilingual GGML models. English-only (".en")
 * models would ignore any non-English language request.
 */
enum class WhisperModel(val assetPath: String, val displayName: String) {
    BASE("models/whisper/ggml-base-q8_0.bin", "Base Q8_0 (bundled)");

    companion object {
        val DEFAULT = BASE

        /**
         * True if [modelSpec] points to a model imported from the file system.
         * Imported models are copied into app-internal storage and referenced by an absolute
         * file path; bundled models use a relative asset path.
         */
        fun isCustom(modelSpec: String?): Boolean =
            modelSpec != null && modelSpec.startsWith("/")
    }
}
