package dev.soupslurpr.transcribro.recognitionservice.whisper

import java.util.Locale

/**
 * Maps an Android [Locale] / BCP-47 language tag to the language code expected by the
 * multilingual Whisper models (as bundled in whisper.cpp).
 *
 * Whisper identifies languages by their ISO 639-1 two-letter code (lowercase), e.g. "en",
 * "de", "es". [java.util.Locale] mostly returns those same codes, but it keeps a few obsolete
 * ISO codes (e.g. "iw" for Hebrew, "in" for Indonesian) which have to be remapped.
 *
 * Anything that does not resolve to a language Whisper supports falls back to [AUTO], which
 * tells Whisper to detect the spoken language itself.
 */
object WhisperLanguage {
    /** Sentinel telling Whisper to auto-detect the spoken language. */
    const val AUTO = "auto"

    /**
     * Languages offered in the "force language" picker (in display order). Every entry is a
     * code present in [SUPPORTED] and round-trips cleanly through [fromLanguageTag].
     */
    val SELECTABLE: List<String> = listOf(
        "en", "de", "es", "fr", "it", "pt", "nl", "ru", "uk", "pl", "tr", "ar", "hi", "zh",
        "ja", "ko"
    )

    /** Human-readable English name for a language code, e.g. "de" -> "German". */
    fun displayName(code: String): String =
        Locale.forLanguageTag(code).getDisplayLanguage(Locale.ENGLISH).ifBlank { code }

    /**
     * Languages supported by the multilingual Whisper models (the keys of whisper.cpp's
     * internal language table). English-only (".en") models only support "en".
     */
    private val SUPPORTED: Set<String> = setOf(
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
        "su", "yue"
    )

    /**
     * Remap obsolete or alternative ISO codes that [Locale.getLanguage] can emit onto the
     * codes Whisper actually uses.
     */
    private val REMAP: Map<String, String> = mapOf(
        "iw" to "he", // Hebrew (legacy code)
        "in" to "id", // Indonesian (legacy code)
        "ji" to "yi", // Yiddish (legacy code)
        "nb" to "no", // Norwegian Bokmål -> Norwegian
        "jv" to "jw", // Javanese (ISO 639-1 "jv" -> Whisper "jw")
    )

    /** Resolve [locale] to a Whisper language code, or [AUTO] if it is not supported. */
    fun fromLocale(locale: Locale): String {
        val raw = locale.language.lowercase(Locale.ROOT)
        val code = REMAP[raw] ?: raw
        return if (code in SUPPORTED) code else AUTO
    }

    /**
     * Resolve a BCP-47 language tag (e.g. "de-DE", as produced by
     * [android.view.inputmethod.InputMethodSubtype.getLanguageTag]) to a Whisper language code.
     * Returns [AUTO] for null/blank/unrecognized input.
     */
    fun fromLanguageTag(tag: String?): String {
        if (tag.isNullOrBlank() || tag == AUTO) return AUTO
        return try {
            fromLocale(Locale.forLanguageTag(tag))
        } catch (e: Exception) {
            AUTO
        }
    }
}
