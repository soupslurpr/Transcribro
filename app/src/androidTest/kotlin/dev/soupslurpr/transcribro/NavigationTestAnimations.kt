package dev.soupslurpr.transcribro

import android.animation.ValueAnimator
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry

/** Make exit frames observable, restoring both Android's live scale and its stored setting. */
internal fun withNavigationAnimationScale(scale: Float, block: () -> Unit) {
    val automation = InstrumentationRegistry.getInstrumentation().uiAutomation
    fun shell(command: String) =
        ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command))
            .bufferedReader().use { it.readText().trim() }
    fun setScale(value: String) {
        shell("settings put global animator_duration_scale $value")
        val deadline = SystemClock.uptimeMillis() + 5_000
        while (ValueAnimator.getDurationScale() != value.toFloat()) {
            check(SystemClock.uptimeMillis() < deadline) { "Animation scale did not become $value" }
            SystemClock.sleep(20)
        }
    }
    val originalSetting = shell("settings get global animator_duration_scale")
    val originalRuntime = ValueAnimator.getDurationScale()
    try {
        setScale(scale.toString())
        block()
    } finally {
        // Deleting an absent key alone leaves WindowManager's live scale unchanged.
        setScale(originalSetting.takeUnless { it == "null" } ?: originalRuntime.toString())
        if (originalSetting == "null") shell("settings delete global animator_duration_scale")
    }
}
