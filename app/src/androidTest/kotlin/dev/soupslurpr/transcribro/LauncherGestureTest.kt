package dev.soupslurpr.transcribro

import android.content.Intent
import android.graphics.Rect
import android.os.SystemClock
import android.provider.Settings
import android.view.InputDevice
import android.view.MotionEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlin.math.abs
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LauncherGestureTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val automation = instrumentation.uiAutomation
    private val context = instrumentation.targetContext

    @Test
    fun bothEdgesCancelAndCompleteWithoutReversingOnRelease() {
        assertEquals(2, Settings.Secure.getInt(context.contentResolver, "navigation_mode", -1))
        ActivityScenario.launch<MainActivity>(
            Intent(context, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            }
        ).use { scenario ->
            withNavigationAnimationScale(3f) {
                var width = 0
                var y = 0f
                scenario.onActivity {
                    val bounds = it.windowManager.currentWindowMetrics.bounds
                    width = bounds.width()
                    y = bounds.centerY().toFloat()
                }
                click { it.text?.toString() == "About" }
                settledTitle("About Transcribro")
                val tolerance = width / 50
                // A short committed swipe leaves the outgoing title visible after release.
                // A farther swipe clips it off-screen before we can check the handoff.
                val dragDistance = width * .18f
                for (rightEdge in listOf(false, true)) {
                    click { it.text?.toString() == "Open-source licenses" }
                    val original = settledTitle("Open-source licenses")
                    for (cancel in listOf(true, false)) {
                        val downTime = SystemClock.uptimeMillis()
                        var pointerDown = false
                        fun touch(action: Int, distance: Float) {
                            val x = if (rightEdge) width - 2f - distance else 2f + distance
                            val event = MotionEvent.obtain(
                                downTime, SystemClock.uptimeMillis(), action, x, y, 0,
                            ).apply { source = InputDevice.SOURCE_TOUCHSCREEN }
                            try {
                                assertTrue("Android rejected the edge input", automation.injectInputEvent(event, true))
                                pointerDown = action != MotionEvent.ACTION_UP && action != MotionEvent.ACTION_CANCEL
                            } finally {
                                event.recycle()
                            }
                        }
                        touch(MotionEvent.ACTION_DOWN, 0f)
                        try {
                            repeat(12) { step ->
                                touch(MotionEvent.ACTION_MOVE, dragDistance * (step + 1) / 12)
                                SystemClock.sleep(25)
                            }
                            SystemClock.sleep(250)
                            val preview = requireNotNull(titleBounds("Open-source licenses")) {
                                "The outgoing page vanished during the held gesture"
                            }
                            val displacement = if (rightEdge) original.right - preview.right
                                else preview.left - original.left
                            assertTrue("The page did not follow the physical edge", displacement > tolerance)
                            if (cancel) {
                                repeat(12) { step ->
                                    touch(MotionEvent.ACTION_MOVE, dragDistance * (11 - step) / 12)
                                    SystemClock.sleep(25)
                                }
                                touch(MotionEvent.ACTION_UP, 0f)
                                val restored = settledTitle("Open-source licenses")
                                assertTrue("Cancellation changed the page position", abs(restored.left - original.left) <= tolerance)
                            } else {
                                touch(MotionEvent.ACTION_UP, dragDistance)
                                var previous = preview
                                var samples = 0
                                val deadline = SystemClock.uptimeMillis() + 700
                                while (SystemClock.uptimeMillis() < deadline) {
                                    titleBounds("Open-source licenses")?.let { current ->
                                        samples++
                                        assertTrue(
                                            "Back reversed after release: rightEdge=$rightEdge, previous=$previous, current=$current",
                                            if (rightEdge) current.right <= previous.right + tolerance
                                            else current.left >= previous.left - tolerance,
                                        )
                                        previous = current
                                    }
                                    SystemClock.sleep(16)
                                }
                                assertTrue("The outgoing title could not be tracked after release: rightEdge=$rightEdge", samples > 0)
                                settledTitle("About Transcribro")
                            }
                        } finally {
                            if (pointerDown) touch(MotionEvent.ACTION_CANCEL, 0f)
                        }
                    }
                }
                // A completed right-edge gesture must not change subsequent toolbar Back.
                click { it.text?.toString() == "Open-source licenses" }
                settledTitle("Open-source licenses")
                click { it.contentDescription?.toString() == "Back" }
                settledTitle("About Transcribro")
                click { it.contentDescription?.toString() == "Back" }
                awaitNode { it.text?.toString() == "Try dictation" }
            }
        }
    }

    private fun titleBounds(title: String): Rect? {
        automation.clearCache()
        val node = find(automation.rootInActiveWindow) {
            it.text?.toString() == title && Rect().also(it::getBoundsInScreen).top < 400
        } ?: return null
        return Rect().also(node::getBoundsInScreen)
    }

    private fun settledTitle(title: String): Rect {
        var previous: Rect? = null
        var stable = 0
        val deadline = SystemClock.uptimeMillis() + 10_000
        while (SystemClock.uptimeMillis() < deadline) {
            val current = titleBounds(title)
            stable = if (current != null && current == previous) stable + 1 else 0
            if (stable == 4) return requireNotNull(current)
            previous = current
            SystemClock.sleep(100)
        }
        error("Toolbar '$title' did not settle")
    }

    private fun click(predicate: (AccessibilityNodeInfo) -> Boolean) {
        var node = awaitNode(predicate)
        while (!node.isClickable && node.parent != null) node = node.parent
        assertTrue("Could not click navigation control", node.performAction(AccessibilityNodeInfo.ACTION_CLICK))
    }

    private fun awaitNode(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
        val deadline = SystemClock.uptimeMillis() + 10_000
        while (SystemClock.uptimeMillis() < deadline) {
            automation.clearCache()
            find(automation.rootInActiveWindow, predicate)?.let { return it }
            SystemClock.sleep(50)
        }
        error("Navigation control did not appear")
    }

    private fun find(node: AccessibilityNodeInfo?, predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
        if (node == null || node.packageName?.toString() != context.packageName) return null
        if (node.isVisibleToUser && predicate(node)) return node
        repeat(node.childCount) { find(node.getChild(it), predicate)?.let { return it } }
        return null
    }
}
