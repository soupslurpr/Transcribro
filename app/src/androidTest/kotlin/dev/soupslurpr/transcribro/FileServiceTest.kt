package dev.soupslurpr.transcribro

import android.Manifest
import android.app.Notification
import android.app.NotificationManager
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.audio.FileTranscriptionService
import dev.soupslurpr.transcribro.audio.FileTranscriptions
import dev.soupslurpr.transcribro.engine.DictationPriority
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FileServiceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val fixture =
        Uri.parse("content://dev.soupslurpr.transcribro.test.audio/dictation-stereo.wav")
    private val notifications = context.getSystemService(NotificationManager::class.java)

    @Before
    fun prepare() {
        instrumentation.uiAutomation.grantRuntimePermission(
            context.packageName,
            Manifest.permission.POST_NOTIFICATIONS,
        )
        onMain { FileTranscriptions.clear(context) }
    }

    @After
    fun cleanup() {
        shell("input keyevent 224")
        shell("wm dismiss-keyguard")
        onMain { FileTranscriptions.clear(context) }
    }

    @Test
    fun fileResumesAfterDictationAndFinishesWithActivityDestroyedAndScreenOff() {
        val dictation = DictationPriority.acquire()
        try {
            scenario().use { activity ->
                activity.onActivity {
                    ViewModelProvider(it)[MainViewModel::class.java].importAudio(fixture)
                }
                await { current()?.paused == true && notification() != null }
                assertEquals(0L, current()!!.completedMs)
                assertEquals(
                    "Paused while dictating",
                    notification()!!.extras.getString(Notification.EXTRA_TITLE),
                )
            }
            // No ViewModel remains. The service must keep the import alive and wake the CPU itself.
            shell("input keyevent 223")
            dictation.close()
            await { current()?.active == false }
            val result = checkNotNull(current())
            assertFalse(result.message, result.failed)
            assertTrue(result.text.contains("voice", ignoreCase = true))
            assertEquals(result.durationMs, result.completedMs)
            await { notification() == null }
            shell("input keyevent 224")
            shell("wm dismiss-keyguard")
            scenario().use { activity ->
                activity.onActivity {
                    assertEquals(
                        result.id,
                        ViewModelProvider(it)[MainViewModel::class.java].file?.id,
                    )
                }
            }
        } finally {
            dictation.close()
        }
    }

    @Test
    fun notificationStopsPausedWorkAndAnOldActionCannotStopItsReplacement() {
        val dictation = DictationPriority.acquire()
        try {
            scenario().use { activity ->
                activity.onActivity {
                    ViewModelProvider(it)[MainViewModel::class.java].importAudio(fixture)
                }
                await { notification()?.actions?.isNotEmpty() == true }
                val first = checkNotNull(current()).id
                val oldStop = notification()!!.actions.single().actionIntent
                activity.onActivity {
                    ViewModelProvider(it)[MainViewModel::class.java].importAudio(fixture)
                }
                await {
                    current()?.id != first &&
                        notification()!!.actions.single().actionIntent != oldStop
                }
                val replacement = checkNotNull(current()).id
                oldStop.send()
                instrumentation.waitForIdleSync()
                assertEquals(replacement, current()?.id)
                assertTrue(current()!!.active)
                notification()!!.actions.single().actionIntent.send()
                await { current()?.active == false && notification() == null }
                dictation.close()
                Thread.sleep(500)
                assertEquals("", current()?.text)
                assertFalse(current()!!.active)
                assertEquals("Transcription stopped. Completed text is kept.", current()?.message)
            }
        } finally {
            dictation.close()
        }
    }

    private fun scenario() =
        ActivityScenario.launch<MainActivity>(
            Intent(context, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        )

    private fun current(): FileTranscription? {
        var result: FileTranscription? = null
        onMain { result = FileTranscriptions.current }
        return result
    }

    private fun notification() =
        notifications.activeNotifications
            .firstOrNull { it.id == FileTranscriptionService.NOTIFICATION_ID }
            ?.notification

    private fun await(condition: () -> Boolean) {
        val deadline = android.os.SystemClock.elapsedRealtime() + 90_000
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            if (condition()) return
            Thread.sleep(50)
        }
        fail("Background transcription did not reach the expected state: ${current()}")
    }

    private fun onMain(action: () -> Unit) = instrumentation.runOnMainSync(action)

    private fun shell(command: String) {
        android.os.ParcelFileDescriptor.AutoCloseInputStream(
                instrumentation.uiAutomation.executeShellCommand(command)
            )
            .use { it.readBytes() }
    }
}
