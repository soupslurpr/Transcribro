package dev.soupslurpr.transcribro

import android.app.ActivityManager
import android.content.Context
import android.os.Process
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.soupslurpr.transcribro.speech.NativeEngine
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * IN-APP BENCH (measurement only): drives the product NativeEngine through the
 * Gate 4/6 fixture set inside the real app process, replicating the standalone
 * run_stream protocol (round512 padding, 8192-sample chunks, 16000-sample
 * analysis interval, per-clip fresh engine = run_stream's per-clip reset).
 * The JNI bridge adds its own 8192-sample drain on finish, so the shell
 * protocol's trailing-zero block is NOT fed here.
 */
@RunWith(AndroidJUnit4::class)
class InAppBenchTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    private fun pssKiB(): Int {
        val am =
            instrumentation.targetContext.getSystemService(Context.ACTIVITY_SERVICE)
                as ActivityManager
        return am.getProcessMemoryInfo(intArrayOf(Process.myPid()))[0].totalPss
    }

    private fun freqProbe(): JSONObject {
        val o = JSONObject()
        for (cpu in listOf(0, 4, 7)) {
            o.put(
                "cpu$cpu",
                try {
                    File("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq")
                        .readText()
                        .trim()
                } catch (e: Exception) {
                    "unreadable"
                },
            )
        }
        o.put(
            "self_stat_processor",
            try {
                File("/proc/self/stat").readText().split(" ")[38]
            } catch (e: Exception) {
                "unreadable"
            },
        )
        return o
    }

    private fun readClip(name: String): FloatArray {
        val bytes = instrumentation.context.assets.open("inapp/$name").use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    @Test
    fun fixtureBench() {
        val args = InstrumentationRegistry.getArguments()
        val flavor = args.getString("benchFlavor", "unknown")
        val runTag = args.getString("benchRun", "0")
        val clipNames =
            listOf("jfk.f32", "synth.f32", "quiet.f32", "silence.f32", "mix.f32", "short.f32")
        val root = JSONObject()
        root.put("flavor", flavor)
        root.put("run", runTag)
        root.put("pss_baseline_kib", pssKiB())
        val clipsJson = JSONArray()
        var wallTotal = 0.0
        var createTotal = 0.0
        var peakPss = 0
        for ((ci, name) in clipNames.withIndex()) {
            val samples = readClip(name)
            val paddedLen = ((samples.size + 511) / 512) * 512
            val cj = JSONObject()
            cj.put("name", name)
            cj.put("samples", samples.size)
            val cpu0 = Process.getElapsedCpuTime()
            val t0 = System.nanoTime()
            val engine = NativeEngine(instrumentation.targetContext.assets, previews = true)
            val createS = (System.nanoTime() - t0) / 1e9
            createTotal += createS
            cj.put("create_s", createS)
            if (ci == 0) root.put("pss_after_create_kib", pssKiB())
            val delivered = mutableListOf<String>()
            var preview = ""
            var fedSince = 0
            var wall = 0.0
            val updates = JSONArray()
            val calls = JSONArray()
            var offset = 0
            while (offset < paddedLen) {
                val n = minOf(8192, paddedLen - offset)
                val chunk = FloatArray(n) { i ->
                    val idx = offset + i
                    if (idx < samples.size) samples[idx] else 0f
                }
                val c0 = System.nanoTime()
                val result = engine.stream(chunk, finishPhrase = false)
                val callWall = (System.nanoTime() - c0) / 1e9
                wall += callWall
                fedSince += n
                val isAnalysis = fedSince >= 16000
                if (isAnalysis) fedSince = 0
                val call = JSONObject()
                call.put("n", n)
                call.put("wall_s", callWall)
                call.put("returned", result.size)
                call.put("analysis", isAnalysis)
                calls.put(call)
                if (result.isNotEmpty()) {
                    delivered += result.dropLast(1)
                    preview = result.last()
                }
                if (isAnalysis) {
                    val parts = if (preview.isNotBlank()) delivered + preview else delivered
                    val u = JSONObject()
                    u.put("text", parts.joinToString(" ").trim())
                    u.put("call_wall_s", callWall)
                    updates.put(u)
                }
                offset += n
            }
            val f0 = System.nanoTime()
            val fin = engine.stream(FloatArray(0), finishPhrase = true)
            val finWall = (System.nanoTime() - f0) / 1e9
            wall += finWall
            if (fin.isNotEmpty()) {
                delivered += fin.dropLast(1)
                preview = fin.last()
            }
            val fu = JSONObject()
            fu.put("text", delivered.joinToString(" ").trim())
            fu.put("call_wall_s", finWall)
            updates.put(fu)
            cj.put("cpu_s", (Process.getElapsedCpuTime() - cpu0) / 1000.0)
            cj.put("final_preview_tail", preview)
            cj.put("wall_s", wall)
            cj.put("updates", updates)
            cj.put("calls", calls)
            val pss = pssKiB()
            cj.put("pss_after_kib", pss)
            if (pss > peakPss) peakPss = pss
            wallTotal += wall
            clipsJson.put(cj)
            engine.close()
        }
        root.put("freq_probe", freqProbe())
        root.put("clips", clipsJson)
        root.put("wall_total_s", wallTotal)
        root.put("create_total_s", createTotal)
        root.put("peak_pss_kib", peakPss)
        val dir = File(instrumentation.targetContext.getExternalFilesDir(null), "inapp-bench")
        dir.mkdirs()
        File(dir, "result-$flavor-$runTag.json").writeText(root.toString(1))
    }
}
