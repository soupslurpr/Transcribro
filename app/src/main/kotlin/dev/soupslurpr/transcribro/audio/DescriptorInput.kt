package dev.soupslurpr.transcribro.audio

import android.os.ParcelFileDescriptor
import android.system.Os
import android.system.OsConstants
import android.system.StructPollfd
import java.io.InputStream

/** Finite polling lets a stopped session leave a pipe even when its writer never closes. */
internal class DescriptorInput(
    private val descriptor: ParcelFileDescriptor,
    private val cancelled: () -> Boolean,
) : InputStream() {
    private val poll =
        arrayOf(
            StructPollfd().apply {
                fd = descriptor.fileDescriptor
                events = OsConstants.POLLIN.toShort()
            }
        )

    override fun read(): Int =
        ByteArray(1).let {
            if (read(it, 0, 1) < 0) {
                -1
            } else {
                it[0].toInt() and 255
            }
        }

    override fun read(bytes: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        while (!cancelled()) {
            if (Os.poll(poll, 100) == 0) continue
            val count = Os.read(descriptor.fileDescriptor, bytes, offset, length)
            return if (count == 0) -1 else count
        }
        return -1
    }

    override fun close() = descriptor.close()
}
