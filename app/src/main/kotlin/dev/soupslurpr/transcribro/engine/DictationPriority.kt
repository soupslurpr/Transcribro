package dev.soupslurpr.transcribro.engine

import java.io.Closeable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Process-local leases: a file yields until every interactive recognition session finishes. */
internal object DictationPriority {
    private val owners = mutableSetOf<Any>()
    private val mutableActive = MutableStateFlow(false)
    val active = mutableActive.asStateFlow()

    @Synchronized
    fun acquire(): Closeable {
        val owner = Any()
        owners.add(owner)
        mutableActive.value = true
        return Closeable {
            synchronized(this) {
                owners.remove(owner)
                mutableActive.value = owners.isNotEmpty()
            }
        }
    }
}
