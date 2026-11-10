//! Gate 6 S5 scheduling: on Android/aarch64, inference compute is
//! pinned to the big-core cluster for the duration of a call.
//!
//! Measured on the Pixel 8a (Tensor G3) from the adb-shell bench:
//! an unpinned compute thread migrates onto the little/mid cores
//! and the governor holds the big cores at their 402 MHz floor for
//! the whole run -- a ~4x end-to-end penalty that no kernel work
//! can recover. The C++ oracle does not suffer it (its runtime
//! keeps its compute resident on the big cores). The guard sets
//! the calling thread's affinity to the top-4 online cores and
//! restores the previous mask on drop; pool workers spawned while
//! a guard is held inherit the mask.

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
mod imp {
    pub struct BigCoreGuard {
        prev: libc::cpu_set_t,
    }

    pub fn big_cores() -> Option<BigCoreGuard> {
        unsafe {
            let size = std::mem::size_of::<libc::cpu_set_t>();
            let mut prev: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, size, &mut prev) != 0 {
                return None;
            }
            let ncpu = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(8);
            let first = ncpu.saturating_sub(4);
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            let mut any = false;
            for c in first..ncpu {
                if libc::CPU_ISSET(c, &prev) {
                    libc::CPU_SET(c, &mut set);
                    any = true;
                }
            }
            if !any {
                return None;
            }
            if libc::sched_setaffinity(0, size, &set) != 0 {
                return None;
            }
            Some(BigCoreGuard { prev })
        }
    }

    impl Drop for BigCoreGuard {
        fn drop(&mut self) {
            unsafe {
                libc::sched_setaffinity(
                    0,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &self.prev,
                );
            }
        }
    }
}

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
pub use imp::{big_cores, BigCoreGuard};

#[cfg(not(all(target_os = "android", target_arch = "aarch64")))]
pub struct BigCoreGuard;

#[cfg(not(all(target_os = "android", target_arch = "aarch64")))]
pub fn big_cores() -> Option<BigCoreGuard> {
    None
}
