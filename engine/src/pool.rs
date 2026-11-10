//! Gate 6 S5: a tiny persistent worker pool for row-parallel
//! kernels. `std::thread::scope` per call measured ~30 s of system
//! time on the Pixel 8a bench (~30k spawns per clip: clone + stack
//! mmap + futex churn, and the churn keeps the scheduler from ever
//! boosting the big cores). Persistent workers stay resident, and
//! chunk claiming via one atomic counter balances the heterogeneous
//! cores (fast cores simply claim more chunks).
//!
//! Jobs are submitted by a single control thread (the stream/JNI
//! path is serialized); job closures never submit jobs themselves.
//! The completion protocol: the caller publishes the job under the
//! mutex, participates itself, then waits until every chunk is
//! claimed and no registered participant is still inside the job;
//! only then is the job pointer cleared and the job dropped.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
    total: usize,
    next: AtomicUsize,
}

fn trampoline<F>(data: *const (), i: usize)
where
    F: Fn(usize) + Sync,
{
    unsafe { (*(data as *const F))(i) }
}

// The closure reference lives in the submitting stack frame, which
// blocks in `run` until the job is fully drained -- see module docs.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct State {
    job: usize, // *const Job, 0 = none
    in_flight: usize,
}

pub struct Pool {
    state: Mutex<State>,
    cv_work: Condvar,
    cv_done: Condvar,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

pub fn pool() -> &'static Pool {
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State { job: 0, in_flight: 0 }),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        }));
        // Caller + 3 workers = the four big cores the affinity
        // guard pins compute to.
        let workers = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(2)
            .min(3);
        let addr = pool as *const Pool as usize;
        for _ in 0..workers {
            std::thread::spawn(move || worker_loop(addr));
        }
        pool
    })
}

fn worker_loop(addr: usize) {
    let pool = unsafe { &*(addr as *const Pool) };
    loop {
        let job_ptr = {
            let mut st = pool.state.lock().unwrap();
            loop {
                if st.job != 0 {
                    st.in_flight += 1;
                    break st.job;
                }
                st = pool.cv_work.wait(st).unwrap();
            }
        };
        pool.participate(job_ptr);
        let mut st = pool.state.lock().unwrap();
        st.in_flight -= 1;
        if st.in_flight == 0 {
            pool.cv_done.notify_all();
        }
    }
}

impl Pool {
    fn participate(&self, job_ptr: usize) {
        let job = unsafe { &*(job_ptr as *const Job) };
        loop {
            let i = job.next.fetch_add(1, Ordering::Relaxed);
            if i >= job.total {
                return;
            }
            unsafe { (job.call)(job.data, i) };
        }
    }

    /// Run `f(i)` for i in 0..parts across the pool plus the calling
    /// thread, blocking until all parts are done.
    pub fn run<F>(&self, parts: usize, f: &F)
    where
        F: Fn(usize) + Sync,
    {
        if parts <= 1 {
            f(0);
            return;
        }
        let job = Job {
            data: f as *const F as *const (),
            call: trampoline::<F>,
            total: parts,
            next: AtomicUsize::new(0),
        };
        let job_ptr = &job as *const Job as usize;
        {
            let mut st = self.state.lock().unwrap();
            debug_assert_eq!(st.job, 0, "pool jobs never nest");
            st.job = job_ptr;
            self.cv_work.notify_all();
        }
        self.participate(job_ptr);
        let mut st = self.state.lock().unwrap();
        loop {
            let claimed_all = job.next.load(Ordering::Relaxed) >= job.total;
            if claimed_all && st.in_flight == 0 {
                st.job = 0;
                return;
            }
            st = self.cv_done.wait(st).unwrap();
        }
    }
}

/// Convenience: split `rows` into at most `max_parts` contiguous
/// chunks (each at least `min_chunk` rows when possible) and run
/// `f(chunk_start, chunk_len)` for each on the pool.
pub fn parallel_rows(
    rows: usize,
    max_parts: usize,
    min_chunk: usize,
    f: &(dyn Fn(usize, usize) + Sync),
) {
    if rows == 0 {
        return;
    }
    let parts = (rows / min_chunk.max(1)).min(max_parts).max(1);
    if parts <= 1 {
        f(0, rows);
        return;
    }
    let chunk = rows.div_ceil(parts);
    let parts = rows.div_ceil(chunk);
    pool().run(parts, &|i| {
        let start = i * chunk;
        if start < rows {
            f(start, chunk.min(rows - start));
        }
    });
}
