//! Thread-local recycling for large scratch buffers.
//!
//! Fresh multi-MB `Vec` allocations are mmap-backed: every first
//! touch faults the pages in (~2.3M minor faults per fixture bench
//! in the shipped build, ~7 s of system time). Buffers recycled
//! through here keep their pages resident. A buffer's contents are
//! never preserved -- takers must treat them as uninitialized and
//! size them themselves. The pool is a pure recycling bin: any
//! `Vec` of the right element type may be put back, regardless of
//! which function allocated it. Depth is capped so a burst of
//! large buffers cannot pin unbounded memory per thread.

use std::cell::RefCell;

const MAX_DEPTH: usize = 6;

thread_local! {
    static F32_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
    static I32_POOL: RefCell<Vec<Vec<i32>>> = const { RefCell::new(Vec::new()) };
    static I8_POOL: RefCell<Vec<Vec<i8>>> = const { RefCell::new(Vec::new()) };
}

macro_rules! impl_pool {
    ($pool:ident, $take:ident, $put:ident, $t:ty) => {
        pub fn $take(min_cap: usize) -> Vec<$t> {
            $pool.with(|p| {
                let mut p = p.borrow_mut();
                // Prefer the smallest sufficient buffer so big
                // ones stay available for big requests.
                let mut best: Option<usize> = None;
                for (i, v) in p.iter().enumerate() {
                    if v.capacity() >= min_cap
                        && best.is_none_or(|b| v.capacity() < p[b].capacity())
                    {
                        best = Some(i);
                    }
                }
                match best {
                    Some(i) => p.swap_remove(i),
                    None => Vec::with_capacity(min_cap),
                }
            })
        }
        pub fn $put(mut v: Vec<$t>) {
            $pool.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < MAX_DEPTH {
                    v.clear();
                    p.push(v);
                }
            })
        }
    };
}

impl_pool!(F32_POOL, take_f32, put_f32, f32);
impl_pool!(I32_POOL, take_i32, put_i32, i32);
impl_pool!(I8_POOL, take_i8, put_i8, i8);

/// Allocate a Vec of `n` f32 WITHOUT zeroing. Only for buffers
/// the caller provably writes in full before any read (the
/// projection outputs: both the fused SMMLA kernel and the
/// epilogue write every element). Avoids the memset + fresh-page
/// cost of `vec![0.0; n]` on the 22k-call projection path.
pub fn alloc_uninit_f32(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    // SAFETY: capacity is n; every element is written before
    // being read by the caller's dense-write contract.
    unsafe {
        v.set_len(n);
    }
    v
}
