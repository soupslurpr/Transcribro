//! Env-gated (TS_SPEED_PROFILE) stage profiler for the Gate 6 speed
//! program. Accumulates nanoseconds + work counters in atomics; dumped
//! to stderr once per process on destroy. Zero work when the env var
//! is unset beyond a OnceLock load + an Instant::now per timed region
//! (regions are component-scale, never inner loops).
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub static FRONTEND_NS: AtomicU64 = AtomicU64::new(0);
pub static VAD_NS: AtomicU64 = AtomicU64::new(0);
pub static ENC_NS: AtomicU64 = AtomicU64::new(0);
pub static ENC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static ENC_ROWS: AtomicU64 = AtomicU64::new(0);
pub static ADAPTER_NS: AtomicU64 = AtomicU64::new(0);
pub static CROSS_NS: AtomicU64 = AtomicU64::new(0);
pub static DEC_NS: AtomicU64 = AtomicU64::new(0);
pub static DEC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static DEC_R1_CALLS: AtomicU64 = AtomicU64::new(0);
pub static DEC_BULK_TOKENS: AtomicU64 = AtomicU64::new(0);
pub static QUANT_NS: AtomicU64 = AtomicU64::new(0);
pub static QUANT_ELEMS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_ACC_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_EPI_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_R1_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_R1_MACS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_BIG_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_BIG_MACS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_S0_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_S1_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_S2_NS: AtomicU64 = AtomicU64::new(0);
pub static ENC_SHUF_NS: AtomicU64 = AtomicU64::new(0);
pub static ENC_GELU_NS: AtomicU64 = AtomicU64::new(0);
pub static ENC_RESID_NS: AtomicU64 = AtomicU64::new(0);
pub static PROJ_MACS_ENC: AtomicU64 = AtomicU64::new(0);
pub static PROJ_MACS_DEC: AtomicU64 = AtomicU64::new(0);
pub static PROJ_MACS_X: AtomicU64 = AtomicU64::new(0);
pub static ATTN_ENC_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_ENC_PAIRS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_DEC_SELF_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_DEC_CROSS_NS: AtomicU64 = AtomicU64::new(0);
pub static ATTN_DEC_PAIRS: AtomicU64 = AtomicU64::new(0);
pub static HEAD_NS: AtomicU64 = AtomicU64::new(0);
pub static DEC_SHUF_NS: AtomicU64 = AtomicU64::new(0);
pub static DEC_GATE_NS: AtomicU64 = AtomicU64::new(0);
pub static DEC_RESID_NS: AtomicU64 = AtomicU64::new(0);
pub static HEAD_MACS: AtomicU64 = AtomicU64::new(0);
pub static NORM_NS: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TS_SPEED_PROFILE").is_some())
}

pub fn add(c: &AtomicU64, v: u64) {
    if enabled() {
        c.fetch_add(v, Ordering::Relaxed);
    }
}

pub struct Timer {
    start: Instant,
    on: bool,
}

impl Timer {
    pub fn start() -> Timer {
        Timer {
            start: Instant::now(),
            on: enabled(),
        }
    }
    pub fn stop(self, c: &AtomicU64) {
        if self.on {
            c.fetch_add(self.start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

pub fn dump() {
    if !enabled() {
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if ONCE.swap(true, Ordering::SeqCst) {
        return;
    }
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let ms = |c: &AtomicU64| g(c) as f64 / 1.0e6;
    let proj_macs = g(&PROJ_MACS_ENC) + g(&PROJ_MACS_DEC) + g(&PROJ_MACS_X);
    let acc_s = g(&PROJ_ACC_NS) as f64 / 1.0e9;
    eprintln!(
        "PROF frontend={:.1} vad={:.1} enc={:.1} (calls={} rows={}) adapter={:.1} cross={:.1} dec={:.1} (calls={} r1={} bulk_tok={}) norm={:.1}",
        ms(&FRONTEND_NS),
        ms(&VAD_NS),
        ms(&ENC_NS),
        g(&ENC_CALLS),
        g(&ENC_ROWS),
        ms(&ADAPTER_NS),
        ms(&CROSS_NS),
        ms(&DEC_NS),
        g(&DEC_CALLS),
        g(&DEC_R1_CALLS),
        g(&DEC_BULK_TOKENS),
        ms(&NORM_NS),
    );
    eprintln!(
        "PROF quant={:.1} (elems={}) proj_acc={:.1} proj_epi={:.1} proj_calls={} proj_macs={} (enc={} dec={} x={}) proj_rate={:.2} GMAC/s r1={:.1}ms/{} big={:.1}ms/{} site={:.1}/{:.1}/{:.1} eshuf={:.1} egelu={:.1} eresid={:.1}",
        ms(&QUANT_NS),
        g(&QUANT_ELEMS),
        ms(&PROJ_ACC_NS),
        ms(&PROJ_EPI_NS),
        g(&PROJ_CALLS),
        proj_macs,
        g(&PROJ_MACS_ENC),
        g(&PROJ_MACS_DEC),
        g(&PROJ_MACS_X),
        if acc_s > 0.0 {
            proj_macs as f64 / acc_s / 1.0e9
        } else {
            0.0
        },
        ms(&PROJ_R1_NS),
        g(&PROJ_R1_MACS),
        ms(&PROJ_BIG_NS),
        g(&PROJ_BIG_MACS),
        ms(&PROJ_S0_NS),
        ms(&PROJ_S1_NS),
        ms(&PROJ_S2_NS),
        ms(&ENC_SHUF_NS),
        ms(&ENC_GELU_NS),
        ms(&ENC_RESID_NS),
    );
    eprintln!(
        "PROF attn_enc={:.1} (pairs={}) attn_dec_self={:.1} attn_dec_cross={:.1} (pairs={}) head={:.1} (macs={}) shuf={:.1} gate={:.1} resid={:.1}",
        ms(&ATTN_ENC_NS),
        g(&ATTN_ENC_PAIRS),
        ms(&ATTN_DEC_SELF_NS),
        ms(&ATTN_DEC_CROSS_NS),
        g(&ATTN_DEC_PAIRS),
        ms(&HEAD_NS),
        ms(&DEC_SHUF_NS),
        ms(&DEC_GATE_NS),
        ms(&DEC_RESID_NS),
        g(&HEAD_MACS),
    );
}
