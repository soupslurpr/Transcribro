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
pub static TILE_NS: AtomicU64 = AtomicU64::new(0);
pub static TILE_S0_NS: AtomicU64 = AtomicU64::new(0);
pub static TILE_S1_NS: AtomicU64 = AtomicU64::new(0);
pub static TILE_S2_NS: AtomicU64 = AtomicU64::new(0);
pub static TILE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static VAD_STFT_NS: AtomicU64 = AtomicU64::new(0);
pub static VAD_CONVS_NS: AtomicU64 = AtomicU64::new(0);
pub static VAD_RNN_NS: AtomicU64 = AtomicU64::new(0);
pub static VAD_MISC_NS: AtomicU64 = AtomicU64::new(0);
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
pub static STREAM_NS: AtomicU64 = AtomicU64::new(0);
pub static STREAM_CALLS: AtomicU64 = AtomicU64::new(0);
pub static INGEST_NS: AtomicU64 = AtomicU64::new(0);
pub static JNI_NS: AtomicU64 = AtomicU64::new(0);
pub static MARSHAL_NS: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        option_env!("TS_BENCH_PROF").is_some() || std::env::var_os("TS_SPEED_PROFILE").is_some()
    })
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
    if option_env!("TS_BENCH_PROF").is_none() && ONCE.swap(true, Ordering::SeqCst) {
        return;
    }
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let ms = |c: &AtomicU64| g(c) as f64 / 1.0e6;
    eprintln!(
        "PROF stream={:.1} (calls={}) ingest={:.1} jni={:.1} marshal={:.1}",
        ms(&STREAM_NS),
        g(&STREAM_CALLS),
        ms(&INGEST_NS),
        ms(&JNI_NS),
        ms(&MARSHAL_NS),
    );
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
    #[cfg(ts_bench_prof)]
    {
        // Bench builds only: mirror the headline buckets to logcat,
        // where the in-app gate can read them (stderr routing inside
        // an app process is not guaranteed). Never compiled into
        // normal builds.
        extern "C" {
            fn __android_log_print(prio: i32, tag: *const u8, fmt: *const u8, ...) -> i32;
        }
        let sec = |c: &AtomicU64| g(c) as f64 / 1.0e9;
        let line = format!(
            "fe={:.4} vad={:.4} enc={:.4} ad={:.4} cr={:.4} dec={:.4} norm={:.4} quant={:.4} proj_acc={:.4} proj_epi={:.4} proj_r1={:.4} proj_big={:.4} attn_enc={:.4} attn_self={:.4} attn_cross={:.4} head={:.4} egelu={:.4} stream={:.4} ingest={:.4} jni={:.4} marshal={:.4} enc_calls={} dec_calls={} dec_r1={} bulk_tok={} tile={:.4} tile0={:.4} tile1={:.4} tile2={:.4} tile_calls={} macs_big={} macs_enc={} macs_x={} macs_dec={} s0={:.4} s1={:.4} s2={:.4} enc_rows={} vad_s={:.4} vad_c={:.4} vad_r={:.4} vad_m={:.4}",
            sec(&FRONTEND_NS),
            sec(&VAD_NS),
            sec(&ENC_NS),
            sec(&ADAPTER_NS),
            sec(&CROSS_NS),
            sec(&DEC_NS),
            sec(&NORM_NS),
            sec(&QUANT_NS),
            sec(&PROJ_ACC_NS),
            sec(&PROJ_EPI_NS),
            sec(&PROJ_R1_NS),
            sec(&PROJ_BIG_NS),
            sec(&ATTN_ENC_NS),
            sec(&ATTN_DEC_SELF_NS),
            sec(&ATTN_DEC_CROSS_NS),
            sec(&HEAD_NS),
            sec(&ENC_GELU_NS),
            sec(&STREAM_NS),
            sec(&INGEST_NS),
            sec(&JNI_NS),
            sec(&MARSHAL_NS),
            g(&ENC_CALLS),
            g(&DEC_CALLS),
            g(&DEC_R1_CALLS),
            g(&DEC_BULK_TOKENS),
            sec(&TILE_NS),
            sec(&TILE_S0_NS),
            sec(&TILE_S1_NS),
            sec(&TILE_S2_NS),
            g(&TILE_CALLS),
            g(&PROJ_BIG_MACS),
            g(&PROJ_MACS_ENC),
            g(&PROJ_MACS_X),
            g(&PROJ_MACS_DEC),
            sec(&PROJ_S0_NS),
            sec(&PROJ_S1_NS),
            sec(&PROJ_S2_NS),
            g(&ENC_ROWS),
            sec(&VAD_STFT_NS),
            sec(&VAD_CONVS_NS),
            sec(&VAD_RNN_NS),
            sec(&VAD_MISC_NS),
        );
        if let Ok(text) = std::ffi::CString::new(line) {
            unsafe {
                __android_log_print(4, b"TSPROF\0".as_ptr(), b"%s\0".as_ptr(), text.as_ptr());
            }
        }
    }
}
