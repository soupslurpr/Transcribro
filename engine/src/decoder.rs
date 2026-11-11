//! Gate 4: the decoder, replicating medium.cpp's `Medium::decoder`
//! operation-for-operation.
//!
//! Embedding: the tied uint8 affine embedding (packed64 layout arithmetic
//! in model::embedding_code), dequantized per element. Rotary: Eigen
//! ArrayXf sin/cos over float(past + r) * frequency[i / 2], applied to
//! the first 32 dims of each head of Q and K with plain mul/add
//! (-ffp-contract=off forms). Projections: the Gate 3 integer pipeline
//! (one shared quantization for Q/K/V per layer, per-projection
//! quantization elsewhere). Attention: the shared medium.cpp attention
//! models (ascending-FMA score dot with the scale multiplied once at the
//! end, kernels::softmax_row, kernels::gemm_nn) with the decoder's exact
//! 0.125f scale. Head: MlasGemm column tiles of 128 over the tiled FP32
//! head matrix, accumulated by kernels::gemm_nn; predictions use
//! token_selection.h's vocabulary-order running scan. At rows == 1 the
//! AV and head GEMMs take MLAS's M1 kernel instead (kernels::m1_dot) —
//! probe-fitted at Gate 4; the scores GEMM keeps the FMA chain at every
//! row count.

use crate::encoder::{heads, normalize, project_s, quantize, run_s, unheads};
use crate::kernels;
use crate::model::{embedding_code, Model};

pub const CACHE_CAPACITY: usize = 257;
const ATTN_SCALE: f32 = 0.125f32;

/// medium.cpp `DecoderCache`: per-layer K/V arenas with a 257-position
/// stride, zero-initialized, plus the count of valid positions.
pub struct Cache {
    pub count: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            count: 0,
            keys: vec![0.0; 14 * 640 * CACHE_CAPACITY],
            values: vec![0.0; 14 * 640 * CACHE_CAPACITY],
        }
    }
}

/// medium.cpp `attention` (generic x86 path): per head, scores GEMM
/// (TransB), mask to -inf, MlasComputeSoftmax per row, values GEMM.
/// Mask 1 allows c <= past + r; any other mask value allows everything
/// (the decoder only uses 0 and 1). `head_stride` is in elements.
#[allow(clippy::too_many_arguments)]
fn attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    memory: usize,
    mask: i32,
    past: usize,
    head_stride: usize,
) -> Vec<f32> {
    let _pt = crate::prof::Timer::start();
    let _site_mask = mask;
    crate::prof::add(&crate::prof::ATTN_DEC_PAIRS, (10 * rows * memory) as u64);
    let mut out = vec![0.0f32; rows * 640];
    let mut scores = vec![0.0f32; rows * memory];
    for h in 0..10 {
        let q = &queries[h * rows * 64..(h + 1) * rows * 64];
        let k = &keys[h * head_stride..h * head_stride + memory * 64];
        let v = &values[h * head_stride..h * head_stride + memory * 64];
        for r in 0..rows {
            for c in 0..memory {
                let total = kernels::dot_asc(&q[r * 64..(r + 1) * 64], &k[c * 64..(c + 1) * 64]);
                scores[r * memory + c] = ATTN_SCALE * total;
            }
        }
        for r in 0..rows {
            let row = &mut scores[r * memory..(r + 1) * memory];
            for (c, s) in row.iter_mut().enumerate() {
                if mask == 1 && c > past + r {
                    *s = f32::NEG_INFINITY;
                }
            }
            kernels::softmax_row(row);
        }
        if rows == 1 {
            #[cfg(target_arch = "aarch64")]
            {
                // Rank-1 form of the same accumulation: out[j] =
                // sum_c scores[c] * v[c * 64 + j], c ascending, NEON
                // over the contiguous value rows, one accumulator per
                // 4-element position block.
                use std::arch::aarch64::*;
                let orow = &mut out[h * 64..(h + 1) * 64];
                unsafe {
                    let mut acc = [vdupq_n_f32(0.0); 16];
                    for c in 0..memory {
                        let s = scores[c];
                        if s == 0.0 {
                            continue;
                        }
                        let sv = vdupq_n_f32(s);
                        let vrow = v.as_ptr().add(c * 64);
                        for i in 0..16 {
                            acc[i] = vfmaq_f32(acc[i], sv, vld1q_f32(vrow.add(4 * i)));
                        }
                    }
                    for i in 0..16 {
                        vst1q_f32(orow.as_mut_ptr().add(4 * i), acc[i]);
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                // MLAS dispatches M == 1 NoTrans SGEMM to the M1 kernel
                // (kernels::m1_dot; Gate 4 probe fit at K in {2, 17, 33,
                // 224, 602}, Gate 5b correction for K % 4 == 3 - see
                // kernels.rs).
                for j in 0..64 {
                    out[h * 64 + j] = kernels::m1_dot(&scores, v, j, 64, 0.0, false);
                }
            }
        } else {
            kernels::gemm_nn(
                &scores,
                rows,
                memory,
                v,
                64,
                &mut out[h * rows * 64..(h + 1) * rows * 64],
            );
        }
    }
    _pt.stop(if _site_mask == 1 {
        &crate::prof::ATTN_DEC_SELF_NS
    } else {
        &crate::prof::ATTN_DEC_CROSS_NS
    });
    out
}


/// One 128-column head GEMM block: MlasGemm(rows, 128, 640) over the
/// tiled head matrix. At rows == 1 MLAS takes the M1 kernel
/// (kernels::m1_dot; probe-fitted at Gates 4/5b, 0 mismatches; gemm_nn
/// fails 120 of 128 elements there).
/// Head logit dot: sum_k x[k] * (code[k] - zero). NEON form on
/// aarch64 (4 partial accumulators over the 640-wide row, codes
/// widened u8 -> f32 in registers); the plain ascending sum elsewhere.
#[cfg(target_arch = "aarch64")]
fn head_dot(x: &[f32], codes: &[u8], zero: i32) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let zv = vdupq_n_f32(zero as f32);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut j = 0usize;
        while j + 16 <= x.len() {
            let c16 = vld1q_u8(codes.as_ptr().add(j));
            let lo16 = vmovl_u8(vget_low_u8(c16));
            let hi16 = vmovl_u8(vget_high_u8(c16));
            let q0 = vsubq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16)))), zv);
            let q1 = vsubq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16)))), zv);
            let q2 = vsubq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16)))), zv);
            let q3 = vsubq_f32(vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16)))), zv);
            acc0 = vfmaq_f32(acc0, vld1q_f32(x.as_ptr().add(j)), q0);
            acc1 = vfmaq_f32(acc1, vld1q_f32(x.as_ptr().add(j + 4)), q1);
            acc2 = vfmaq_f32(acc2, vld1q_f32(x.as_ptr().add(j + 8)), q2);
            acc3 = vfmaq_f32(acc3, vld1q_f32(x.as_ptr().add(j + 12)), q3);
            j += 16;
        }
        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut lanes = [0.0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), s);
        let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while j < x.len() {
            total += x[j] * (codes[j] as i32 - zero) as f32;
            j += 1;
        }
        total
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn head_dot(x: &[f32], codes: &[u8], zero: i32) -> f32 {
    let mut sum = 0.0f32;
    for k in 0..x.len() {
        sum += x[k] * (codes[k] as i32 - zero) as f32;
    }
    sum
}

/// Widen 16 u8 codes to four f32x4 vectors of (code - zero),
/// exactly as `head_dot` does (small integers: exact in f32).
#[cfg(target_arch = "aarch64")]
unsafe fn head_widen16(
    cp: *const u8,
    zv: std::arch::aarch64::float32x4_t,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    unsafe {
        let c16 = vld1q_u8(cp);
        let lo16 = vmovl_u8(vget_low_u8(c16));
        let hi16 = vmovl_u8(vget_high_u8(c16));
        [
            vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16)))),
                zv,
            ),
            vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16)))),
                zv,
            ),
            vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16)))),
                zv,
            ),
            vsubq_f32(
                vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16)))),
                zv,
            ),
        ]
    }
}

/// Tiled output-head kernel for the `Codes` store: computes one
/// 128-column block ([rows][128] output) from the block's codes
/// ([128][640]). Per logit the accumulation is bit-identical to
/// `head_dot` (four partial vectors over 16-wide j blocks,
/// combined ((p0 + p1) + (p2 + p3)) then ((l0 + l1) + (l2 + l3)),
/// scaled at the end; depth 640 has no scalar tail). The tiling
/// only shares work: x loads across a 4-vocab tile when rows == 1,
/// and the code widening across rows (in chunks of 4) otherwise.
#[cfg(target_arch = "aarch64")]
unsafe fn head_codes(
    x: &[f32],
    rows: usize,
    codes: &[u8],
    zero: i32,
    scale: f32,
    block: &mut [f32],
) {
    use std::arch::aarch64::*;
    let zv = vdupq_n_f32(zero as f32);
    unsafe {
        if rows == 1 {
            for vt in (0..128).step_by(4) {
                let mut pv = [[vdupq_n_f32(0.0); 4]; 4];
                for j16 in 0..40 {
                    let xb = x.as_ptr().add(j16 * 16);
                    let xv = [
                        vld1q_f32(xb),
                        vld1q_f32(xb.add(4)),
                        vld1q_f32(xb.add(8)),
                        vld1q_f32(xb.add(12)),
                    ];
                    for vi in 0..4 {
                        let q = head_widen16(codes.as_ptr().add((vt + vi) * 640 + j16 * 16), zv);
                        for g in 0..4 {
                            pv[vi][g] = vfmaq_f32(pv[vi][g], xv[g], q[g]);
                        }
                    }
                }
                for vi in 0..4 {
                    let s = vaddq_f32(
                        vaddq_f32(pv[vi][0], pv[vi][1]),
                        vaddq_f32(pv[vi][2], pv[vi][3]),
                    );
                    let mut lanes = [0.0f32; 4];
                    vst1q_f32(lanes.as_mut_ptr(), s);
                    block[vt + vi] = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) * scale;
                }
            }
            return;
        }
        for rchunk in (0..rows).step_by(4) {
            let rn = (rows - rchunk).min(4);
            for v in 0..128 {
                let mut pv = [[vdupq_n_f32(0.0); 4]; 4];
                for j16 in 0..40 {
                    let q = head_widen16(codes.as_ptr().add(v * 640 + j16 * 16), zv);
                    for ri in 0..rn {
                        let xb = x.as_ptr().add((rchunk + ri) * 640 + j16 * 16);
                        for g in 0..4 {
                            let xv = vld1q_f32(xb.add(4 * g));
                            pv[ri][g] = vfmaq_f32(pv[ri][g], xv, q[g]);
                        }
                    }
                }
                for ri in 0..rn {
                    let s = vaddq_f32(
                        vaddq_f32(pv[ri][0], pv[ri][1]),
                        vaddq_f32(pv[ri][2], pv[ri][3]),
                    );
                    let mut lanes = [0.0f32; 4];
                    vst1q_f32(lanes.as_mut_ptr(), s);
                    block[(rchunk + ri) * 128 + v] =
                        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) * scale;
                }
            }
        }
    }
}

/// Lever 8 engagement: TS_HEAD_INT8 set (and not "0") plus a
/// load-built head Matrix. ADOPTED 2026-11-10 under the user's
/// conditional grant (corpus ΔWER gate passed round 3; activation
/// evidence pack in GATE6-SPEED3.md) -- default ON; the env var
/// remains a kill switch.
fn head_int8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TS_HEAD_INT8").map(|v| v != "0").unwrap_or(true))
}

fn head_block(model: &Model, normalized: &[f32], rows: usize, column: usize, block: &mut [f32]) {
    match &model.head {
        crate::model::HeadStore::Tiled(head) => {
            let head = head.as_slice();
            let b = &head[column * 640..column * 640 + 640 * 128];
            if rows == 1 {
                for j in 0..128 {
                    block[j] = kernels::m1_dot(normalized, b, j, 128, 0.0, false);
                }
            } else {
                kernels::gemm_nn(normalized, rows, 640, b, 128, block);
            }
        }
        crate::model::HeadStore::Codes { codes, scale, zero, tiled } => {
            let codes = codes.as_slice();
            let tiled = tiled.as_slice();
            // CompactHead form (aarch64): logit(v) = sum_j x_j *
            // (code(v, j) - zero) * scale. The tiled kernel keeps
            // head_dot's exact per-logit accumulation (four j-lane
            // partials over 16-blocks, combined pairwise) while
            // sharing the u8->f32 widening across a 4-vocab tile
            // (and across rows in bulk calls).
            #[cfg(target_arch = "aarch64")]
            {
                if rows > 1 && !tiled.is_empty() {
                    // Bulk head as FP32 GEMM over the dequantized
                    // tiles (the C++ oracle's bulk structure; the
                    // accumulation order differs from head_dot's
                    // partials -- verified by the parity protocol).
                    let b = &tiled[column * 640..column * 640 + 640 * 128];
                    kernels::gemm_nn(normalized, rows, 640, b, 128, block);
                } else {
                    let b = &codes[column * 640..column * 640 + 640 * 128];
                    unsafe {
                        head_codes(normalized, rows, b, *zero, *scale, block);
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for r in 0..rows {
                    let x = &normalized[r * 640..(r + 1) * 640];
                    for j in 0..128 {
                        let v = column + j;
                        let crow = &codes[v * 640..(v + 1) * 640];
                        block[r * 128 + j] = head_dot(x, crow, *zero) * scale;
                    }
                }
            }
        }
    }
}

enum Head<'a> {
    Logits(&'a mut [f32]),
    Predictions(&'a mut [i64]),
}

enum Dest<'a> {
    Caller { out_k: &'a mut [f32], out_v: &'a mut [f32] },
    Cache(&'a mut Cache),
}


// ---- TS_HEAD_PROBE (Round 4, Track A): activation-level int8-head
// evidence. When TS_HEAD_PROBE=<jsonl path> is set, every head
// evaluation in run_decoder additionally computes BOTH production
// logit rows from the same normalized inputs (float via head_block
// at the call's real shape; int8 via quantize + project_s over the
// head Matrix, production block loop) and appends one JSON record
// per row. Purely observational: no production value flows back.

static PROBE_FILE: std::sync::OnceLock<
    Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>,
> = std::sync::OnceLock::new();
static PROBE_CALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn probe_file() -> Option<&'static std::sync::Mutex<std::io::BufWriter<std::fs::File>>> {
    PROBE_FILE
        .get_or_init(|| {
            let path = std::env::var("TS_HEAD_PROBE").ok()?;
            let f = std::fs::File::create(path).ok()?;
            Some(std::sync::Mutex::new(std::io::BufWriter::new(f)))
        })
        .as_ref()
}

fn probe_fnv1a(x: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in x {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// (max, mean, p99) of |a - b| elementwise; p99 from a 512-bin
/// histogram over [0, max] (upper edge of the crossing bin).
fn probe_delta_stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
        }
        sum += d as f64;
    }
    let mean = (sum / a.len() as f64) as f32;
    if max == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mut bins = [0u32; 512];
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let mut bin = (d / max * 512.0) as usize;
        if bin > 511 {
            bin = 511;
        }
        bins[bin] += 1;
    }
    let target = (a.len() as f64 * 0.99).ceil() as u32;
    let mut cum = 0u32;
    let mut p99 = max;
    for (i, &c) in bins.iter().enumerate() {
        cum += c;
        if cum >= target {
            p99 = (i + 1) as f32 / 512.0 * max;
            break;
        }
    }
    (max, mean, p99)
}

fn head_probe_eval(model: &Model, normalized: &[f32], rows: usize, past: usize, tokens: &[i64]) {
    use std::io::Write;
    let Some(mu) = probe_file() else { return };
    let Some(hm) = &model.head_matrix else { return };
    let call = PROBE_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let clip = std::env::var("TS_HEAD_PROBE_CLIP").unwrap_or_default();
    let drive = if head_int8_enabled() { "i8" } else { "f32" };
    // Float logits for the whole call, production shape.
    let mut logits_f = vec![0.0f32; rows * 32768];
    {
        let mut block = vec![0.0f32; rows * 128];
        for column in (0..32768).step_by(128) {
            head_block(model, normalized, rows, column, &mut block);
            for r in 0..rows {
                logits_f[r * 32768 + column..r * 32768 + column + 128]
                    .copy_from_slice(&block[r * 128..(r + 1) * 128]);
            }
        }
    }
    // Int8 logits for the whole call, production block loop.
    let mut logits_i = vec![0.0f32; rows * 32768];
    let mut done = 0usize;
    while done < rows {
        let take = (rows - done).min(256);
        let q = match quantize(&normalized[done * 640..(done + take) * 640], take, 640) {
            Ok(q) => q,
            Err(_) => return,
        };
        let out = project_s(hm, &q, take, 1);
        logits_i[done * 32768..(done + take) * 32768].copy_from_slice(&out);
        done += take;
    }
    let toks: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    let toks = toks.join(",");
    let mut text = String::new();
    for r in 0..rows {
        let x = &normalized[r * 640..(r + 1) * 640];
        let lf = &logits_f[r * 32768..(r + 1) * 32768];
        let li = &logits_i[r * 32768..(r + 1) * 32768];
        // Float top-1 / top-2, first-maximum scan order.
        let (mut t1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (i, &v) in lf.iter().enumerate() {
            if v > v1 {
                v2 = v1;
                v1 = v;
                t1 = i;
            } else if v > v2 {
                v2 = v;
            }
        }
        let t2 = (0..32768).find(|&i| i != t1 && lf[i] == v2).unwrap_or(usize::MAX);
        // Int8 argmax, first-maximum scan order.
        let (mut ia, mut iv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in li.iter().enumerate() {
            if v > iv {
                iv = v;
                ia = i;
            }
        }
        let (d_max, d_mean, d_p99) = probe_delta_stats(lf, li);
        let row_scale = lf.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // Input-side delta of the int8 path on this row.
        let (inq_max, inq_mean, inq_p99) = match quantize(x, 1, 640) {
            Ok(q1) => {
                let dq = q1.dequantized_row(0, 640);
                probe_delta_stats(x, &dq)
            }
            Err(_) => (0.0, 0.0, 0.0),
        };
        text.push_str(&format!(
            "{{\"clip\":\"{clip}\",\"drive\":\"{drive}\",\"call\":{call},\"row\":{r},\"rows\":{rows},\"past\":{past},\"tokens\":[{toks}],\"in_sha\":{},\"inq_max\":{inq_max:.6e},\"inq_mean\":{inq_mean:.6e},\"inq_p99\":{inq_p99:.6e},\"f_top1\":{t1},\"f_top2\":{t2},\"f_margin\":{:.6e},\"f_top1_val\":{v1:.6e},\"i_argmax\":{ia},\"d_at_f1\":{:.6e},\"d_at_chal\":{:.6e},\"d_max\":{d_max:.6e},\"d_mean\":{d_mean:.6e},\"d_p99\":{d_p99:.6e},\"row_scale\":{row_scale:.6e},\"flip\":{}}}\n",
            probe_fnv1a(x),
            v1 - v2,
            li[t1] - lf[t1],
            li[ia] - lf[ia],
            if ia != t1 { 1 } else { 0 },
        ));
    }
    if let Ok(mut w) = mu.lock() {
        let _ = w.write_all(text.as_bytes());
        let _ = w.flush();
    }
}

/// The decoder body shared by both ABI entries; validation matching
/// medium.cpp's requires happens in the entry functions below.
#[allow(clippy::too_many_arguments)]
fn run_decoder(
    model: &Model,
    tokens: &[i64],
    past: usize,
    old: Option<(&[f32], &[f32])>,
    cross_k: &[f32],
    cross_v: &[f32],
    frames: usize,
    mut head: Head,
    mut dest: Dest,
) -> Result<(), String> {
    let rows = tokens.len();
    let _pt = crate::prof::Timer::start();
    crate::prof::add(&crate::prof::DEC_CALLS, 1);
    if rows == 1 {
        crate::prof::add(&crate::prof::DEC_R1_CALLS, 1);
    } else {
        crate::prof::add(&crate::prof::DEC_BULK_TOKENS, rows as u64);
    }
    let total = past + rows;
    let stride = match &dest {
        Dest::Cache(_) => CACHE_CAPACITY,
        Dest::Caller { .. } => total,
    };
    // Rotary tables: Eigen sin/cos of float(past + r) * frequency[i / 2].
    let mut sines = vec![0.0f32; rows * 32];
    let mut cosines = vec![0.0f32; rows * 32];
    for r in 0..rows {
        for i in 0..32 {
            let angle = (past + r) as f32 * model.rotary[i / 2];
            sines[r * 32 + i] = kernels::eigen_sin(angle);
            cosines[r * 32 + i] = kernels::eigen_cos(angle);
        }
    }
    // Embedding lookup.
    let mut hidden = vec![0.0f32; rows * 640];
    for r in 0..rows {
        let token = tokens[r];
        if token < 0 || token >= 32768 {
            return Err("invalid token".into());
        }
        for j in 0..640 {
            let code = embedding_code(
                &model.embedding_codes,
                model.embedding_packed,
                token as usize,
                j,
            );
            hidden[r * 640 + j] =
                (code as i32 - model.embedding_zero) as f32 * model.embedding_scale;
        }
    }
    for layer in 0..14 {
        let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[layer * 3]);
        let q_in = quantize(&normalized, rows, 640)?;
        let qs = project_s(&model.decoder_matrices[layer * 8], &q_in, rows, 1);
        let ks = project_s(&model.decoder_matrices[layer * 8 + 1], &q_in, rows, 1);
        let vs = project_s(&model.decoder_matrices[layer * 8 + 2], &q_in, rows, 1);
        let _tshuf = crate::prof::Timer::start();
        let mut qheads = heads(&qs, rows);
        let mut kheads = heads(&ks, rows);
        let vheads = heads(&vs, rows);
        crate::scratch::put_f32(qs);
        crate::scratch::put_f32(ks);
        crate::scratch::put_f32(vs);
        crate::scratch::put_f32(normalized);
        _tshuf.stop(&crate::prof::DEC_SHUF_NS);
        for h in 0..10 {
            for r in 0..rows {
                for pair in 0..16 {
                    let co = cosines[r * 32 + pair * 2];
                    let si = sines[r * 32 + pair * 2];
                    let at = (h * rows + r) * 64 + pair * 2;
                    for vec in [&mut qheads, &mut kheads] {
                        let a = vec[at];
                        let b = vec[at + 1];
                        vec[at] = a * co + (-b) * si;
                        vec[at + 1] = b * co + a * si;
                    }
                }
            }
        }
        // Append this layer's K/V into the store (copying the caller's
        // past states first in uncached mode), then self-attend over it.
        let attended = {
            let (store_k, store_v): (&mut [f32], &mut [f32]) = match &mut dest {
                Dest::Cache(cache) => (&mut cache.keys, &mut cache.values),
                Dest::Caller { out_k, out_v } => (&mut **out_k, &mut **out_v),
            };
            let layer_base = layer * stride * 640;
            for h in 0..10 {
                if past > 0 {
                    if let Some((old_k, old_v)) = old {
                        let src = (layer * 10 + h) * past * 64;
                        let dst = layer_base + h * stride * 64;
                        store_k[dst..dst + past * 64]
                            .copy_from_slice(&old_k[src..src + past * 64]);
                        store_v[dst..dst + past * 64]
                            .copy_from_slice(&old_v[src..src + past * 64]);
                    }
                }
                let dst = layer_base + (h * stride + past) * 64;
                store_k[dst..dst + rows * 64]
                    .copy_from_slice(&kheads[h * rows * 64..(h + 1) * rows * 64]);
                store_v[dst..dst + rows * 64]
                    .copy_from_slice(&vheads[h * rows * 64..(h + 1) * rows * 64]);
            }
            attention(
                &qheads,
                &store_k[layer_base..layer_base + stride * 640],
                &store_v[layer_base..layer_base + stride * 640],
                rows,
                total,
                1,
                past,
                stride * 64,
            )
        };
        let _tshuf2 = crate::prof::Timer::start();
        let joined = unheads(&attended, rows);
        _tshuf2.stop(&crate::prof::DEC_SHUF_NS);
        let projected = run_s(&model.decoder_matrices[layer * 8 + 3], &joined, rows, 640, 1)?;
        let _tres = crate::prof::Timer::start();
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        _tres.stop(&crate::prof::DEC_RESID_NS);
        crate::scratch::put_f32(projected);
        // Cross-attention over this layer's slice of the cross-KV.
        let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[layer * 3 + 1]);
        let qs = run_s(&model.decoder_matrices[layer * 8 + 4], &normalized, rows, 640, 1)?;
        let _tshuf3 = crate::prof::Timer::start();
        let qheads = heads(&qs, rows);
        crate::scratch::put_f32(qs);
        crate::scratch::put_f32(normalized);
        _tshuf3.stop(&crate::prof::DEC_SHUF_NS);
        let cross_base = layer * frames * 640;
        let attended = attention(
            &qheads,
            &cross_k[cross_base..cross_base + frames * 640],
            &cross_v[cross_base..cross_base + frames * 640],
            rows,
            frames,
            0,
            0,
            frames * 64,
        );
        let _tshuf4 = crate::prof::Timer::start();
        let joined = unheads(&attended, rows);
        _tshuf4.stop(&crate::prof::DEC_SHUF_NS);
        let projected = run_s(&model.decoder_matrices[layer * 8 + 5], &joined, rows, 640, 1)?;
        let _tres = crate::prof::Timer::start();
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        _tres.stop(&crate::prof::DEC_RESID_NS);
        crate::scratch::put_f32(projected);
        // SwiGLU FFN: logistic on the second half, gated by the first.
        let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[layer * 3 + 2]);
        let expanded = run_s(&model.decoder_matrices[layer * 8 + 6], &normalized, rows, 640, 1)?;
        let width = 2560;
        let mut gated = crate::scratch::take_f32(rows * width);
        gated.clear();
        gated.resize(rows * width, 0.0);
        // SwiGLU gate. On aarch64 the 4-lane form is bit-identical
        // to the scalar loop on finite inputs (kernels::logistic
        // and its vector twin share the Horner/FMA sequence and
        // the exact divide); other targets keep the scalar loop.
        let _tgate = crate::prof::Timer::start();
        #[cfg(target_arch = "aarch64")]
        {
            kernels::swiglu_gate_neon(&expanded, rows, width, &mut gated);
        }
        _tgate.stop(&crate::prof::DEC_GATE_NS);
        #[cfg(not(target_arch = "aarch64"))]
        {
            for r in 0..rows {
                for j in 0..width {
                    let x = expanded[r * width * 2 + width + j];
                    let silu = x * kernels::logistic(x);
                    gated[r * width + j] = expanded[r * width * 2 + j] * silu;
                }
            }
        }
        let projected = run_s(&model.decoder_matrices[layer * 8 + 7], &gated, rows, width, 1)?;
        crate::scratch::put_f32(gated);
        crate::scratch::put_f32(expanded);
        crate::scratch::put_f32(normalized);
        let _tres = crate::prof::Timer::start();
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        _tres.stop(&crate::prof::DEC_RESID_NS);
        crate::scratch::put_f32(projected);
    }
    let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[42]);
    if probe_file().is_some() {
        head_probe_eval(model, &normalized, rows, past, tokens);
    }
    let _pth = crate::prof::Timer::start();
    crate::prof::add(&crate::prof::HEAD_MACS, (rows as u64) * 32768 * 640);
    let mut head_done = false;
    if head_int8_enabled() {
        if let Some(hm) = &model.head_matrix {
            // Int8 head (lever 8): quantize the normalized rows
            // (the same activation quantization every encoder
            // projection applies) and project over the head's
            // own codes; the epilogue's i32 corrections make
            // the weight side exact. Predictions keep the
            // first-maximum scan order over the full logit row.
            let mut done = 0usize;
            while done < rows {
                let take = (rows - done).min(256);
                let q = quantize(&normalized[done * 640..(done + take) * 640], take, 640)?;
                let out = project_s(hm, &q, take, 1);
                match &mut head {
                    Head::Logits(logits) => {
                        (*logits)[done * 32768..(done + take) * 32768].copy_from_slice(&out);
                    }
                    Head::Predictions(preds) => {
                        for r in 0..take {
                            let row = &out[r * 32768..(r + 1) * 32768];
                            let mut best = row[0];
                            let mut sel = 0i64;
                            for (i, &v) in row.iter().enumerate().skip(1) {
                                if best < v {
                                    best = v;
                                    sel = i as i64;
                                }
                            }
                            (*preds)[done + r] = sel;
                        }
                    }
                }
                done += take;
            }
            head_done = true;
        }
    }
    if !head_done {
    let mut block = crate::scratch::take_f32(rows * 128);
    block.clear();
    block.resize(rows * 128, 0.0);
    match head {
        Head::Logits(logits) => {
            for column in (0..32768).step_by(128) {
                head_block(model, &normalized, rows, column, &mut block);
                for r in 0..rows {
                    for j in 0..128 {
                        logits[r * 32768 + column + j] = block[r * 128 + j];
                    }
                }
            }
        }
        Head::Predictions(predictions) => {
            // token_selection.h: consume vocabulary columns in order,
            // retaining the first maximum exactly as std::max_element
            // does; the running (maximum, selected) pair carries across
            // blocks, never a per-block reduction.
            let mut best = vec![0.0f32; rows];
            for column in (0..32768).step_by(128) {
                head_block(model, &normalized, rows, column, &mut block);
                for r in 0..rows {
                    if column == 0 {
                        best[r] = block[r * 128];
                        predictions[r] = 0;
                    }
                    let mut maximum = best[r];
                    let mut selected = predictions[r];
                    for i in (if column == 0 { 1 } else { 0 })..128 {
                        if maximum < block[r * 128 + i] {
                            maximum = block[r * 128 + i];
                            selected = (column + i) as i64;
                        }
                    }
                    best[r] = maximum;
                    predictions[r] = selected;
                }
            }
        }
    }
    }
    _pth.stop(&crate::prof::HEAD_NS);
    if let Dest::Cache(cache) = dest {
        cache.count = total;
    }
    _pt.stop(&crate::prof::DEC_NS);
    Ok(())
}

/// The `tm_decoder` entry: caller-owned K/V buffers and full logits.
#[allow(clippy::too_many_arguments)]
pub fn decoder_uncached(
    model: &Model,
    tokens: Option<&[i64]>,
    old: Option<(&[f32], &[f32])>,
    past: usize,
    cross_k: Option<&[f32]>,
    cross_v: Option<&[f32]>,
    frames: usize,
    logits: Option<&mut [f32]>,
    out: Option<(&mut [f32], &mut [f32])>,
) -> Result<(), String> {
    let (tokens, cross_k, cross_v, logits, (out_k, out_v)) =
        match (tokens, cross_k, cross_v, logits, out) {
            (Some(t), Some(ck), Some(cv), Some(l), Some(o)) => (t, ck, cv, l, o),
            _ => return Err("null decoder buffer".into()),
        };
    if past > 0 && old.is_none() {
        return Err("null decoder buffer".into());
    }
    let rows = tokens.len();
    if rows == 0 || rows > 1024 || past > 4096 || rows > 4096 - past || frames == 0 || frames > 1024
    {
        return Err("decoder extent".into());
    }
    run_decoder(
        model,
        tokens,
        past,
        old,
        cross_k,
        cross_v,
        frames,
        Head::Logits(logits),
        Dest::Caller { out_k, out_v },
    )
}

/// The `tm_decoder_cached` entry: predictions via the cache's
/// TokenSelection state; on any decoder error the cache count resets to
/// 0, exactly as the C++ wrapper's catch does.
pub fn decoder_cached(
    model: &Model,
    cache: &mut Cache,
    tokens: &[i64],
    cross_k: Option<&[f32]>,
    cross_v: Option<&[f32]>,
    frames: usize,
    predictions: &mut [i64],
) -> Result<(), String> {
    let past = cache.count;
    let result = (|| {
        let (cross_k, cross_v) = match (cross_k, cross_v) {
            (Some(k), Some(v)) => (k, v),
            _ => return Err("null decoder buffer".into()),
        };
        let rows = tokens.len();
        if rows == 0
            || rows > 1024
            || past > 4096
            || rows > 4096 - past
            || frames == 0
            || frames > 1024
        {
            return Err("decoder extent".into());
        }
        if past > CACHE_CAPACITY || rows > CACHE_CAPACITY - past {
            return Err("decoder cache limit".into());
        }
        run_decoder(
            model,
            tokens,
            past,
            None,
            cross_k,
            cross_v,
            frames,
            Head::Predictions(predictions),
            Dest::Cache(cache),
        )
    })();
    if result.is_err() {
        cache.count = 0;
    }
    result
}
