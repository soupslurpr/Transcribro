//! Gate 3: the encoder, adapter, and cross-KV producer, replicating
//! medium.cpp's component functions operation-for-operation.
//!
//! The projections are exact integer math plus one fixed FP32 epilogue
//! (projection.h's scalar reference path, which the host's VNNI kernel
//! matches): quantize the whole activation block once (per-block min/max
//! seeded at 0, nearbyint rounding), accumulate
//! sum_j (a_j - 128) * w[j][c] + (128 - zp_a) * sums[c]
//!     - centered_sums[r] * (zp_b[c] - 128)
//! in int32, then out = float(acc) * (q.scale * scale[c]) + bias[c].
//!
//! Attention: scores are a plain ascending-FMA dot over the 64-wide head
//! dimension with the scale constant multiplied once at the end; softmax
//! is kernels::softmax_row; the value mix is kernels::gemm_nn — all three
//! probe-validated bit-for-bit against MLAS (Gate 3 log in NOTES.md).

use crate::kernels;
use crate::model::{Matrix, Model};

const ATTN_SCALE: f32 = 0.1249999925494194f32; // bits 0x3dffffff
const SQRT2: f32 = 1.4142135381698608f32; // bits 0x3fb504f3

/// A dynamically quantized activation block (projection.h `Quantized`,
/// x86 form: natural row-major layout, values stored as code - 128).
pub struct Quantized {
    values: Vec<i8>,          // [rows][k]
    centered_sums: Vec<i32>,  // [rows]: sum_j (code_j - zero)
    scale: f32,
    zero: i32,
}

/// projection.h `quantize` (x86 path): min/max over the whole block seeded
/// with 0.0, scale = (high - low) / 255 (0 -> 1), zero = clamp(nearbyint(
/// -low / scale), 0, 255), per element nearbyint(x / scale) first, then
/// + float(zero), then clamp — all divisions, never reciprocal multiplies.
pub fn quantize(x: &[f32], rows: usize, k: usize) -> Result<Quantized, String> {
    if rows == 0 || rows > 1024 || k == 0 || k > 16384 {
        return Err("unsupported activation dimensions".into());
    }
    debug_assert_eq!(x.len(), rows * k);
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    for &v in x {
        if !v.is_finite() {
            return Err("nonfinite activation".into());
        }
        if v < low {
            low = v;
        }
        if v > high {
            high = v;
        }
    }
    let mut scale = (high - low) / 255.0f32;
    if scale == 0.0 {
        scale = 1.0;
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err("unrepresentable activation range".into());
    }
    let zero = (-low / scale).round_ties_even().clamp(0.0, 255.0) as i32;
    let mut values = vec![0i8; rows * k];
    let mut centered_sums = vec![0i32; rows];
    for r in 0..rows {
        let mut sum = 0i32;
        #[cfg(target_arch = "aarch64")]
        {
            // NEON form of the identical scalar math below: vdivq is
            // the IEEE divide, vrndnq is round-ties-even, the clamps
            // and the integral conversion are exact, and the centered
            // sum is integer (order-free). Codes are stored as a - 128.
            use std::arch::aarch64::*;
            let xrow = &x[r * k..(r + 1) * k];
            let vrow = &mut values[r * k..(r + 1) * k];
            unsafe {
                let sv = vdupq_n_f32(scale);
                let zv = vdupq_n_f32(zero as f32);
                let lov = vdupq_n_f32(0.0);
                let hiv = vdupq_n_f32(255.0);
                let mut acc = vdupq_n_s32(0);
                let mut j = 0usize;
                while j + 16 <= k {
                    let mut iv = [vdupq_n_s32(0); 4];
                    for g in 0..4 {
                        let v = vld1q_f32(xrow.as_ptr().add(j + 4 * g));
                        let v = vrndnq_f32(vdivq_f32(v, sv));
                        let v = vminq_f32(vmaxq_f32(vaddq_f32(v, zv), lov), hiv);
                        iv[g] = vcvtq_s32_f32(v);
                        acc = vaddq_s32(acc, iv[g]);
                    }
                    let s0 = vqmovn_s32(vsubq_s32(iv[0], vdupq_n_s32(128)));
                    let s1 = vqmovn_s32(vsubq_s32(iv[1], vdupq_n_s32(128)));
                    let s2 = vqmovn_s32(vsubq_s32(iv[2], vdupq_n_s32(128)));
                    let s3 = vqmovn_s32(vsubq_s32(iv[3], vdupq_n_s32(128)));
                    let lo8 = vqmovn_s16(vcombine_s16(s0, s1));
                    let hi8 = vqmovn_s16(vcombine_s16(s2, s3));
                    let packed = vcombine_s8(lo8, hi8);
                    vst1q_s8(vrow.as_mut_ptr().add(j) as *mut i8, packed);
                    j += 16;
                }
                let mut lanes = [0i32; 4];
                vst1q_s32(lanes.as_mut_ptr(), acc);
                sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) - (j as i32) * zero;
                for jj in j..k {
                    let a = ((xrow[jj] / scale).round_ties_even() + zero as f32).clamp(0.0, 255.0) as i32;
                    vrow[jj] = (a - 128) as i8;
                    sum += a - zero;
                }
            }
            centered_sums[r] = sum;
            continue;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for j in 0..k {
                let a = ((x[r * k + j] / scale).round_ties_even() + zero as f32).clamp(0.0, 255.0) as i32;
                values[r * k + j] = (a - 128) as i8;
                sum += a - zero;
            }
            centered_sums[r] = sum;
        }
    }
    Ok(Quantized { values, centered_sums, scale, zero })
}

/// One row of the integer accumulation: acc[c] += sum_j a_j * w[j][c].
/// The accumulation is exact integer math, so any evaluation order --
/// scalar, NEON, or row-parallel -- produces identical results.
/// Runtime dotprod detection: `vdotq_s32` only inlines with the
/// target feature enabled; without it stdarch emits a function call
/// per dot product (measured: the whole SDOT pass ran at widening-MLA
/// speed until this dispatch was added).
#[cfg(target_arch = "aarch64")]
fn have_dotprod() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// SDOT projection kernel over the `pack_sdot` tile layout: 16
/// MACs per instruction. Register-blocked form: for each 64-output
/// block, 16 accumulators live in registers across the whole k
/// range and the packed B operand is one contiguous 256-byte read
/// per j-block. (Two earlier forms were measured on the Pixel 8a:
/// j-outer/accumulator-in-memory ~12 GMAC/s boosted, and this form
/// is the MLAS-class structure the C++ oracle uses.)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn project_row_sdot(a: &[i8], m: &Matrix, arow: &mut [i32]) {
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let p = m.packed_sdot.as_ptr();
    let tiles_per_band = n / 4;
    unsafe {
        let ap = arow.as_mut_ptr();
        let mut c0 = 0usize;
        while c0 + 64 <= n {
            let mut acc = [vdupq_n_s32(0); 16];
            for g in 0..16 {
                acc[g] = vld1q_s32(ap.add(c0 + 4 * g) as *const i32);
            }
            for j4 in 0..k / 4 {
                let aword = (a.as_ptr().add(j4 * 4) as *const i32).read_unaligned();
                if aword == 0 {
                    continue;
                }
                let avec = vreinterpretq_s8_s32(vdupq_n_s32(aword));
                let base = p.add((j4 * tiles_per_band + c0 / 4) * 16);
                for g in 0..16 {
                    acc[g] = vdotq_s32(acc[g], avec, vld1q_s8(base.add(16 * g)));
                }
            }
            for g in 0..16 {
                vst1q_s32(ap.add(c0 + 4 * g), acc[g]);
            }
            c0 += 64;
        }
        // Remainder columns (< 64): accumulator-in-memory j-outer form.
        while c0 < n {
            for j4 in 0..k / 4 {
                let aword = (a.as_ptr().add(j4 * 4) as *const i32).read_unaligned();
                if aword == 0 {
                    continue;
                }
                let avec = vreinterpretq_s8_s32(vdupq_n_s32(aword));
                let base = p.add((j4 * tiles_per_band + c0 / 4) * 16);
                for g in 0..4 {
                    let v = vdotq_s32(
                        vld1q_s32(ap.add(c0 + 4 * g) as *const i32),
                        avec,
                        vld1q_s8(base.add(16 * g)),
                    );
                    vst1q_s32(ap.add(c0 + 4 * g), v);
                }
            }
            c0 += 16;
        }
    }
}

/// Accumulate `rows` projection rows (starting at row `r0` of `q`)
/// into `acc` ([rows][n]). The aarch64 path uses the SDOT kernels
/// when the packed layout is available.
fn project_range(m: &Matrix, q: &Quantized, r0: usize, rows: usize, acc: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if m.packed_sdot.len() == m.k * m.n && m.k % 4 == 0 && m.n % 16 == 0 && have_dotprod() {
            for i in 0..rows {
                unsafe {
                    project_row_sdot(
                        &q.values[(r0 + i) * m.k..(r0 + i + 1) * m.k],
                        m,
                        &mut acc[i * m.n..(i + 1) * m.n],
                    );
                }
            }
            return;
        }
    }
    for i in 0..rows {
        project_row(
            &q.values[(r0 + i) * m.k..(r0 + i + 1) * m.k],
            m,
            &mut acc[i * m.n..(i + 1) * m.n],
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn project_row(a: &[i8], m: &Matrix, arow: &mut [i32]) {
    if m.packed_sdot.len() == m.k * m.n && m.k % 4 == 0 && m.n % 16 == 0 && have_dotprod() {
        unsafe {
            project_row_sdot(a, m, arow);
        }
        return;
    }
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let w = m.weights();
    // j-outer form: the weight matrix streams exactly once per row and
    // the accumulator row (n i32s) stays L1-resident. (The first NEON
    // form looped k inside each 16-wide output block, re-streaming the
    // weights n/16 times per row -- the dominant on-device cost at
    // Gate 6 S5, ~0.15 MACs/cycle measured.)
    for j in 0..k {
        let av = a[j];
        if av == 0 {
            continue;
        }
        let wrow = &w[j * n..(j + 1) * n];
        unsafe {
            let dup = vdupq_n_s16(av as i16);
            let dlo = vget_low_s16(dup);
            let dhi = vget_high_s16(dup);
            let ap = arow.as_mut_ptr() as *mut i32;
            let wp = wrow.as_ptr();
            let mut c = 0usize;
            while c + 16 <= n {
                let wv = vld1q_s8(wp.add(c));
                let lo = vmovl_s8(vget_low_s8(wv));
                let hi = vmovl_s8(vget_high_s8(wv));
                let a0 = vmlal_s16(vld1q_s32(ap.add(c) as *const i32), vget_low_s16(lo), dlo);
                let a1 = vmlal_s16(vld1q_s32(ap.add(c + 4) as *const i32), vget_high_s16(lo), dlo);
                let a2 = vmlal_s16(vld1q_s32(ap.add(c + 8) as *const i32), vget_low_s16(hi), dhi);
                let a3 = vmlal_s16(vld1q_s32(ap.add(c + 12) as *const i32), vget_high_s16(hi), dhi);
                vst1q_s32(ap.add(c), a0);
                vst1q_s32(ap.add(c + 4), a1);
                vst1q_s32(ap.add(c + 8), a2);
                vst1q_s32(ap.add(c + 12), a3);
                c += 16;
            }
            while c < n {
                *ap.add(c) += av as i32 * *wp.add(c) as i32;
                c += 1;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn project_row(a: &[i8], m: &Matrix, arow: &mut [i32]) {
    let (k, n) = (m.k, m.n);
    for j in 0..k {
        let av = a[j] as i32;
        if av == 0 {
            continue;
        }
        let wrow = &m.weights()[j * n..(j + 1) * n];
        for c in 0..n {
            arow[c] += av * wrow[c] as i32;
        }
    }
}

/// projection.h `Projection::multiply` semantics (integer-exact).
/// Rows are independent, so large blocks are split across threads; the
/// integer result is identical at any thread count.
pub fn project(m: &Matrix, q: &Quantized, rows: usize) -> Vec<f32> {
    let (k, n) = (m.k, m.n);
    let mut acc = vec![0i32; rows * n];
    let work = rows.saturating_mul(k).saturating_mul(n);
    // Pool engagement bar: set above every call size this
    // workload produces. Measured twice on the Pixel 8a (scope
    // threads, then a persistent pool, with and without the
    // affinity guard): multi-threaded projection is consistently
    // ~2-3x WORSE end-to-end than one hot thread in the Android
    // shell environment, so the product path is single-threaded;
    // the pool remains for potential future batch use.
    let threads = if work >= (1 << 30) && rows >= 4 {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
            .min(rows)
            .min(4)
    } else {
        1
    };
    if threads <= 1 {
        project_range(m, q, 0, rows, &mut acc);
    } else {
        let acc_addr = acc.as_mut_ptr() as usize;
        crate::pool::parallel_rows(rows, threads, 4, &|start, take| {
            let part = unsafe {
                std::slice::from_raw_parts_mut((acc_addr as *mut i32).add(start * n), take * n)
            };
            project_range(m, q, start, take, part);
        });
    }
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let total = acc[r * n + c] + (128 - q.zero) * m.sums[c] - q.centered_sums[r] * m.zeros[c];
            out[r * n + c] = total as f32 * (q.scale * m.scales[c]) + m.bias[c];
        }
    }
    out
}

/// Convenience for Matrix::run(x, ...): quantize the float input, project.
pub fn run(m: &Matrix, x: &[f32], rows: usize, k: usize) -> Result<Vec<f32>, String> {
    let q = quantize(x, rows, k)?;
    Ok(project(m, &q, rows))
}

/// medium.cpp `normalize` (x86 path): per-row ascending sum and sum of
/// squares, variance = squares/width - mean^2, deviation =
/// sqrt(variance + 1e-5), out = ((x - mean) / deviation) * gain.
pub fn normalize(x: &[f32], rows: usize, width: usize, gain: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let src = &x[r * width..(r + 1) * width];
        let mut sum = 0.0f32;
        let mut squares = 0.0f32;
        for &v in src {
            sum += v;
            squares += v * v;
        }
        let mean = sum / width as f32;
        let variance = squares / width as f32 - mean * mean;
        let deviation = (variance + 1.0e-5f32).sqrt();
        for i in 0..width {
            out[r * width + i] = ((src[i] - mean) / deviation) * gain[i];
        }
    }
    out
}

/// [rows][640] -> head-major [10][rows][64] (pure copies).
pub fn heads(x: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for h in 0..10 {
        for r in 0..rows {
            let src = &x[r * 640 + h * 64..r * 640 + h * 64 + 64];
            let dst = &mut out[(h * rows + r) * 64..(h * rows + r) * 64 + 64];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Head-major [10][rows][64] -> [rows][640] (pure copies).
pub fn unheads(x: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for h in 0..10 {
        for r in 0..rows {
            let src = &x[(h * rows + r) * 64..(h * rows + r) * 64 + 64];
            let dst = &mut out[r * 640 + h * 64..r * 640 + h * 64 + 64];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// medium.cpp `attention` (generic x86 path): per head, scores GEMM
/// (TransB), mask to -inf, MlasComputeSoftmax per row, values GEMM.
/// Mask 2: c in [r-16, r+4]; mask 3: c in [r-16, r]. Inputs are head-major.
fn attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    memory: usize,
    mask: i32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * 640];
    let mut scores = vec![0.0f32; rows * memory];
    for h in 0..10 {
        let q = &queries[h * rows * 64..(h + 1) * rows * 64];
        let k = &keys[h * memory * 64..(h + 1) * memory * 64];
        let v = &values[h * memory * 64..(h + 1) * memory * 64];
        for r in 0..rows {
            for c in 0..memory {
                let total = kernels::dot_asc(&q[r * 64..(r + 1) * 64], &k[c * 64..(c + 1) * 64]);
                scores[r * memory + c] = ATTN_SCALE * total;
            }
        }
        for r in 0..rows {
            let row = &mut scores[r * memory..(r + 1) * memory];
            for (c, s) in row.iter_mut().enumerate() {
                let allowed = match mask {
                    2 => (r as i64 - c as i64) >= -4 && r <= c + 16,
                    3 => r >= c && r <= c + 16,
                    _ => true,
                };
                if !allowed {
                    *s = f32::NEG_INFINITY;
                }
            }
            kernels::softmax_row(row);
        }
        kernels::gemm_nn(
            &scores,
            rows,
            memory,
            v,
            64,
            &mut out[h * rows * 64..(h + 1) * rows * 64],
        );
    }
    out
}

/// medium.cpp `Medium::encoder`.
pub fn encoder(model: &Model, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    let mut hidden = input.to_vec();
    for layer in 0..14 {
        let normalized = normalize(&hidden, rows, 768, &model.encoder_norms[layer * 2]);
        let q = quantize(&normalized, rows, 768)?;
        let qs = project(&model.encoder_matrices[layer * 6], &q, rows);
        let ks = project(&model.encoder_matrices[layer * 6 + 1], &q, rows);
        let vs = project(&model.encoder_matrices[layer * 6 + 2], &q, rows);
        let attended = attention(
            &heads(&qs, rows),
            &heads(&ks, rows),
            &heads(&vs, rows),
            rows,
            rows,
            if (2..=11).contains(&layer) { 3 } else { 2 },
        );
        let joined = unheads(&attended, rows);
        let projected = run(&model.encoder_matrices[layer * 6 + 3], &joined, rows, 640)?;
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        let normalized = normalize(&hidden, rows, 768, &model.encoder_norms[layer * 2 + 1]);
        let mut expanded = run(&model.encoder_matrices[layer * 6 + 4], &normalized, rows, 768)?;
        kernels::gelu_erf_inplace(&mut expanded);
        let projected = run(&model.encoder_matrices[layer * 6 + 5], &expanded, rows, 3072)?;
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
    }
    Ok(normalize(&hidden, rows, 768, &model.encoder_norms[28]))
}

/// medium.cpp `Medium::adapter`: add the dequantized position row block,
/// then the 768 -> 640 projection.
pub fn adapter(model: &Model, input: &[f32], rows: usize, offset: usize) -> Result<Vec<f32>, String> {
    let mut hidden = vec![0.0f32; rows * 768];
    for i in 0..rows * 768 {
        hidden[i] = input[i]
            + (model.adapter_positions[offset * 768 + i] as i32 - model.adapter_positions_zero)
                as f32
                * model.adapter_positions_scale;
    }
    run(&model.adapter_matrix, &hidden, rows, 768)
}

/// medium.cpp `Medium::cross`: quantize the memory once, produce per-layer
/// K and V from the shared quantized block, stored head-major per layer.
pub fn cross(model: &Model, input: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let q = quantize(input, rows, 640)?;
    let mut keys = vec![0.0f32; 14 * rows * 640];
    let mut values = vec![0.0f32; 14 * rows * 640];
    for layer in 0..14 {
        let k = project(&model.cross_matrices[layer * 2], &q, rows);
        let v = project(&model.cross_matrices[layer * 2 + 1], &q, rows);
        keys[layer * rows * 640..(layer + 1) * rows * 640].copy_from_slice(&heads(&k, rows));
        values[layer * rows * 640..(layer + 1) * rows * 640].copy_from_slice(&heads(&v, rows));
    }
    Ok((keys, values))
}

