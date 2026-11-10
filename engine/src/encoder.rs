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

impl Quantized {
    /// Dequantized values of one row under the epilogue's algebra:
    /// (values + 128 - zero) * scale, elementwise. Used by the
    /// TS_HEAD_PROBE activation evidence (decoder.rs).
    pub fn dequantized_row(&self, row: usize, k: usize) -> Vec<f32> {
        let base = row * k;
        (0..k)
            .map(|j| (self.values[base + j] as i32 + 128 - self.zero) as f32 * self.scale)
            .collect()
    }
}

/// projection.h `quantize` (x86 path): min/max over the whole block seeded
/// with 0.0, scale = (high - low) / 255 (0 -> 1), zero = clamp(nearbyint(
/// -low / scale), 0, 255), per element nearbyint(x / scale) first, then
/// + float(zero), then clamp — all divisions, never reciprocal multiplies.
pub fn quantize(x: &[f32], rows: usize, k: usize) -> Result<Quantized, String> {
    let _pt = crate::prof::Timer::start();
    crate::prof::add(&crate::prof::QUANT_ELEMS, (rows * k) as u64);
    if rows == 0 || rows > 1024 || k == 0 || k > 16384 {
        return Err("unsupported activation dimensions".into());
    }
    debug_assert_eq!(x.len(), rows * k);
    let (low, high) = minmax_finite(x)?;
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
    _pt.stop(&crate::prof::QUANT_NS);
    Ok(Quantized { values, centered_sums, scale, zero })
}

/// The scale/zero derivation from `quantize`, duplicated so
/// `quantize` itself stays byte-identical; the parity gates prove
/// the copies agree.
fn quantize_scale_zero(low: f32, high: f32) -> Result<(f32, i32), String> {
    let mut scale = (high - low) / 255.0f32;
    if scale == 0.0 {
        scale = 1.0;
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err("unrepresentable activation range".into());
    }
    let zero = (-low / scale).round_ties_even().clamp(0.0, 255.0) as i32;
    Ok((scale, zero))
}

/// One row of `quantize`'s code emission (verbatim copy of the
/// per-row block in `quantize`, parameterized on slices): returns
/// the row's centered sum.
fn quantize_row_codes(xrow: &[f32], k: usize, scale: f32, zero: i32, vrow: &mut [i8]) -> i32 {
    let mut sum = 0i32;
    #[cfg(target_arch = "aarch64")]
    {
        // NEON form of the scalar math below: vdivq is the IEEE
        // divide, vrndnq is round-ties-even, the clamps and the
        // integral conversion are exact, and the centered sum is
        // integer (order-free). Codes are stored as a - 128.
        use std::arch::aarch64::*;
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
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..k {
            let a = ((xrow[j] / scale).round_ties_even() + zero as f32).clamp(0.0, 255.0) as i32;
            vrow[j] = (a - 128) as i8;
            sum += a - zero;
        }
        sum
    }
}

/// One row of normalize()'s output: ((x - mean) / deviation) *
/// gain per element, in normalize's op order (the NEON form is
/// IEEE-exact: vsub/vdiv/vmul are the same operations).
fn norm_row(src: &[f32], mean: f32, deviation: f32, gain: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        unsafe {
            let mv = vdupq_n_f32(mean);
            let dv = vdupq_n_f32(deviation);
            let mut i = 0usize;
            while i + 4 <= src.len() {
                let v = vld1q_f32(src.as_ptr().add(i));
                let g = vld1q_f32(gain.as_ptr().add(i));
                let v = vmulq_f32(vdivq_f32(vsubq_f32(v, mv), dv), g);
                vst1q_f32(out.as_mut_ptr().add(i), v);
                i += 4;
            }
            for j in i..src.len() {
                out[j] = ((src[j] - mean) / deviation) * gain[j];
            }
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..src.len() {
            out[j] = ((src[j] - mean) / deviation) * gain[j];
        }
    }
}

/// Min/max over the normalize()d values of the input without
/// materializing them: a mirror of minmax_finite (same 0.0 seed,
/// same "nonfinite activation" error) evaluated on the normed
/// values -- exactly what the unfused normalize -> quantize path
/// scans. Min/max are order-free, so the per-row grouping here
/// returns identical bounds.
fn minmax_normalized(
    x: &[f32],
    rows: usize,
    width: usize,
    gain: &[f32],
    means: &[f32],
    devs: &[f32],
) -> Result<(f32, f32), String> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        unsafe {
            let mut low4 = vdupq_n_f32(0.0);
            let mut high4 = vdupq_n_f32(0.0);
            let mut bad = vdupq_n_u32(0);
            let maxv = vdupq_n_f32(f32::MAX);
            let mut low = 0.0f32;
            let mut high = 0.0f32;
            let mut bad_any = false;
            for r in 0..rows {
                let src = &x[r * width..(r + 1) * width];
                let mv = vdupq_n_f32(means[r]);
                let dv = vdupq_n_f32(devs[r]);
                let mut i = 0usize;
                while i + 4 <= width {
                    let v = vld1q_f32(src.as_ptr().add(i));
                    let g = vld1q_f32(gain.as_ptr().add(i));
                    let v = vmulq_f32(vdivq_f32(vsubq_f32(v, mv), dv), g);
                    low4 = vminq_f32(low4, v);
                    high4 = vmaxq_f32(high4, v);
                    bad = vorrq_u32(bad, vmvnq_u32(vcleq_f32(vabsq_f32(v), maxv)));
                    i += 4;
                }
                while i < width {
                    let v = ((src[i] - means[r]) / devs[r]) * gain[i];
                    if !v.is_finite() {
                        bad_any = true;
                    }
                    if v < low {
                        low = v;
                    }
                    if v > high {
                        high = v;
                    }
                    i += 1;
                }
            }
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), low4);
            for &v in &lanes {
                if v < low {
                    low = v;
                }
            }
            vst1q_f32(lanes.as_mut_ptr(), high4);
            for &v in &lanes {
                if v > high {
                    high = v;
                }
            }
            let mut bad_lanes = [0u32; 4];
            vst1q_u32(bad_lanes.as_mut_ptr(), bad);
            if bad_any || bad_lanes.iter().any(|&b| b != 0) {
                return Err("nonfinite activation".into());
            }
            return Ok((low, high));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut low = 0.0f32;
        let mut high = 0.0f32;
        for r in 0..rows {
            let src = &x[r * width..(r + 1) * width];
            for i in 0..width {
                let v = ((src[i] - means[r]) / devs[r]) * gain[i];
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
        }
        Ok((low, high))
    }
}

/// Fused normalize + quantize (speed round 3): bit-identical to
/// quantize(&normalize(x, rows, width, gain), rows, width). The
/// normalized tensor is never materialized: per-row stats use
/// normalize's ascending scalar reductions; every normed value is
/// computed with normalize's per-element op order, once for the
/// min/max pass and once into an L1-resident row scratch for code
/// emission.
pub fn quantize_normalized(
    x: &[f32],
    rows: usize,
    width: usize,
    gain: &[f32],
) -> Result<Quantized, String> {
    crate::prof::add(&crate::prof::QUANT_ELEMS, (rows * width) as u64);
    if rows == 0 || rows > 1024 || width == 0 || width > 16384 {
        return Err("unsupported activation dimensions".into());
    }
    debug_assert_eq!(x.len(), rows * width);
    debug_assert_eq!(gain.len(), width);
    let _ptn = crate::prof::Timer::start();
    let mut means = vec![0.0f32; rows];
    let mut devs = vec![0.0f32; rows];
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
        means[r] = mean;
        devs[r] = (variance + 1.0e-5f32).sqrt();
    }
    let (low, high) = minmax_normalized(x, rows, width, gain, &means, &devs)?;
    _ptn.stop(&crate::prof::NORM_NS);
    let (scale, zero) = quantize_scale_zero(low, high)?;
    let _ptq = crate::prof::Timer::start();
    let mut values = vec![0i8; rows * width];
    let mut centered_sums = vec![0i32; rows];
    let mut scratch = vec![0.0f32; width];
    for r in 0..rows {
        norm_row(&x[r * width..(r + 1) * width], means[r], devs[r], gain, &mut scratch);
        centered_sums[r] = quantize_row_codes(
            &scratch,
            width,
            scale,
            zero,
            &mut values[r * width..(r + 1) * width],
        );
    }
    _ptq.stop(&crate::prof::QUANT_NS);
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

/// Runtime i8mm detection for the SMMLA projection kernel. The
/// `vmmlaq_s32` stdarch intrinsic is still unstable, so the kernel
/// uses inline asm; the asm needs the i8mm target feature enabled
/// on the enclosing function (see `project_block_mmla`).
#[cfg(target_arch = "aarch64")]
fn have_i8mm() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::arch::is_aarch64_feature_detected!("i8mm"))
}

/// Single-row projection (decoder GEMV) over the packed SDOT
/// layout: chunk (jb, cg) of `packed_sdot` holds columns
/// 4cg..4cg+4 at depths 4jb..4jb+4 in exactly the lane order
/// `vdotq_s32` accumulates when the activation operand carries
/// the same 4 activations in every lane. Accumulation is exact
/// i32 integer math, so `acc` is bit-identical to the GEMM
/// paths'; the shared epilogue applies unchanged.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn project_gemv_sdot(m: &Matrix, q: &Quantized, acc: &mut [i32]) {
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let packed = m.packed_gemv.as_ptr();
    let values = q.values.as_ptr();
    let outp = acc.as_mut_ptr();
    let groups = n / 4;
    let depth = k / 4;
    unsafe {
        let mut cg0 = 0usize;
        while cg0 + 16 <= groups {
            // Depth-outer over a 16-group tile: the activation
            // operand is built once per depth step and shared by
            // all 16 column groups (16 concurrent sequential
            // weight streams, one per group).
            let mut acc = [vdupq_n_s32(0); 16];
            for jb in 0..depth {
                let a32 = (values.add(jb * 4) as *const i32).read_unaligned();
                let adup = vreinterpretq_s8_s32(vdupq_n_s32(a32));
                for (g, a) in acc.iter_mut().enumerate() {
                    let w = vld1q_s8(packed.add(((cg0 + g) * depth + jb) * 16));
                    *a = vdotq_s32(*a, w, adup);
                }
            }
            for (g, a) in acc.iter().enumerate() {
                vst1q_s32(outp.add((cg0 + g) * 4), *a);
            }
            cg0 += 16;
        }
        while cg0 + 8 <= groups {
            let mut acc = [vdupq_n_s32(0); 8];
            for (g, a) in acc.iter_mut().enumerate() {
                let gbase = packed.add((cg0 + g) * depth * 16);
                for jb in 0..depth {
                    let a32 = (values.add(jb * 4) as *const i32).read_unaligned();
                    let adup = vdupq_n_s32(a32);
                    let w = vld1q_s8(gbase.add(jb * 16));
                    *a = vdotq_s32(*a, w, vreinterpretq_s8_s32(adup));
                }
            }
            for (g, a) in acc.iter().enumerate() {
                vst1q_s32(outp.add((cg0 + g) * 4), *a);
            }
            cg0 += 8;
        }
        while cg0 < groups {
            let mut a4 = vdupq_n_s32(0);
            let gbase = packed.add(cg0 * depth * 16);
            for jb in 0..depth {
                let a32 = (values.add(jb * 4) as *const i32).read_unaligned();
                let adup = vdupq_n_s32(a32);
                let w = vld1q_s8(gbase.add(jb * 16));
                a4 = vdotq_s32(a4, w, vreinterpretq_s8_s32(adup));
            }
            vst1q_s32(outp.add(cg0 * 4), a4);
            cg0 += 1;
        }
    }
}

/// One SMMLA step: `acc` lanes are the 2x2 block
/// [r0c0, r0c1, r1c0, r1c1]; `a` packs two rows' 8-deep k runs
/// ([r0 k0..8][r1 k0..8]) and `b` two columns' ([c0 k0..8][c1 k0..8]).
/// Integer accumulation is exact and grouping-insensitive, so this
/// produces accumulations identical to the SDOT path's.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn smmla(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut d = acc;
    std::arch::asm!(
        "smmla {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) d,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    d
}

/// SMMLA projection kernel over the `pack_mmla` tile layout:
/// 8 rows x 8 columns register tiles, 16 SMMLA (512 MACs) per
/// 8-deep k step. Accumulates into `acc` ([rows][n]) exactly like
/// the other project paths. Rows past `rows` in the final tile are
/// fed zero activations and their outputs are never stored.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn project_block_mmla(
    m: &Matrix,
    q: &Quantized,
    r0: usize,
    rows: usize,
    acc: &mut [i32],
) {
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let packed = m.packed_mmla.as_ptr();
    let qp = q.values.as_ptr();
    let zero8 = vdupq_n_s8(0);
    let mut r = 0usize;
    while r < rows {
        let mut c = 0usize;
        while c < n {
            // Load the current 8x8 accumulator tile.
            let mut tile = [[vdupq_n_s32(0); 4]; 4];
            for (g, trow) in tile.iter_mut().enumerate() {
                let rr = r + 2 * g;
                unsafe {
                    if r0 + rr < r0 + rows {
                        let dst = acc.as_ptr().add(rr * n + c);
                        if rr + 1 < rows {
                            let dst1 = acc.as_ptr().add((rr + 1) * n + c);
                            for p in 0..4 {
                                let lo = vld1_s32(dst.add(2 * p) as *const i32);
                                let hi = vld1_s32(dst1.add(2 * p) as *const i32);
                                trow[p] = vcombine_s32(lo, hi);
                            }
                        } else {
                            for p in 0..4 {
                                let lo = vld1_s32(dst.add(2 * p) as *const i32);
                                trow[p] = vcombine_s32(lo, vdup_n_s32(0));
                            }
                        }
                    }
                }
            }
            for jb in 0..k / 8 {
                // A operands: two rows' 8-byte k runs per row group.
                let mut avec = [zero8; 4];
                for g in 0..4 {
                    let rr = r + 2 * g;
                    unsafe {
                        if rr < rows {
                            let lo = vld1_s8(qp.add((r0 + rr) * k + jb * 8));
                            let hi = if rr + 1 < rows {
                                vld1_s8(qp.add((r0 + rr + 1) * k + jb * 8))
                            } else {
                                vget_low_s8(zero8)
                            };
                            avec[g] = vcombine_s8(lo, hi);
                        }
                    }
                }
                let wbase = packed.add(((c / 8) * (k / 8) + jb) * 64);
                for p in 0..4 {
                    let w = unsafe { vld1q_s8(wbase.add(16 * p)) };
                    for g in 0..4 {
                        tile[g][p] = unsafe { smmla(tile[g][p], avec[g], w) };
                    }
                }
            }
            // Store the tile back (only rows that exist).
            for (g, trow) in tile.iter().enumerate() {
                let rr = r + 2 * g;
                if rr >= rows {
                    break;
                }
                unsafe {
                    let dst = acc.as_mut_ptr().add(rr * n + c);
                    for p in 0..4 {
                        vst1_s32(dst.add(2 * p), vget_low_s32(trow[p]));
                    }
                    if rr + 1 < rows {
                        let dst1 = acc.as_mut_ptr().add((rr + 1) * n + c);
                        for p in 0..4 {
                            vst1_s32(dst1.add(2 * p), vget_high_s32(trow[p]));
                        }
                    }
                }
            }
            c += 8;
        }
        r += 8;
    }
}

/// Permute row-major quantized activations into the tiled layout
/// the SMMLA kernel consumes (the C++ engine's
/// `activation_offset`): the A operand for row group `g` of an
/// 8-row block at k block `jb` is 16 contiguous bytes at
/// `(r/8)*k*8 + jb*64 + g*16`. Rows are padded to a multiple of 8
/// with zeros (their outputs are never stored). Values are copied
/// verbatim — the accumulation products are unchanged.
#[cfg(target_arch = "aarch64")]
fn tile_activations(q: &Quantized, rows: usize, k: usize, out: &mut Vec<i8>) {
    let padded = rows.div_ceil(8) * 8;
    out.clear();
    out.resize(padded * k, 0);
    for rb in 0..padded / 8 {
        for jb in 0..k / 8 {
            for g in 0..4 {
                let dst = rb * k * 8 + jb * 64 + g * 16;
                for half in 0..2 {
                    let r = rb * 8 + g * 2 + half;
                    if r < rows {
                        let src = r * k + jb * 8;
                        out[dst + half * 8..dst + half * 8 + 8]
                            .copy_from_slice(&q.values[src..src + 8]);
                    }
                }
            }
        }
    }
}

/// SMMLA projection kernel over tiled activations (see
/// `tile_activations`): one 16-byte load per A operand. Handles
/// any row count (the tiled buffer is zero-padded to 8 rows).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn project_block_mmla_tiled(
    m: &Matrix,
    tiled: &[i8],
    rows: usize,
    acc: &mut [i32],
) {
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let packed = m.packed_mmla.as_ptr();
    let tp = tiled.as_ptr();
    let mut r = 0usize;
    while r < rows {
        let mut c = 0usize;
        while c < n {
            let mut tile = [[vdupq_n_s32(0); 4]; 4];
            for (g, trow) in tile.iter_mut().enumerate() {
                let rr = r + 2 * g;
                if rr < rows {
                    unsafe {
                        let dst = acc.as_ptr().add(rr * n + c);
                        for p in 0..4 {
                            let lo = vld1_s32(dst.add(2 * p) as *const i32);
                            let hi = if rr + 1 < rows {
                                vld1_s32(
                                    acc.as_ptr().add((rr + 1) * n + c + 2 * p) as *const i32
                                )
                            } else {
                                vdup_n_s32(0)
                            };
                            trow[p] = vcombine_s32(lo, hi);
                        }
                    }
                }
            }
            for jb in 0..k / 8 {
                let abase = tp.add((r / 8) * k * 8 + jb * 64);
                let wbase = packed.add(((c / 8) * (k / 8) + jb) * 64);
                let mut avec = [vdupq_n_s8(0); 4];
                for (g, a) in avec.iter_mut().enumerate() {
                    *a = unsafe { vld1q_s8(abase.add(16 * g)) };
                }
                for p in 0..4 {
                    let w = unsafe { vld1q_s8(wbase.add(16 * p)) };
                    for g in 0..4 {
                        tile[g][p] = unsafe { smmla(tile[g][p], avec[g], w) };
                    }
                }
            }
            for (g, trow) in tile.iter().enumerate() {
                let rr = r + 2 * g;
                if rr >= rows {
                    break;
                }
                unsafe {
                    let dst = acc.as_mut_ptr().add(rr * n + c);
                    for p in 0..4 {
                        vst1_s32(dst.add(2 * p), vget_low_s32(trow[p]));
                    }
                    if rr + 1 < rows {
                        let dst1 = acc.as_mut_ptr().add((rr + 1) * n + c);
                        for p in 0..4 {
                            vst1_s32(dst1.add(2 * p), vget_high_s32(trow[p]));
                        }
                    }
                }
            }
            c += 8;
        }
        r += 8;
    }
}

/// SMMLA projection with the epilogue fused in: accumulators go
/// straight from registers through the exact i32 correction and
/// the separate f32 multiply/add of `epilogue` into the output,
/// so the i32 accumulator buffer never round-trips through
/// memory. Bit-identical to accumulate-then-epilogue.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn project_block_mmla_fused(
    m: &Matrix,
    q: &Quantized,
    tiled: &[i8],
    rows: usize,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (k, n) = (m.k, m.n);
    let packed = m.packed_mmla.as_ptr();
    let tp = tiled.as_ptr();
    let coef_a = 128 - q.zero;
    unsafe {
        let sums_p = m.sums.as_ptr();
        let zeros_p = m.zeros.as_ptr();
        let scales_p = m.scales.as_ptr();
        let bias_p = m.bias.as_ptr();
        let qsv = vdupq_n_f32(q.scale);
        let mut r = 0usize;
        while r < rows {
            let mut c = 0usize;
            while c < n {
                let mut tile = [[vdupq_n_s32(0); 4]; 4];
                for jb in 0..k / 8 {
                    let abase = tp.add((r / 8) * k * 8 + jb * 64);
                    let wbase = packed.add(((c / 8) * (k / 8) + jb) * 64);
                    let mut avec = [vdupq_n_s8(0); 4];
                    for (g, a) in avec.iter_mut().enumerate() {
                        *a = vld1q_s8(abase.add(16 * g));
                    }
                    for p in 0..4 {
                        let w = vld1q_s8(wbase.add(16 * p));
                        for g in 0..4 {
                            tile[g][p] = smmla(tile[g][p], avec[g], w);
                        }
                    }
                }
                // Fused epilogue for this 8x8 tile.
                for (g, trow) in tile.iter().enumerate() {
                    let rr = r + 2 * g;
                    if rr >= rows {
                        break;
                    }
                    let cs0 = q.centered_sums[rr];
                    let cs1 = if rr + 1 < rows {
                        q.centered_sums[rr + 1]
                    } else {
                        0
                    };
                    let cs2 = vcombine_s32(vdup_n_s32(cs0), vdup_n_s32(cs1));
                    for p in 0..4 {
                        let col = c + 2 * p;
                        let sums2 = vcombine_s32(
                            vld1_s32(sums_p.add(col) as *const i32),
                            vld1_s32(sums_p.add(col) as *const i32),
                        );
                        let zeros2 = vcombine_s32(
                            vld1_s32(zeros_p.add(col) as *const i32),
                            vld1_s32(zeros_p.add(col) as *const i32),
                        );
                        let total = vsubq_s32(
                            vaddq_s32(trow[p], vmulq_n_s32(sums2, coef_a)),
                            vmulq_s32(zeros2, cs2),
                        );
                        let scales2 = vcombine_f32(
                            vld1_f32(scales_p.add(col)),
                            vld1_f32(scales_p.add(col)),
                        );
                        let bias2 = vcombine_f32(
                            vld1_f32(bias_p.add(col)),
                            vld1_f32(bias_p.add(col)),
                        );
                        let res = vaddq_f32(
                            vmulq_f32(vcvtq_f32_s32(total), vmulq_f32(scales2, qsv)),
                            bias2,
                        );
                        let dst = out.as_mut_ptr().add(rr * n + col);
                        vst1_f32(dst, vget_low_f32(res));
                        if rr + 1 < rows {
                            let dst1 = out.as_mut_ptr().add((rr + 1) * n + col);
                            vst1_f32(dst1, vget_high_f32(res));
                        }
                    }
                }
                c += 8;
            }
            r += 8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    static TILE_BUF: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
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


/// Signed 4-bit code at flat operand position `pos` of the packed
/// int4 stream (see Int4Matrix for the layout contract).
fn int4_code_at(m4: &crate::model::Int4Matrix, pos: usize) -> i32 {
    let byte = m4.packed[pos / 2];
    let u = if pos % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    u as i32 - 8
}

fn int4_group_sums(q: &Quantized, group: usize, ngroups: usize) -> Vec<i32> {
    let mut av = vec![0i32; ngroups];
    for (gi, s) in av.iter_mut().enumerate() {
        *s = q.values[gi * group..(gi + 1) * group]
            .iter()
            .map(|&v| v as i32)
            .sum();
    }
    av
}

/// Definitional int4 GEMV (host/reference): out = q.scale *
/// (sum_j a[j] * W4[j][c]) + bias, with a[j] = values[j] + 128 -
/// zero carrying the epilogue's offset itself (no separate
/// colsum term -- the group-fold form adds it exactly once).
fn project_gemv_int4_scalar(m4: &crate::model::Int4Matrix, q: &Quantized, out: &mut [f32]) {
    let (k, n) = (m4.k, m4.n);
    for c in 0..n {
        let mut acc = 0.0f32;
        for j in 0..k {
            let a = (q.values[j] as i32 + 128 - q.zero) as f32;
            acc += a * m4.weight_at(j, c);
        }
        out[c] = q.scale * acc + m4.bias[c];
    }
}

/// Group-fold int4 GEMV in scalar code (the NEON kernel's exact
/// algebra; host tests pin the two together): per group, SDOT over
/// biased codes u = q4 + 8, corrected by 8 * (activation group
/// sum), folded with the group's per-column scale.
fn project_gemv_int4_fold(m4: &crate::model::Int4Matrix, q: &Quantized, out: &mut [f32]) {
    let (k, n, g) = (m4.k, m4.n, m4.group);
    let ngroups = k / g;
    let av = int4_group_sums(q, g, ngroups);
    for c in 0..n {
        let cg = c / 4;
        let dc = c % 4;
        let mut accf = 0.0f32;
        for gi in 0..ngroups {
            let mut pacc = 0i32;
            for j in gi * g..(gi + 1) * g {
                let pos = (cg * (k / 4) + j / 4) * 16 + dc * 4 + j % 4;
                pacc += q.values[j] as i32 * (int4_code_at(m4, pos) + 8);
            }
            accf += m4.gscales[gi * n + c] * (pacc - 8 * av[gi]) as f32;
        }
        out[c] = q.scale * (accf + (128 - q.zero) as f32 * m4.colsums[c]) + m4.bias[c];
    }
}

#[cfg(target_arch = "aarch64")]
fn project_gemv_int4(m4: &crate::model::Int4Matrix, q: &Quantized, out: &mut [f32]) {
    if (m4.n / 4) % 4 == 0 {
        let av = int4_group_sums(q, m4.group, m4.k / m4.group);
        unsafe { project_gemv_int4_neon(m4, q, &av, out) }
    } else {
        project_gemv_int4_fold(m4, q, out)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn project_gemv_int4(m4: &crate::model::Int4Matrix, q: &Quantized, out: &mut [f32]) {
    project_gemv_int4_fold(m4, q, out)
}

/// aarch64 int4 GEMV: mirrors project_gemv_sdot's tiling (16
/// column groups per tile, depth streamed per column group), with
/// the weight operand unpacked from nibbles per 16-value block
/// and a per-group f32 fold.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn project_gemv_int4_neon(
    m4: &crate::model::Int4Matrix,
    q: &Quantized,
    av: &[i32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let (k, n, g) = (m4.k, m4.n, m4.group);
    let ngroups = k / g;
    let blocks_per_group = g / 4;
    let depth_blocks = k / 4;
    let mask = vdupq_n_u8(0x0F);
    let sa = vdupq_n_f32(q.scale);
    let zterm = vdupq_n_f32((128 - q.zero) as f32);
    // Four column groups in flight: four independent SDOT chains
    // (the single-chain form is latency-bound), sharing each
    // depth block's activation word. Requires (n/4) % 4 == 0,
    // enforced at twin-build time.
    let mut cg0 = 0usize;
    while cg0 < n / 4 {
        let mut accf = [vdupq_n_f32(0.0); 4];
        for gi in 0..ngroups {
            let mut acc = [vdupq_n_s32(0); 4];
            // Two depth blocks per step: one 16-byte load holds
            // both blocks' nibbles; zip1/zip2 split them.
            for jb2 in (gi * blocks_per_group..(gi + 1) * blocks_per_group).step_by(2) {
                let a0 = std::ptr::read_unaligned(
                    q.values.as_ptr().add(jb2 * 4) as *const u32
                );
                let a1 = std::ptr::read_unaligned(
                    q.values.as_ptr().add((jb2 + 1) * 4) as *const u32
                );
                let av0 = vreinterpretq_s8_s32(vdupq_n_s32(a0 as i32));
                let av1 = vreinterpretq_s8_s32(vdupq_n_s32(a1 as i32));
                for (q4i, acc_i) in acc.iter_mut().enumerate() {
                    let cg = cg0 + q4i;
                    let wp = m4.packed.as_ptr().add((cg * depth_blocks + jb2) * 8);
                    let wb = vld1q_u8(wp);
                    let lo = vandq_u8(wb, mask);
                    let hi = vshrq_n_u8::<4>(wb);
                    // SDOT over the BIASED codes u = q4 + 8
                    // (0..=15, exact in i8); the group fold
                    // subtracts 8 * (activation group sum).
                    let w0 = vreinterpretq_s8_u8(vzip1q_u8(lo, hi));
                    let w1 = vreinterpretq_s8_u8(vzip2q_u8(lo, hi));
                    *acc_i = vdotq_s32(*acc_i, w0, av0);
                    *acc_i = vdotq_s32(*acc_i, w1, av1);
                }
            }
            let corr = vdupq_n_f32((8 * av[gi]) as f32);
            for (q4i, accf_i) in accf.iter_mut().enumerate() {
                let cg = cg0 + q4i;
                let sg = vld1q_f32(m4.gscales.as_ptr().add(gi * n + cg * 4));
                let tf = vsubq_f32(vcvtq_f32_s32(acc[q4i]), corr);
                *accf_i = vfmaq_f32(*accf_i, tf, sg);
            }
        }
        for (q4i, accf_i) in accf.iter().enumerate() {
            let cg = cg0 + q4i;
            let sw = vld1q_f32(m4.colsums.as_ptr().add(cg * 4));
            let bv = vld1q_f32(m4.bias.as_ptr().add(cg * 4));
            let tot = vfmaq_f32(*accf_i, zterm, sw);
            let res = vfmaq_f32(bv, tot, sa);
            vst1q_f32(out.as_mut_ptr().add(cg * 4), res);
        }
        cg0 += 4;
    }
}

/// Accumulate `rows` projection rows (starting at row `r0` of `q`)
/// into `acc` ([rows][n]). The aarch64 path uses the SDOT kernels
/// when the packed layout is available.
fn project_range(m: &Matrix, q: &Quantized, r0: usize, rows: usize, acc: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if r0 == 0
            && rows == 1
            && m.packed_gemv.len() == m.k * m.n
            && m.k % 4 == 0
            && m.n % 4 == 0
            && have_dotprod()
        {
            unsafe {
                project_gemv_sdot(m, q, acc);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if m.packed_mmla.len() == m.k * m.n && m.k % 8 == 0 && m.n % 8 == 0 && have_i8mm() {
            if rows >= 8 {
                // Tiled-activation SMMLA path (any row count).
                TILE_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    tile_activations(q, rows, m.k, &mut buf);
                    // SAFETY: the tiled buffer slice is reborrowed
                    // raw so the borrow ends before the kernel runs;
                    // the buffer is only mutated by this call site.
                    let tiled: &[i8] = unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };
                    unsafe {
                        project_block_mmla_tiled(m, tiled, rows, acc);
                    }
                });
                return;
            }
            // Full 8-row blocks through the SMMLA kernel; the tail
            // (< 8 rows) falls through to the per-row paths below.
            let full = rows / 8 * 8;
            if full > 0 {
                unsafe {
                    project_block_mmla(m, q, r0, full, acc);
                }
            }
            if full < rows {
                project_range_tail(m, q, r0 + full, rows - full, &mut acc[full * m.n..]);
            }
            return;
        }
    }
    project_range_tail(m, q, r0, rows, acc);
}

/// Per-row projection accumulation (SDOT kernel when available,
/// else the NEON/scalar row path). Also serves as the < 8-row tail
/// of the SMMLA block path.
fn project_range_tail(m: &Matrix, q: &Quantized, r0: usize, rows: usize, acc: &mut [i32]) {
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
    project_s(m, q, rows, 0)
}

/// `project` with a profiling site tag (0 = encoder/adapter,
/// 1 = decoder, 2 = cross).
fn proj_bucket(t0: std::time::Instant, rows: usize, macs: u64, site: u8) {
    if !crate::prof::enabled() {
        return;
    }
    let ns = t0.elapsed().as_nanos() as u64;
    match site {
        1 => crate::prof::add(&crate::prof::PROJ_S1_NS, ns),
        2 => crate::prof::add(&crate::prof::PROJ_S2_NS, ns),
        _ => crate::prof::add(&crate::prof::PROJ_S0_NS, ns),
    }
    if rows == 1 {
        crate::prof::add(&crate::prof::PROJ_R1_NS, ns);
        crate::prof::add(&crate::prof::PROJ_R1_MACS, macs);
    } else if rows >= 8 {
        crate::prof::add(&crate::prof::PROJ_BIG_NS, ns);
        crate::prof::add(&crate::prof::PROJ_BIG_MACS, macs);
    }
}

pub fn project_s(m: &Matrix, q: &Quantized, rows: usize, site: u8) -> Vec<f32> {
    let (k, n) = (m.k, m.n);
    let _bucket_t0 = std::time::Instant::now();
    crate::prof::add(&crate::prof::PROJ_CALLS, 1);
    let _macs = (rows as u64) * (k as u64) * (n as u64);
    match site {
        1 => crate::prof::add(&crate::prof::PROJ_MACS_DEC, _macs),
        2 => crate::prof::add(&crate::prof::PROJ_MACS_X, _macs),
        _ => crate::prof::add(&crate::prof::PROJ_MACS_ENC, _macs),
    }
    // Round 4: int4 GEMV twin (rows == 1, decoder matrices only).
    if rows == 1 && crate::model::dec_int4_enabled() {
        if let Some(m4) = &m.int4 {
            let mut out = crate::scratch::alloc_uninit_f32(n);
            let _pt4 = crate::prof::Timer::start();
            project_gemv_int4(m4, q, &mut out);
            _pt4.stop(&crate::prof::PROJ_ACC_NS);
            proj_bucket(_bucket_t0, rows, _macs, site);
            return out;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if rows >= 2 && m.packed_mmla.len() == m.k * m.n && m.k % 8 == 0 && m.n % 8 == 0 && have_i8mm()
        {
            // Fused SMMLA path: no accumulator round-trip.
            let mut out = crate::scratch::alloc_uninit_f32(rows * n);
            let _ptf = crate::prof::Timer::start();
            TILE_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                tile_activations(q, rows, m.k, &mut buf);
                let tiled: &[i8] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };
                unsafe {
                    project_block_mmla_fused(m, q, tiled, rows, &mut out);
                }
            });
            _ptf.stop(&crate::prof::PROJ_ACC_NS);
            proj_bucket(_bucket_t0, rows, (rows as u64) * (k as u64) * (n as u64), site);
            return out;
        }
    }
    let mut acc = crate::scratch::take_i32(rows * n);
    acc.clear();
    acc.resize(rows * n, 0);
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
    let _pt = crate::prof::Timer::start();
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
    _pt.stop(&crate::prof::PROJ_ACC_NS);
    let _pte = crate::prof::Timer::start();
    let mut out = crate::scratch::alloc_uninit_f32(rows * n);
    epilogue(m, q, &acc, rows, &mut out);
    _pte.stop(&crate::prof::PROJ_EPI_NS);
    crate::scratch::put_i32(acc);
    proj_bucket(_bucket_t0, rows, (rows as u64) * (k as u64) * (n as u64), site);
    out
}

/// The projection epilogue: per output element,
/// `total = acc + (128 - zero) * sums[c] - centered_sums[r] *
/// zeros[c]` in exact i32, then `total as f32 * (scale *
/// scales[c]) + bias[c]`. The NEON form computes the identical
/// operations per element (integer mul/add wrap identically;
/// the f32 part is a separate multiply and add, never an FMA),
/// so results are bit-identical to the scalar form.
fn epilogue(m: &Matrix, q: &Quantized, acc: &[i32], rows: usize, out: &mut [f32]) {
    let n = m.n;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let coef_a = 128 - q.zero;
        unsafe {
            let sums_p = m.sums.as_ptr();
            let zeros_p = m.zeros.as_ptr();
            let scales_p = m.scales.as_ptr();
            let bias_p = m.bias.as_ptr();
            for r in 0..rows {
                let coef_b = q.centered_sums[r];
                let qs = vdupq_n_f32(q.scale);
                let acc_p = acc.as_ptr().add(r * n);
                let out_p = out.as_mut_ptr().add(r * n);
                let mut c = 0usize;
                while c + 4 <= n {
                    let total = vaddq_s32(
                        vld1q_s32(acc_p.add(c) as *const i32),
                        vmulq_n_s32(vld1q_s32(sums_p.add(c) as *const i32), coef_a),
                    );
                    let total = vsubq_s32(
                        total,
                        vmulq_n_s32(vld1q_s32(zeros_p.add(c) as *const i32), coef_b),
                    );
                    let scaled = vmulq_f32(
                        vcvtq_f32_s32(total),
                        vmulq_f32(vld1q_f32(scales_p.add(c)), qs),
                    );
                    let res = vaddq_f32(scaled, vld1q_f32(bias_p.add(c)));
                    vst1q_f32(out_p.add(c), res);
                    c += 4;
                }
                while c < n {
                    let total = acc[r * n + c] + coef_a * m.sums[c] - coef_b * m.zeros[c];
                    out[r * n + c] = total as f32 * (q.scale * m.scales[c]) + m.bias[c];
                    c += 1;
                }
            }
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for r in 0..rows {
            for c in 0..n {
                let total =
                    acc[r * n + c] + (128 - q.zero) * m.sums[c] - q.centered_sums[r] * m.zeros[c];
                out[r * n + c] = total as f32 * (q.scale * m.scales[c]) + m.bias[c];
            }
        }
    }
}

/// Block min/max with the nonfinite check, for `quantize`.
/// The NEON form: vector min/max over finite inputs is exact
/// (same result as the scalar scan; the +/-0.0 distinction it
/// can differ in cannot change the derived scale or zero point),
/// and any lane with |v| > f32::MAX (i.e. inf or NaN, since NaN
/// comparisons are false) fails the finite check exactly as the
/// scalar scan does.
fn minmax_finite(x: &[f32]) -> Result<(f32, f32), String> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        unsafe {
            let mut low4 = vdupq_n_f32(0.0);
            let mut high4 = vdupq_n_f32(0.0);
            let mut bad = vdupq_n_u32(0);
            let maxv = vdupq_n_f32(f32::MAX);
            let mut i = 0usize;
            while i + 4 <= x.len() {
                let v = vld1q_f32(x.as_ptr().add(i));
                low4 = vminq_f32(low4, v);
                high4 = vmaxq_f32(high4, v);
                bad = vorrq_u32(bad, vmvnq_u32(vcleq_f32(vabsq_f32(v), maxv)));
                i += 4;
            }
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), low4);
            let mut low = 0.0f32;
            for &v in &lanes {
                if v < low {
                    low = v;
                }
            }
            vst1q_f32(lanes.as_mut_ptr(), high4);
            let mut high = 0.0f32;
            for &v in &lanes {
                if v > high {
                    high = v;
                }
            }
            let mut bad_lanes = [0u32; 4];
            vst1q_u32(bad_lanes.as_mut_ptr(), bad);
            let mut bad_any = bad_lanes.iter().any(|&b| b != 0);
            while i < x.len() {
                let v = x[i];
                if !v.is_finite() {
                    bad_any = true;
                }
                if v < low {
                    low = v;
                }
                if v > high {
                    high = v;
                }
                i += 1;
            }
            if bad_any {
                return Err("nonfinite activation".into());
            }
            return Ok((low, high));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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
        Ok((low, high))
    }
}

/// Convenience for Matrix::run(x, ...): quantize the float input, project.
pub fn run(m: &Matrix, x: &[f32], rows: usize, k: usize) -> Result<Vec<f32>, String> {
    run_s(m, x, rows, k, 0)
}

/// `run` with a profiling site tag (see `project_s`).
pub fn run_s(m: &Matrix, x: &[f32], rows: usize, k: usize, site: u8) -> Result<Vec<f32>, String> {
    let q = quantize(x, rows, k)?;
    Ok(project_s(m, &q, rows, site))
}

/// `run` on a normalize()d input, fused (see quantize_normalized).
pub fn run_normalized(
    m: &Matrix,
    x: &[f32],
    rows: usize,
    width: usize,
    gain: &[f32],
) -> Result<Vec<f32>, String> {
    let q = quantize_normalized(x, rows, width, gain)?;
    Ok(project_s(m, &q, rows, 0))
}

/// medium.cpp `normalize` (x86 path): per-row ascending sum and sum of
/// squares, variance = squares/width - mean^2, deviation =
/// sqrt(variance + 1e-5), out = ((x - mean) / deviation) * gain.
pub fn normalize(x: &[f32], rows: usize, width: usize, gain: &[f32]) -> Vec<f32> {
    let _pt = crate::prof::Timer::start();
    let mut out = crate::scratch::alloc_uninit_f32(x.len());
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
    _pt.stop(&crate::prof::NORM_NS);
    out
}

/// Write `x` ([rows][640]) into `out` in head-major [10][rows][64]
/// order — the exact permutation of `heads` without the temp buffer.
pub fn scatter_heads(x: &[f32], rows: usize, out: &mut [f32]) {
    for h in 0..10 {
        for r in 0..rows {
            let src = &x[r * 640 + h * 64..r * 640 + h * 64 + 64];
            let dst = &mut out[(h * rows + r) * 64..(h * rows + r) * 64 + 64];
            dst.copy_from_slice(src);
        }
    }
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
/// The valid column band of row `r` under the encoder masks
/// (2: [r-16, r+4]; 3: [r-16, r]; anything else: full row).
fn band_range(r: usize, memory: usize, mask: i32) -> (usize, usize) {
    match mask {
        2 => (
            r.saturating_sub(16),
            (r + 4).min(memory.saturating_sub(1)),
        ),
        3 => (r.saturating_sub(16), r.min(memory.saturating_sub(1))),
        _ => (0, memory.saturating_sub(1)),
    }
}

/// Softmax over one row's band, reproducing `softmax_row_neon`'s
/// exact accumulation structure on the full row: the row max is
/// the band max (out-of-band entries are -inf), each in-band
/// element's exp is evaluated by the same variant its absolute
/// index would get (vector form below `n - n % 4`, scalar form in
/// the tail), lane partial sums accumulate by absolute index mod
/// 4, and the tail accumulates ascending, combined as
/// ((a0 + a1) + (a2 + a3)) + tail. The skipped out-of-band terms
/// contribute exp(-inf) ~= 7e-46 each, which cannot change any
/// f32 partial sum that also receives an in-band term, nor the
/// final sum's rounding -- so results are bit-identical to the
/// full-row computation for this network's value ranges.
#[cfg(target_arch = "aarch64")]
fn softmax_banded(row: &mut [f32], lo: usize, n: usize, max: f32) {
    use std::arch::aarch64::*;
    let main_end = n - n % 4;
    let mut acc4 = [0.0f32; 4];
    let mut tail_acc = 0.0f32;
    for (idx, item) in row.iter_mut().enumerate() {
        let i = lo + idx;
        let e = if i < main_end {
            // sumexp_vec is lane-wise; a splat evaluates the lane
            // function for this element exactly.
            unsafe {
                let v = vdupq_n_f32(*item - max);
                let e = kernels::sumexp_vec_pub(v);
                let mut tmp = [0.0f32; 4];
                vst1q_f32(tmp.as_mut_ptr(), e);
                tmp[0]
            }
        } else {
            kernels::sumexp_exp(*item - max)
        };
        *item = e;
        if i < main_end {
            acc4[i % 4] += e;
        } else {
            tail_acc += e;
        }
    }
    let sum = ((acc4[0] + acc4[1]) + (acc4[2] + acc4[3])) + tail_acc;
    let scale = 1.0f32 / sum;
    for item in row.iter_mut() {
        *item *= scale;
    }
}

/// Banded attention output accumulation: out[j] =
/// sum over the band (ascending) of score * v[c][j], one FMA
/// chain per element from 0.0 -- the same per-element order as
/// the full scores GEMM (whose out-of-band terms are the ~1e-45
/// softmax outputs, absorbed without effect).
#[cfg(target_arch = "aarch64")]
/// `av_banded` with an explicit value-row stride (for the
/// row-major layout, where one head's rows sit 640 apart).
#[cfg(target_arch = "aarch64")]
fn av_banded_strided(scores: &[f32], v: &[f32], stride: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        for (i, &s) in scores.iter().enumerate() {
            let sv = vdupq_n_f32(s);
            let base = v.as_ptr().add(i * stride);
            for (j, a) in acc.iter_mut().enumerate() {
                let vv = vld1q_f32(base.add(4 * j));
                *a = vfmaq_f32(*a, sv, vv);
            }
        }
        for (j, a) in acc.iter().enumerate() {
            vst1q_f32(out.as_mut_ptr().add(4 * j), *a);
        }
    }
}

/// Banded encoder attention directly over the row-major
/// [rows][640] projection outputs: head h's row for position p
/// is at p * 640 + h * 64, so the heads()/unheads() layout
/// shuffles (and their buffers) are skipped entirely. The
/// arithmetic per output element is identical to the head-major
/// banded path (same dot_asc scores, same softmax_banded, same
/// rank-1 AV accumulation order).
#[cfg(target_arch = "aarch64")]
fn attention_banded_rowmajor(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    mask: i32,
) -> Vec<f32> {
    let _pt = crate::prof::Timer::start();
    let mut pairs: u64 = 0;
    let mut out = crate::scratch::alloc_uninit_f32(rows * 640);
    let mut band = vec![0.0f32; 21];
    for h in 0..10 {
        for r in 0..rows {
            let (lo, hi) = band_range(r, rows, mask);
            let w = hi - lo + 1;
            pairs += w as u64;
            let srow = &mut band[..w];
            let mut max = f32::NEG_INFINITY;
            for (i, c) in (lo..=hi).enumerate() {
                let total = kernels::dot_asc(
                    &queries[r * 640 + h * 64..r * 640 + h * 64 + 64],
                    &keys[c * 640 + h * 64..c * 640 + h * 64 + 64],
                );
                let s = ATTN_SCALE * total;
                srow[i] = s;
                if s > max {
                    max = s;
                }
            }
            softmax_banded(srow, lo, rows, max);
            av_banded_strided(
                srow,
                &values[lo * 640 + h * 64..],
                640,
                &mut out[r * 640 + h * 64..r * 640 + h * 64 + 64],
            );
        }
    }
    crate::prof::add(&crate::prof::ATTN_ENC_PAIRS, 10 * pairs);
    _pt.stop(&crate::prof::ATTN_ENC_NS);
    out
}

#[cfg(target_arch = "aarch64")]
fn av_banded(scores: &[f32], v: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        for (i, &s) in scores.iter().enumerate() {
            let sv = vdupq_n_f32(s);
            let base = v.as_ptr().add(i * 64);
            for (j, a) in acc.iter_mut().enumerate() {
                let vv = vld1q_f32(base.add(4 * j));
                *a = vfmaq_f32(*a, sv, vv);
            }
        }
        for (j, a) in acc.iter().enumerate() {
            vst1q_f32(out.as_mut_ptr().add(4 * j), *a);
        }
    }
}

fn attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    rows: usize,
    memory: usize,
    mask: i32,
) -> Vec<f32> {
    let _pt = crate::prof::Timer::start();
    #[cfg(target_arch = "aarch64")]
    {
        if mask == 2 || mask == 3 {
            let mut pairs: u64 = 0;
            let mut out = crate::scratch::alloc_uninit_f32(rows * 640);
            let mut band = vec![0.0f32; 21];
            for h in 0..10 {
                let q = &queries[h * rows * 64..(h + 1) * rows * 64];
                let k = &keys[h * memory * 64..(h + 1) * memory * 64];
                let v = &values[h * memory * 64..(h + 1) * memory * 64];
                for r in 0..rows {
                    let (lo, hi) = band_range(r, memory, mask);
                    let w = hi - lo + 1;
                    pairs += w as u64;
                    let srow = &mut band[..w];
                    let mut max = f32::NEG_INFINITY;
                    for (i, c) in (lo..=hi).enumerate() {
                        let total = kernels::dot_asc(
                            &q[r * 64..(r + 1) * 64],
                            &k[c * 64..(c + 1) * 64],
                        );
                        let s = ATTN_SCALE * total;
                        srow[i] = s;
                        if s > max {
                            max = s;
                        }
                    }
                    softmax_banded(srow, lo, memory, max);
                    av_banded(
                        srow,
                        &v[lo * 64..(hi + 1) * 64],
                        &mut out[(h * rows + r) * 64..(h * rows + r + 1) * 64],
                    );
                }
            }
            crate::prof::add(&crate::prof::ATTN_ENC_PAIRS, 10 * pairs);
            _pt.stop(&crate::prof::ATTN_ENC_NS);
            return out;
        }
    }
    crate::prof::add(&crate::prof::ATTN_ENC_PAIRS, (10 * rows * memory) as u64);
    let mut out = crate::scratch::alloc_uninit_f32(rows * 640);
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
    _pt.stop(&crate::prof::ATTN_ENC_NS);
    out
}

/// medium.cpp `Medium::encoder`.
pub fn encoder(model: &Model, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    let _pt = crate::prof::Timer::start();
    crate::prof::add(&crate::prof::ENC_CALLS, 1);
    crate::prof::add(&crate::prof::ENC_ROWS, rows as u64);
    let mut hidden = input.to_vec();
    for layer in 0..14 {
        let q = quantize_normalized(&hidden, rows, 768, &model.encoder_norms[layer * 2])?;
        let qs = project(&model.encoder_matrices[layer * 6], &q, rows);
        let ks = project(&model.encoder_matrices[layer * 6 + 1], &q, rows);
        let vs = project(&model.encoder_matrices[layer * 6 + 2], &q, rows);
        let mask = if (2..=11).contains(&layer) { 3 } else { 2 };
        #[cfg(target_arch = "aarch64")]
        let joined = attention_banded_rowmajor(&qs, &ks, &vs, rows, mask);
        #[cfg(not(target_arch = "aarch64"))]
        let joined = {
            let attended = attention(&heads(&qs, rows), &heads(&ks, rows), &heads(&vs, rows), rows, rows, mask);
            unheads(&attended, rows)
        };
        let projected = run(&model.encoder_matrices[layer * 6 + 3], &joined, rows, 640)?;
        let _tere = crate::prof::Timer::start();
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        _tere.stop(&crate::prof::ENC_RESID_NS);
        let mut expanded = run_normalized(
            &model.encoder_matrices[layer * 6 + 4],
            &hidden,
            rows,
            768,
            &model.encoder_norms[layer * 2 + 1],
        )?;
        let _tege = crate::prof::Timer::start();
        kernels::gelu_erf_inplace(&mut expanded);
        _tege.stop(&crate::prof::ENC_GELU_NS);
        let projected = run(&model.encoder_matrices[layer * 6 + 5], &expanded, rows, 3072)?;
        let _tere = crate::prof::Timer::start();
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        _tere.stop(&crate::prof::ENC_RESID_NS);
    }
    _pt.stop(&crate::prof::ENC_NS);
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
        let k = project_s(&model.cross_matrices[layer * 2], &q, rows, 2);
        let v = project_s(&model.cross_matrices[layer * 2 + 1], &q, rows, 2);
        // Head-major scatter written directly into the cache
        // (values identical to heads()+copy; one pass, no temp).
        scatter_heads(&k, rows, &mut keys[layer * rows * 640..(layer + 1) * rows * 640]);
        scatter_heads(&v, rows, &mut values[layer * rows * 640..(layer + 1) * rows * 640]);
    }
    Ok((keys, values))
}


#[cfg(test)]
mod int4_tests {
    use super::*;
    use crate::model::{Int4Matrix, Matrix, WeightStore};

    fn test_matrix(k: usize, n: usize, seed: u32) -> Matrix {
        let mut state = seed;
        let mut rnd = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as i32
        };
        let weights: Vec<i8> = (0..k * n).map(|_| (rnd() % 255 - 127) as i8).collect();
        let sums: Vec<i32> = (0..n)
            .map(|c| (0..k).map(|j| weights[j * n + c] as i32).sum())
            .collect();
        let scales: Vec<f32> = (0..n).map(|c| 0.001 + 0.0001 * (c % 7) as f32).collect();
        let zeros: Vec<i32> = (0..n).map(|c| (c % 5) as i32 - 2).collect();
        let bias: Vec<f32> = (0..n).map(|c| 0.01 * (c as f32 - 2.0)).collect();
        Matrix {
            k,
            n,
            weights: WeightStore::Owned(weights),
            packed_sdot: WeightStore::Owned(Vec::new()),
            packed_mmla: WeightStore::Owned(Vec::new()),
            packed_gemv: WeightStore::Owned(Vec::new()),
            sums,
            scales,
            zeros,
            bias,
            int4: None,
        }
    }

    #[test]
    fn int4_fold_matches_definition() {
        let m = test_matrix(128, 8, 42);
        let m4 = Int4Matrix::from_matrix(&m);
        let x: Vec<f32> = (0..128).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.05).collect();
        let q = quantize(&x, 1, 128).unwrap();
        let mut a = vec![0.0f32; 8];
        let mut b = vec![0.0f32; 8];
        project_gemv_int4_scalar(&m4, &q, &mut a);
        project_gemv_int4_fold(&m4, &q, &mut b);
        for c in 0..8 {
            assert!(
                (a[c] - b[c]).abs() <= 3e-3 * (1.0 + a[c].abs()),
                "c={c} def={} fold={}",
                a[c],
                b[c]
            );
        }
    }

    #[test]
    fn int4_dispatch_matches_fold() {
        // On aarch64 the dispatch runs the NEON kernel; elsewhere
        // it IS the fold. Compares against the scalar fold algebra.
        for &(k, n) in &[(64usize, 4usize), (128, 4), (128, 8), (640, 4), (64, 640), (2560, 640), (640, 5120), (640, 640)] {
            let m = test_matrix(k, n, 7);
            let m4 = Int4Matrix::from_matrix(&m);
            let x: Vec<f32> = (0..k).map(|i| ((i * 53 % 211) as f32 - 105.0) * 0.03).collect();
            let q = quantize(&x, 1, k).unwrap();
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            project_gemv_int4(&m4, &q, &mut a);
            project_gemv_int4_fold(&m4, &q, &mut b);
            let mut maxd = 0.0f32;
            for c in 0..n {
                maxd = maxd.max((a[c] - b[c]).abs());
            }
            println!("shape {k}x{n}: max diff {maxd}");
            assert!(maxd <= 3e-3 * 20.0, "shape {k}x{n} max diff {maxd}");
        }
    }



    #[test]
    fn int4_gemv_bench() {
        // Kernel economics on cold streams: rotate over copies so
        // each call streams weights from DRAM, like a decode token.
        for &(k, n) in &[(640usize, 640usize), (640, 5120), (2560, 640)] {
            // Enough copies that the rotation streams from DRAM
            // (>= ~48 MB of weights), like a real decode token.
            let copies = (48_000_000 / (k * n)).max(6).min(160);
            let mut ms = Vec::new();
            let mut m4s = Vec::new();
            for i in 0..copies {
                let mut m = test_matrix(k, n, 100 + i as u32);
                m.packed_gemv = WeightStore::Owned(crate::layout::pack_gemv(m.weights(), k, n));
                m.packed_sdot = WeightStore::Owned(crate::layout::pack_sdot(m.weights(), k, n));
                m4s.push(Int4Matrix::from_matrix(&m));
                ms.push(m);
            }
            let x: Vec<f32> = (0..k).map(|i| ((i * 53 % 211) as f32 - 105.0) * 0.03).collect();
            let q = quantize(&x, 1, k).unwrap();
            let iters = 120;
            // int8 production path (project_s rows == 1).
            let t0 = std::time::Instant::now();
            let mut sink = 0.0f32;
            for it in 0..iters {
                let out = project_s(&ms[it % copies], &q, 1, 1);
                sink += out[it % n];
            }
            let t8 = t0.elapsed().as_secs_f64() / iters as f64;
            // int4 kernel.
            let mut out4 = vec![0.0f32; n];
            let t0 = std::time::Instant::now();
            for it in 0..iters {
                project_gemv_int4(&m4s[it % copies], &q, &mut out4);
                sink += out4[it % n];
            }
            let t4 = t0.elapsed().as_secs_f64() / iters as f64;
            let macs = (k * n) as f64;
            println!(
                "shape {k}x{n}: int8 {:.1} us ({:.2} GMAC/s) | int4 {:.1} us ({:.2} GMAC/s) | ratio {:.3} (sink {sink:.2})",
                t8 * 1e6, macs / t8 / 1e9, t4 * 1e6, macs / t4 / 1e9, t4 / t8
            );
        }
    }

    #[test]
    fn int4_close_to_int8() {
        let m = test_matrix(640, 64, 7);
        let m4 = Int4Matrix::from_matrix(&m);
        let x: Vec<f32> = (0..640).map(|i| ((i * 53 % 211) as f32 - 105.0) * 0.03).collect();
        let q = quantize(&x, 1, 640).unwrap();
        let r8 = project_s(&m, &q, 1, 1);
        let mut r4 = vec![0.0f32; 64];
        project_gemv_int4_fold(&m4, &q, &mut r4);
        let rms = (r8.iter().map(|v| v * v).sum::<f32>() / 64.0).sqrt();
        let err = (0..64).map(|c| (r8[c] - r4[c]).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.25 * rms, "max err {err} vs rms {rms}");
    }
}
