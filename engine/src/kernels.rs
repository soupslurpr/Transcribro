//! Numeric kernels replicating the reference MLAS accumulation semantics
//! exactly as measured on x86-64 (AVX-512 host, MLAS from ONNX Runtime
//! 1.23.2). The models below were validated bit-for-bit against the real
//! MLAS static library with randomized probes (see Gate 2 log in NOTES.md):
//!
//! * General SGEMM (TransA=NoTrans): K is split into blocks of an adjusted
//!   StrideK; each block is an independent FMA partial sum from 0.0, and the
//!   partials are added to the total in ascending order.
//! * M1 TransposeB kernel (AVX, mul+add, no FMA): 8 lane partials
//!   (lane = k mod 8), each a mul-then-add chain, reduced as
//!   (((l0+l1)+l2)+l3) + (((l4+l5)+l6)+l7), then the beta seed is added.
//! * M1 kernel (AVX, mul+add, no FMA): K in groups of 4; each group is a
//!   sequential mul-add partial starting from the first product; partials
//!   accumulate into C in ascending order.
//!
//! Rust never contracts `a * b + c` into an FMA, so plain expressions stay
//! mul+add like the oracle's `-ffp-contract=off` build; `mul_add` is used
//! only where the reference fuses.

/// Eigen's horizontal sum for the 80-wide frontend rows under SSE2
/// (PacketSize 4, two packet accumulators), as decoded from Eigen's Redux.h.
pub fn eigen_sum_80(row: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), 80);
    // Eigen LinearVectorizedTraversal redux, PacketSize 4, alignedStart 0:
    // two packet accumulators over elements [8j + l] and [8j + 4 + l],
    // combined lane-wise, then predux = (l0 + l2) + (l1 + l3).
    let mut p0 = [0.0f32; 4];
    let mut p1 = [0.0f32; 4];
    for j in 0..10 {
        for l in 0..4 {
            p0[l] += row[8 * j + l];
            p1[l] += row[8 * j + 4 + l];
        }
    }
    let lane = [
        p0[0] + p1[0],
        p0[1] + p1[1],
        p0[2] + p1[2],
        p0[3] + p1[3],
    ];
    (lane[0] + lane[2]) + (lane[1] + lane[3])
}

/// The SGEMM K-block size for a (N, K) call with TransA=NoTrans, per
/// MlasSgemmOperation's stride adjustment (MLAS_SGEMM_STRIDEK = 128).
fn sgemm_stride_k(n: usize, k: usize) -> usize {
    let mut stride_n = 128usize;
    let mut stride_k = 128usize;
    if n >= k {
        while stride_k / 2 >= k {
            stride_n *= 2;
            stride_k /= 2;
        }
    } else {
        while stride_n > 16 && stride_n / 2 >= n {
            stride_k *= 2;
            stride_n /= 2;
        }
    }
    stride_k
}

/// The frontend-linear GEMM form used by the reference `gemm` helper:
/// C[t][j] = sum_k A[t][k] * B[k][j], where the MLAS call is
/// MlasGemm(NoTrans, NoTrans, M=n, N=m, K=k) — so the SGEMM K-blocking is
/// governed by (N=m, K=k). beta = 0.
// ---------------------------------------------------------------------
// Gate 6 S5: aarch64 NEON fast paths. The x86 forms in this file are
// bit-exact models of MLAS kernels and stay untouched; on aarch64 the
// bar is transcript equivalence (NOTES.md numeric policy), so these
// paths use clean ascending-FMA orders with NEON vectorization. The exp
// polynomial is the same MLAS Horner chain evaluated lane-wise, which
// is per-element identical to the scalar form apart from the softmax
// summation order.

/// Ascending dot product. On non-aarch64 this is exactly the scalar
/// mul_add chain the attention code used inline (bit-identical).
pub fn dot_asc(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        debug_assert_eq!(a.len(), b.len());
        unsafe {
            use std::arch::aarch64::*;
            let n = a.len();
            let (ap, bp) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut i = 0usize;
            while i + 16 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
                i += 16;
            }
            let s01 = vaddq_f32(acc0, acc1);
            let s23 = vaddq_f32(acc2, acc3);
            let s = vaddq_f32(s01, s23);
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), s);
            let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            while i < n {
                total = a[i].mul_add(b[i], total);
                i += 1;
            }
            return total;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut total = 0.0f32;
        for i in 0..a.len() {
            total = a[i].mul_add(b[i], total);
        }
        total
    }
}

/// `sumexp_exp` evaluated on 4 lanes: the identical MLAS Horner chain
/// with vfmaq (per-element identical), finished with the exact
/// power-of-two scale per lane.
/// `p * 2^m` for 4 lanes with integer-valued m in [-150, 200]:
/// two bit-constructed power-of-two multiplies (m split into a
/// clamped part and a remainder, both in the normal-exponent
/// range). Two roundings instead of the exact f64 path's one --
/// immaterial under the aarch64 transcript-equivalence bar, and it
/// keeps the softmax/erf inner loops fully vectorized.
#[cfg(target_arch = "aarch64")]
unsafe fn scale_pow2_vec(
    p: std::arch::aarch64::float32x4_t,
    m: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let m1 = vminq_f32(vmaxq_f32(m, vdupq_n_f32(-100.0)), vdupq_n_f32(100.0));
    let m2 = vsubq_f32(m, m1);
    let bits1 = vreinterpretq_f32_u32(vshlq_n_u32(
        vreinterpretq_u32_s32(vaddq_s32(vcvtq_s32_f32(m1), vdupq_n_s32(127))),
        23,
    ));
    let bits2 = vreinterpretq_f32_u32(vshlq_n_u32(
        vreinterpretq_u32_s32(vaddq_s32(vcvtq_s32_f32(m2), vdupq_n_s32(127))),
        23,
    ));
    vmulq_f32(vmulq_f32(p, bits1), bits2)
}

#[cfg(target_arch = "aarch64")]
unsafe fn sumexp_vec(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let v = vmaxq_f32(x, vdupq_n_f32(-103.9720840454));
    let t = vfmaq_f32(vdupq_n_f32(12582912.0), vdupq_n_f32(1.4426950408889634), v);
    let m = vsubq_f32(t, vdupq_n_f32(12582912.0));
    let mut r = vfmaq_f32(v, vdupq_n_f32(-0.693145752), m);
    r = vfmaq_f32(r, vdupq_n_f32(-1.42860677e-6), m);
    let mut p = vdupq_n_f32(f32::from_bits(0x3ab4a000));
    p = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3c092f6e)), p, r);
    p = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3d2aadad)), p, r);
    p = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3e2aaa28)), p, r);
    p = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3efffffb)), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
    scale_pow2_vec(p, m)
}

#[cfg(target_arch = "aarch64")]
fn softmax_row_neon(row: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = row.len();
    if n == 0 {
        return;
    }
    unsafe {
        let mut max4 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;
        while i + 4 <= n {
            max4 = vmaxq_f32(max4, vld1q_f32(row.as_ptr().add(i)));
            i += 4;
        }
        let mut ma = [0.0f32; 4];
        vst1q_f32(ma.as_mut_ptr(), max4);
        let mut max = f32::NEG_INFINITY;
        for &v in &ma {
            if v > max {
                max = v;
            }
        }
        while i < n {
            if row[i] > max {
                max = row[i];
            }
            i += 1;
        }
        let neg = vdupq_n_f32(-max);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut tail_acc = 0.0f32;
        let mut i = 0usize;
        while i + 4 <= n {
            let e = sumexp_vec(vaddq_f32(vld1q_f32(row.as_ptr().add(i)), neg));
            vst1q_f32(row.as_mut_ptr().add(i), e);
            acc4 = vaddq_f32(acc4, e);
            i += 4;
        }
        while i < n {
            let e = sumexp_exp(row[i] - max);
            row[i] = e;
            tail_acc += e;
            i += 1;
        }
        let mut aa = [0.0f32; 4];
        vst1q_f32(aa.as_mut_ptr(), acc4);
        let sum = ((aa[0] + aa[1]) + (aa[2] + aa[3])) + tail_acc;
        let scale = 1.0f32 / sum;
        let sv = vdupq_n_f32(scale);
        let mut i = 0usize;
        while i + 4 <= n {
            let v = vmulq_f32(vld1q_f32(row.as_ptr().add(i)), sv);
            vst1q_f32(row.as_mut_ptr().add(i), v);
            i += 4;
        }
        while i < n {
            row[i] *= scale;
            i += 1;
        }
    }
}

/// NEON SGEMM (NoTrans/NoTrans): C = A * B, all row-major. Rank-1
/// update form over k with contiguous B rows -- the scalar model's
/// StrideK walk over B columns is cache-hostile and is the single
/// biggest on-device cost. Rows are split across scoped threads for
/// large problems.
#[cfg(target_arch = "aarch64")]
fn gemm_nn_neon(a: &[f32], n: usize, k: usize, b: &[f32], m: usize, c: &mut [f32]) {
    let work = (n as u64) * (k as u64) * (m as u64);
    if n >= 8 && work >= (1 << 30) {
        let c_addr = c.as_mut_ptr() as usize;
        crate::pool::parallel_rows(n, 4, 8, &|start, take| {
            let cslice = unsafe {
                std::slice::from_raw_parts_mut((c_addr as *mut f32).add(start * m), take * m)
            };
            gemm_rows_neon(a, start, start + take, k, b, m, cslice);
        });
        return;
    }
    gemm_rows_neon(a, 0, n, k, b, m, c);
}

#[cfg(target_arch = "aarch64")]
fn gemm_rows_neon(a: &[f32], t0: usize, t1: usize, k: usize, b: &[f32], m: usize, c: &mut [f32]) {
    use std::arch::aarch64::*;
    let bp = b.as_ptr();
    for t in t0..t1 {
        let arow = &a[t * k..(t + 1) * k];
        let crow = &mut c[(t - t0) * m..(t - t0 + 1) * m];
        let ap = arow.as_ptr();
        let mut j = 0usize;
        while j + 16 <= m {
            unsafe {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);
                for kk in 0..k {
                    let av = vdupq_n_f32(*ap.add(kk));
                    let brow = bp.add(kk * m + j);
                    acc0 = vfmaq_f32(acc0, av, vld1q_f32(brow));
                    acc1 = vfmaq_f32(acc1, av, vld1q_f32(brow.add(4)));
                    acc2 = vfmaq_f32(acc2, av, vld1q_f32(brow.add(8)));
                    acc3 = vfmaq_f32(acc3, av, vld1q_f32(brow.add(12)));
                }
                let cp = crow.as_mut_ptr().add(j);
                vst1q_f32(cp, acc0);
                vst1q_f32(cp.add(4), acc1);
                vst1q_f32(cp.add(8), acc2);
                vst1q_f32(cp.add(12), acc3);
            }
            j += 16;
        }
        while j < m {
            let mut total = 0.0f32;
            for kk in 0..k {
                total = arow[kk].mul_add(b[kk * m + j], total);
            }
            crow[j] = total;
            j += 1;
        }
    }
}

/// NEON conv1d: same im2col column buffer as the scalar model, but the
/// accumulation is a k-ascending rank-1 update over the contiguous
/// time axis (bias folded into the initial value).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn conv1d_neon(
    input: &[f32],
    channels: usize,
    length: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    filters: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    use std::arch::aarch64::*;
    let count = (length + 2 * pad - kernel) / stride + 1;
    let k_total = channels * kernel;
    let mut columns = vec![0.0f32; k_total * count];
    for c in 0..channels {
        for kw in 0..kernel {
            let row = (c * kernel + kw) * count;
            for t in 0..count {
                let pos = t as isize * stride as isize + kw as isize - pad as isize;
                if pos >= 0 && (pos as usize) < length {
                    columns[row + t] = input[c * length + pos as usize];
                }
            }
        }
    }
    let mut out = vec![0.0f32; filters * count];
    for f in 0..filters {
        let w_row = &weight[f * k_total..(f + 1) * k_total];
        let init = bias.map(|b| b[f]).unwrap_or(0.0);
        let orow = &mut out[f * count..(f + 1) * count];
        orow.fill(init);
        for kk in 0..k_total {
            let w = w_row[kk];
            let col = &columns[kk * count..(kk + 1) * count];
            unsafe {
                let wv = vdupq_n_f32(w);
                let mut t = 0usize;
                while t + 4 <= count {
                    let cur = vld1q_f32(orow.as_ptr().add(t));
                    let v = vfmaq_f32(cur, wv, vld1q_f32(col.as_ptr().add(t)));
                    vst1q_f32(orow.as_mut_ptr().add(t), v);
                    t += 4;
                }
                while t < count {
                    orow[t] = w.mul_add(col[t], orow[t]);
                    t += 1;
                }
            }
        }
    }
    out
}

/// `erf_mlas` on 4 lanes: both polynomial branches evaluated and
/// blend-selected, exactly like the scalar form; the power-of-two
/// scale is the exact f64 construction per lane.
#[cfg(target_arch = "aarch64")]
unsafe fn erf_vec(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x8000_0000));
    let a = vminq_f32(vabsq_f32(x), vdupq_n_f32(3.925));
    let z = vmulq_f32(a, a);
    let mut rs = vdupq_n_f32(-5.99104969e-4f32);
    for &c in &[4.99339588e-3f32, -2.67667342e-2, 1.12818025e-1, -3.76124859e-1, 1.28379151e-1] {
        rs = vfmaq_f32(vdupq_n_f32(c), rs, z);
    }
    rs = vfmaq_f32(a, rs, a);
    let mut rb = vdupq_n_f32(1.72948930e-5f32);
    for &c in &[-3.83208680e-4f32, 3.88393435e-3, -2.42545605e-2, 1.06777847e-1, 6.34846687e-1, 1.28717512e-1] {
        rb = vfmaq_f32(vdupq_n_f32(c), rb, a);
    }
    rb = vfmaq_f32(a, rb, a);
    let rb = vmaxq_f32(vnegq_f32(rb), vdupq_n_f32(-88.3762626647949));
    let t = vfmaq_f32(vdupq_n_f32(12582912.0), vdupq_n_f32(1.4426950408889634), rb);
    let r = vsubq_f32(t, vdupq_n_f32(12582912.0));
    let mut fx = vfmaq_f32(rb, vdupq_n_f32(-0.693145752), r);
    fx = vfmaq_f32(fx, vdupq_n_f32(-1.42860677e-6), r);
    let mut y = vdupq_n_f32(1.38319808e-3f32);
    for &c in &[8.37550033e-3f32, 4.16689515e-2, 1.66664466e-1, 4.99999851e-1, 1.0, 1.0] {
        y = vfmaq_f32(vdupq_n_f32(c), y, fx);
    }
    let yv = vsubq_f32(vdupq_n_f32(1.0), scale_pow2_vec(y, r));
    let mask = vcleq_f32(a, vdupq_n_f32(0.921875));
    let res = vbslq_f32(mask, rs, yv);
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(res), sign))
}

/// Elementwise erf (MLAS model) over a slice; NEON-vectorized on
/// aarch64, the scalar function elsewhere.
pub fn erf_mlas_slice(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = values.len();
        let mut i = 0usize;
        unsafe {
            while i + 4 <= n {
                let v = erf_vec(vld1q_f32(values.as_ptr().add(i)));
                vst1q_f32(values.as_mut_ptr().add(i), v);
                i += 4;
            }
        }
        while i < n {
            values[i] = erf_mlas(values[i]);
            i += 1;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in values.iter_mut() {
            *v = erf_mlas(*v);
        }
    }
}

/// The encoder MLP's exact-erf GELU applied in place:
/// v *= 0.5 * (erf(v / sqrt(2)) + 1). SQRT2 matches encoder.rs.
pub fn gelu_erf_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        const SQRT2: f32 = 1.4142135381698608f32; // bits 0x3fb504f3
        let n = values.len();
        let mut i = 0usize;
        unsafe {
            let sq = vdupq_n_f32(SQRT2);
            while i + 4 <= n {
                let v = vld1q_f32(values.as_ptr().add(i));
                let e = erf_vec(vdivq_f32(v, sq));
                let f = vmulq_f32(vdupq_n_f32(0.5), vaddq_f32(e, vdupq_n_f32(1.0)));
                vst1q_f32(values.as_mut_ptr().add(i), vmulq_f32(v, f));
                i += 4;
            }
        }
        while i < n {
            let e = erf_mlas(values[i] / SQRT2);
            values[i] *= 0.5f32 * (e + 1.0f32);
            i += 1;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        const SQRT2: f32 = 1.4142135381698608f32;
        for v in values.iter_mut() {
            let e = erf_mlas(*v / SQRT2);
            *v *= 0.5f32 * (e + 1.0f32);
        }
    }
}

/// `logistic` on 4 lanes (identical rational polynomial, exact vector
/// divide per lane).
#[cfg(target_arch = "aarch64")]
unsafe fn logistic_vec(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let c = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-18.0)), vdupq_n_f32(18.0));
    let v2 = vmulq_f32(c, c);
    let mut p = vdupq_n_f32(4.37031012579801e-11f32);
    for &k in &[1.15627324459942e-07f32, 6.08574864600143e-05, 8.51377133304701e-03, 2.48287947061529e-01] {
        p = vfmaq_f32(vdupq_n_f32(k), p, v2);
    }
    p = vmulq_f32(p, c);
    let mut q = vdupq_n_f32(6.10247389755681e-13f32);
    for &k in &[5.76102136993427e-09f32, 6.29106785017040e-06, 1.70198817374094e-03, 1.16817656904453e-01, 9.93151921023180e-01] {
        q = vfmaq_f32(vdupq_n_f32(k), q, v2);
    }
    let r = vaddq_f32(vdivq_f32(p, q), vdupq_n_f32(0.5));
    vminq_f32(vmaxq_f32(r, vdupq_n_f32(0.0)), vdupq_n_f32(1.0))
}

pub fn gemm_nn(a: &[f32], n: usize, k: usize, b: &[f32], m: usize, c: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        gemm_nn_neon(a, n, k, b, m, c);
        return;
    }
    let stride_k = sgemm_stride_k(m, k);
    for t in 0..n {
        for j in 0..m {
            let mut total = 0.0f32;
            let mut first = true;
            let mut s = 0usize;
            while s < k {
                let e = (s + stride_k).min(k);
                let mut partial = 0.0f32;
                for kk in s..e {
                    partial = a[t * k + kk].mul_add(b[kk * m + j], partial);
                }
                total = if first { partial } else { total + partial };
                first = false;
                s = e;
            }
            c[t * m + j] = total;
        }
    }
}

/// The VAD LSTM's vector-matrix product: the reference calls
/// MlasGemm(NoTrans, Trans, 1, rows, k) which dispatches to the M1
/// TransposeB kernel. `w` is row-major [rows][k]; with `accumulate` the old
/// C value is the beta seed, otherwise +0.0 is added (both as the kernel
/// does).
pub fn gemv_t(c: &mut [f32], x: &[f32], w: &[f32], accumulate: bool) {
    let k = x.len();
    for i in 0..c.len() {
        c[i] = m1tb_dot(x, &w[i * k..(i + 1) * k], c[i], accumulate);
    }
}

/// M1 TransposeB kernel: C[i] = dot(x, W[i]) with 8 mul-add lanes and the
/// 4-row-block reduction tree; then `total + seed` where seed is the old C
/// value for beta = 1 and +0.0 for beta = 0 (the kernel always adds it).
/// `w` is row-major [rows][k].
pub fn m1tb_dot(x: &[f32], w_row: &[f32], seed: f32, beta: bool) -> f32 {
    let k = x.len();
    let mut lane = [0.0f32; 8];
    for kk in 0..k {
        lane[kk % 8] = lane[kk % 8] + x[kk] * w_row[kk];
    }
    let lo = ((lane[0] + lane[1]) + lane[2]) + lane[3];
    let hi = ((lane[4] + lane[5]) + lane[6]) + lane[7];
    (lo + hi) + if beta { seed } else { 0.0 }
}

/// M1 kernel (NoTrans B): dot of `a` with column `n` of B[K][N] (row-major).
/// MlasSgemmKernelM1Avx accumulates into C block by block: sequential
/// mul-add partials over blocks of 4, then a block of 2, then a single
/// element (the kernel's CountK decomposition), each partial added to the
/// running total in ascending order; the first block also absorbs the seed
/// (old C for beta = 1, +0.0 for beta = 0). Gate 5b: the earlier model used
/// one trailing block of K % 4, which mis-associates the sum when
/// K % 4 == 3 (the kernel splits that tail 2 + 1) - probe-verified against
/// SgemmKernelM1Avx.S and 62 MLAS probe cases at K = 2..602, 0 mismatches.
pub fn m1_dot(a: &[f32], b: &[f32], n: usize, ncols: usize, seed: f32, beta: bool) -> f32 {
    let k = a.len();
    let mut total = if beta { seed } else { 0.0 };
    let mut first = true;
    let mut s = 0usize;
    while s < k {
        let remaining = k - s;
        let block = if remaining == 3 { 2 } else { remaining.min(4) };
        let e = s + block;
        let mut partial = a[s] * b[s * ncols + n];
        for kk in (s + 1)..e {
            partial = partial + a[kk] * b[kk * ncols + n];
        }
        total = if first { partial + total } else { total + partial };
        first = false;
        s = e;
    }
    total
}

/// Leaf K-block boundaries for one output element of an MLAS
/// convolution, replicating the algorithm selection in
/// MlasConvPrepare/MlasConv: ExpandThenGemm (filters > count) is one SGEMM
/// call; the segmented path runs MlasConvOperation, whose own K blocks are
/// each re-blocked by the SGEMM call they feed (the SGEMM blocking depends
/// on the size of the N segment the element falls in).
fn conv_leaf_blocks(filters: usize, count: usize, k: usize, t: usize) -> Vec<(usize, usize)> {
    if filters > count {
        let stride_k = sgemm_stride_k(count, k);
        let mut blocks = Vec::new();
        let mut s = 0;
        while s < k {
            let e = (s + stride_k).min(k);
            blocks.push((s, e));
            s = e;
        }
        return blocks;
    }
    // MlasConvOperation stride adjustment for SegmentCountN = count.
    let mut stride_n = 128usize;
    let mut stride_k = 128usize;
    if count >= k {
        while stride_k / 2 >= k {
            stride_n *= 2;
            stride_k /= 2;
        }
    } else {
        while stride_n > 16 && stride_n / 2 >= count {
            stride_k *= 2;
            stride_n /= 2;
        }
    }
    let seg_start = (t / stride_n) * stride_n;
    let seg = (count - seg_start).min(stride_n);
    let mut blocks = Vec::new();
    let mut s = 0;
    while s < k {
        let e = (s + stride_k).min(k);
        let sub = sgemm_stride_k(seg, e - s);
        let mut u = s;
        while u < e {
            let ue = (u + sub).min(e);
            blocks.push((u, ue));
            u = ue;
        }
        s = e;
    }
    blocks
}

/// 1D convolution with the reference's MLAS semantics: im2col columns in
/// (channel, tap) order with explicit zero padding, the filter layout
/// [filter][channel][tap], accumulation per `conv_leaf_blocks`, and the bias
/// added once at the end (the activation epilogue) when present. When
/// filters == 1 the SGEMM dispatches to the M1 kernel, whose mul-add
/// group-of-4 semantics replace the FMA partials.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    input: &[f32],
    channels: usize,
    length: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    filters: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        return conv1d_neon(input, channels, length, weight, bias, filters, kernel, stride, pad);
    }
    let count = (length + 2 * pad - kernel) / stride + 1;
    let k_total = channels * kernel;
    // Column buffer, [k_total][count] like the MLAS working buffer.
    let mut columns = vec![0.0f32; k_total * count];
    for c in 0..channels {
        for kw in 0..kernel {
            let row = (c * kernel + kw) * count;
            for t in 0..count {
                let pos = t as isize * stride as isize + kw as isize - pad as isize;
                if pos >= 0 && (pos as usize) < length {
                    columns[row + t] = input[c * length + pos as usize];
                }
            }
        }
    }
    let mut out = vec![0.0f32; filters * count];
    for f in 0..filters {
        let w_row = &weight[f * k_total..(f + 1) * k_total];
        for t in 0..count {
            let value = if filters == 1 {
                m1_dot(w_row, &columns, t, count, 0.0, false)
            } else if count == 1 {
                // The GEMM's N == 1 flips MlasSgemmOperation to the M1
                // TransposeB kernel over the single column.
                m1tb_dot(&columns[..k_total], w_row, 0.0, false)
            } else {
                let mut total = 0.0f32;
                let mut first = true;
                for (s, e) in conv_leaf_blocks(filters, count, k_total, t) {
                    let mut partial = 0.0f32;
                    for kk in s..e {
                        partial = w_row[kk].mul_add(columns[kk * count + t], partial);
                    }
                    total = if first { partial } else { total + partial };
                    first = false;
                }
                total
            };
            out[f * count + t] = match bias {
                Some(b) => value + b[f],
                None => value,
            };
        }
    }
    out
}

/// MLAS logistic (Eigen rational polynomial, FMA Horner form as dispatched
/// on x86). Coefficients from core/mlas/lib/logistic.cpp.
pub fn logistic(x: f32) -> f32 {
    let c = x.clamp(-18.0, 18.0);
    let v2 = c * c;
    let p = 4.37031012579801e-11f32
        .mul_add(v2, 1.15627324459942e-07)
        .mul_add(v2, 6.08574864600143e-05)
        .mul_add(v2, 8.51377133304701e-03)
        .mul_add(v2, 2.48287947061529e-01)
        * c;
    let q = 6.10247389755681e-13f32
        .mul_add(v2, 5.76102136993427e-09)
        .mul_add(v2, 6.29106785017040e-06)
        .mul_add(v2, 1.70198817374094e-03)
        .mul_add(v2, 1.16817656904453e-01)
        .mul_add(v2, 9.93151921023180e-01);
    (p / q + 0.5).clamp(0.0, 1.0)
}

/// MLAS tanh (Eigen rational polynomial, FMA Horner form as dispatched on
/// x86). Coefficients from core/mlas/lib/tanh.cpp.
pub fn tanh_mlas(x: f32) -> f32 {
    let c = x.clamp(-9.0, 9.0);
    let v2 = c * c;
    let p = (-2.76076847742355e-16f32)
        .mul_add(v2, 2.00018790482477e-13)
        .mul_add(v2, -8.60467152213735e-11)
        .mul_add(v2, 5.12229709037114e-08)
        .mul_add(v2, 1.48572235717979e-05)
        .mul_add(v2, 6.37261928875436e-04)
        .mul_add(v2, 4.89352455891786e-03)
        * c;
    let q = 1.19825839466702e-06f32
        .mul_add(v2, 1.18534705686654e-04)
        .mul_add(v2, 2.26843463243900e-03)
        .mul_add(v2, 4.89352518554385e-03);
    p / q
}

pub fn silu_inplace(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = values.len();
        let mut i = 0usize;
        unsafe {
            while i + 4 <= n {
                let v = vld1q_f32(values.as_ptr().add(i));
                let r = vmulq_f32(v, logistic_vec(v));
                vst1q_f32(values.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
        while i < n {
            values[i] = values[i] * logistic(values[i]);
            i += 1;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in values.iter_mut() {
            *v = *v * logistic(*v);
        }
    }
}

/// The reference's ReLU: `std::max(value, 0.0f)` semantics (keeps -0.0 and
/// NaN), not a clamp.
pub fn relu_cpp(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Exact power-of-two scaling: the correctly rounded value of `p * 2^m`
/// (what vscalefps / MlasPowerOf2 produce), computed through f64 where the
/// product is always exact.
fn scale_pow2(p: f32, m: f32) -> f32 {
    (p as f64 * 2f64.powi(m as i32)) as f32
}

/// The MLAS exp() element function used by MlasComputeSoftmax's AVX-512
/// SumExp kernel (constants from core/mlas/lib/compute.cpp's
/// MlasExpConstants; evaluation order from TransKernelAvx512F.asm). The
/// input is the score already biased by the negative row maximum; the lower
/// clamp is applied inside, exactly as the kernel does. Probe-validated
/// bit-for-bit as part of `softmax_row` (see Gate 3 log in NOTES.md).
pub fn sumexp_exp(v_in: f32) -> f32 {
    let v = if v_in < -103.9720840454f32 {
        -103.9720840454f32
    } else {
        v_in
    };
    let t = 1.4426950408889634f32.mul_add(v, 12582912.0);
    let m = t - 12582912.0;
    let r = m.mul_add(-0.693145752f32, v);
    let r = m.mul_add(-1.42860677e-6f32, r);
    let mut p = f32::from_bits(0x3ab4a000);
    p = p.mul_add(r, f32::from_bits(0x3c092f6e));
    p = p.mul_add(r, f32::from_bits(0x3d2aadad));
    p = p.mul_add(r, f32::from_bits(0x3e2aaa28));
    p = p.mul_add(r, f32::from_bits(0x3efffffb));
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    scale_pow2(p, m)
}

/// MlasComputeSoftmax over a single row, in place, replicating the AVX-512
/// SumExp kernel's accumulation: one 16-lane accumulator with per-lane
/// ascending adds (48-element blocks, then masked 16-element groups), the
/// ((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)) reduction tree with
/// s_i = acc[i] + acc[i+8], and the reciprocal-multiply output pass.
/// Probe-validated bit-for-bit on 288 rows / 50,304 elements including
/// -inf masked patterns (Gate 3 log).
pub fn softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        softmax_row_neon(row);
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &x in row.iter() {
        if x > max {
            max = x;
        }
    }
    let neg = -max;
    let mut acc = [0.0f32; 16];
    for (i, v) in row.iter_mut().enumerate() {
        let e = sumexp_exp(*v + neg);
        *v = e;
        acc[i % 16] += e;
    }
    let s = |i: usize| acc[i] + acc[i + 8];
    let sum = ((s(0) + s(1)) + (s(2) + s(3))) + ((s(4) + s(5)) + (s(6) + s(7)));
    let scale = 1.0f32 / sum;
    for v in row.iter_mut() {
        *v *= scale;
    }
}

/// MlasComputeErf's element function on this host (ErfKernelFma3): erf.cpp's
/// vector algorithm with FMA Horner steps applied per element — the clamped
/// |x|, the 0.921875 split, both rational/polynomial branches computed and
/// blend-selected, the sign restored by OR-ing the sign bit. Probe-validated
/// bit-for-bit on 2,171,222 values (dense grid, boundaries, lengths 1..130
/// and multi-million random arrays; Gate 3 log).
pub fn erf_mlas(x: f32) -> f32 {
    const SMALL: [f32; 6] = [
        -5.99104969e-4, 4.99339588e-3, -2.67667342e-2, 1.12818025e-1, -3.76124859e-1, 1.28379151e-1,
    ];
    const BIG: [f32; 7] = [
        1.72948930e-5, -3.83208680e-4, 3.88393435e-3, -2.42545605e-2, 1.06777847e-1, 6.34846687e-1,
        1.28717512e-1,
    ];
    let sign = x.to_bits() & 0x8000_0000;
    let a = x.abs().min(3.925f32);
    let z = a * a;
    let mut rs = SMALL[0];
    for &c in &SMALL[1..] {
        rs = rs.mul_add(z, c);
    }
    rs = rs.mul_add(a, a);
    if a <= 0.921875f32 {
        return f32::from_bits(rs.to_bits() | sign);
    }
    let mut rb = BIG[0];
    for &c in &BIG[1..] {
        rb = rb.mul_add(a, c);
    }
    rb = rb.mul_add(a, a);
    let rb = (-rb).max(-88.3762626647949f32);
    let r = 1.4426950408889634f32.mul_add(rb, 12582912.0) - 12582912.0;
    let fx = r.mul_add(-0.693145752f32, rb);
    let fx = r.mul_add(-1.42860677e-6f32, fx);
    let mut y = 1.38319808e-3f32;
    y = y.mul_add(fx, 8.37550033e-3);
    y = y.mul_add(fx, 4.16689515e-2);
    y = y.mul_add(fx, 1.66664466e-1);
    y = y.mul_add(fx, 4.99999851e-1);
    y = y.mul_add(fx, 1.0);
    y = y.mul_add(fx, 1.0);
    let y = 1.0f32 - scale_pow2(y, r);
    f32::from_bits(y.to_bits() | sign)
}

/// Eigen's generic `psincos_float` (arch/Default/GenericPacketMathFunctions.h),
/// non-FMA branch — the build the C++ oracle uses (-ffp-contract=off, no
/// -mfma): |x| reduced by the 2^23 rounding trick to the nearest multiple
/// of pi/2, four mul+add extended-precision reduction steps, then the
/// sin/cos polynomials selected by the octant parity, with the sign taken
/// from bit 1 of the octant counter (xor the input sign for sin). Every
/// operation is elementwise, so this scalar form is the packet computation.
/// The Payne-Hanek huge-input path is unreachable for the decoder: angles
/// are float(past + r) * frequency[i] with past + r <= 4096 and
/// frequency <= 1.0, i.e. |x| <= 4096, far below the 18838 (cos) / 25966
/// (sin) thresholds. Probe-fitted bit-for-bit against the pinned Eigen on
/// 1,078,564 values including every real angle (Gate 4 log in NOTES.md).
fn eigen_psincos(input: f32, sine: bool) -> f32 {
    let xa = input.abs();
    let y0 = xa * 0.636619746685028076171875f32; // 2/pi
    let y_round = y0 + 12582912.0f32; // 2^23
    let yi = y_round.to_bits();
    let y = y_round - 12582912.0f32;
    let mut x = xa;
    x = y * f32::from_bits(0xbfc90000) + x; // -1.5703125
    x = y * f32::from_bits(0xb9fdc000) + x; // -0.000483989715576171875
    x = y * f32::from_bits(0x342ee000) + x; // 1.62865035235881805419921875e-07
    x = y * f32::from_bits(0x2e74b9ee) + x; // 5.5644315544167710640977020375430583953857421875e-11
    let x2 = x * x;
    let mut y1 = 2.4372266125283204019069671630859375e-05f32;
    y1 = y1 * x2 + -0.00138865201734006404876708984375f32;
    y1 = y1 * x2 + 0.041666619479656219482421875f32;
    y1 = y1 * x2 + -0.5f32;
    y1 = y1 * x2 + 1.0f32;
    let mut y2 = -0.0001959234114083702898469196984621021329076029360294342041015625f32;
    y2 = y2 * x2 + 0.00833268736556168516937947998712843400426208972930908203125f32;
    y2 = y2 * x2 + -0.166666620398229825550373561782180331647396087646484375f32;
    y2 = y2 * x2;
    y2 = y2 * x + x;
    let even = (yi & 1) == 0;
    let poly = if sine {
        if even { y2 } else { y1 }
    } else {
        if even { y1 } else { y2 }
    };
    let sign = if sine {
        (input.to_bits() ^ (yi << 30)) & 0x8000_0000
    } else {
        (yi.wrapping_add(1) << 30) & 0x8000_0000
    };
    f32::from_bits(poly.to_bits() ^ sign)
}

/// Eigen ArrayXf sin (packet path), as used by the decoder's rotary.
pub fn eigen_sin(x: f32) -> f32 {
    eigen_psincos(x, true)
}

/// Eigen ArrayXf cos (packet path), as used by the decoder's rotary.
pub fn eigen_cos(x: f32) -> f32 {
    eigen_psincos(x, false)
}
