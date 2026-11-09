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
        for j in 0..k {
            let a = ((x[r * k + j] / scale).round_ties_even() + zero as f32).clamp(0.0, 255.0) as i32;
            values[r * k + j] = (a - 128) as i8;
            sum += a - zero;
        }
        centered_sums[r] = sum;
    }
    Ok(Quantized { values, centered_sums, scale, zero })
}

/// projection.h `Projection::multiply` semantics (integer-exact).
pub fn project(m: &Matrix, q: &Quantized, rows: usize) -> Vec<f32> {
    let (k, n) = (m.k, m.n);
    let mut acc = vec![0i32; rows * n];
    for r in 0..rows {
        for j in 0..k {
            let a = q.values[r * k + j] as i32;
            if a == 0 {
                continue;
            }
            let wrow = &m.weights[j * n..(j + 1) * n];
            let arow = &mut acc[r * n..(r + 1) * n];
            for c in 0..n {
                arow[c] += a * wrow[c] as i32;
            }
        }
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
                let mut total = 0.0f32;
                for j in 0..64 {
                    total = q[r * 64 + j].mul_add(k[c * 64 + j], total);
                }
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
        for v in expanded.iter_mut() {
            let e = kernels::erf_mlas(*v / SQRT2);
            *v *= 0.5f32 * (e + 1.0f32);
        }
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
