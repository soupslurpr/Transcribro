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

use crate::encoder::{heads, normalize, project, quantize, run, unheads};
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
    let mut out = vec![0.0f32; rows * 640];
    let mut scores = vec![0.0f32; rows * memory];
    for h in 0..10 {
        let q = &queries[h * rows * 64..(h + 1) * rows * 64];
        let k = &keys[h * head_stride..h * head_stride + memory * 64];
        let v = &values[h * head_stride..h * head_stride + memory * 64];
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
                if mask == 1 && c > past + r {
                    *s = f32::NEG_INFINITY;
                }
            }
            kernels::softmax_row(row);
        }
        if rows == 1 {
            // MLAS dispatches M == 1 NoTrans SGEMM to the M1 kernel
            // (kernels::m1_dot; Gate 4 probe fit at K in {2, 17, 33, 224,
            // 602}, Gate 5b correction for K % 4 == 3 - see kernels.rs).
            for j in 0..64 {
                out[h * 64 + j] = kernels::m1_dot(&scores, v, j, 64, 0.0, false);
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
    out
}


/// One 128-column head GEMM block: MlasGemm(rows, 128, 640) over the
/// tiled head matrix. At rows == 1 MLAS takes the M1 kernel
/// (kernels::m1_dot; probe-fitted at Gates 4/5b, 0 mismatches; gemm_nn
/// fails 120 of 128 elements there).
fn head_block(model: &Model, normalized: &[f32], rows: usize, column: usize, block: &mut [f32]) {
    let b = &model.head[column * 640..column * 640 + 640 * 128];
    if rows == 1 {
        for j in 0..128 {
            block[j] = kernels::m1_dot(normalized, b, j, 128, 0.0, false);
        }
    } else {
        kernels::gemm_nn(normalized, rows, 640, b, 128, block);
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
    head: Head,
    mut dest: Dest,
) -> Result<(), String> {
    let rows = tokens.len();
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
        let qs = project(&model.decoder_matrices[layer * 8], &q_in, rows);
        let ks = project(&model.decoder_matrices[layer * 8 + 1], &q_in, rows);
        let vs = project(&model.decoder_matrices[layer * 8 + 2], &q_in, rows);
        let mut qheads = heads(&qs, rows);
        let mut kheads = heads(&ks, rows);
        let vheads = heads(&vs, rows);
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
        let joined = unheads(&attended, rows);
        let projected = run(&model.decoder_matrices[layer * 8 + 3], &joined, rows, 640)?;
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        // Cross-attention over this layer's slice of the cross-KV.
        let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[layer * 3 + 1]);
        let qs = run(&model.decoder_matrices[layer * 8 + 4], &normalized, rows, 640)?;
        let qheads = heads(&qs, rows);
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
        let joined = unheads(&attended, rows);
        let projected = run(&model.decoder_matrices[layer * 8 + 5], &joined, rows, 640)?;
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
        // SwiGLU FFN: logistic on the second half, gated by the first.
        let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[layer * 3 + 2]);
        let expanded = run(&model.decoder_matrices[layer * 8 + 6], &normalized, rows, 640)?;
        let width = 2560;
        let mut gated = vec![0.0f32; rows * width];
        for r in 0..rows {
            for j in 0..width {
                let x = expanded[r * width * 2 + width + j];
                let silu = x * kernels::logistic(x);
                gated[r * width + j] = expanded[r * width * 2 + j] * silu;
            }
        }
        let projected = run(&model.decoder_matrices[layer * 8 + 7], &gated, rows, width)?;
        for (h, p) in hidden.iter_mut().zip(&projected) {
            *h += p;
        }
    }
    let normalized = normalize(&hidden, rows, 640, &model.decoder_norms[42]);
    let mut block = vec![0.0f32; rows * 128];
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
    if let Dest::Cache(cache) = dest {
        cache.count = total;
    }
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
