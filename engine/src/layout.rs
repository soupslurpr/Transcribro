//! Kernel-ready weight layouts, single-sourced (Gate 6 speed round 2).
//!
//! These permutations were previously private to `model.rs` and ran at
//! every engine load on aarch64, materializing three extra copies of
//! every projection matrix (~209 MB of codes -> ~627 MB of packed
//! copies) plus the output head's dequantized FP32 tiles (84 MB).
//! Format TRMED003 stores exactly these bytes in the archive, so the
//! loader borrows them from the mmap instead of recomputing them.
//! The tools crate depends on this module (path dependency on the
//! engine lib) so the converter and the loader can never drift; the
//! standalone `verify_trmed003` binary re-derives everything with its
//! own independent code, per the Gate 6 S1 verifier discipline.
//!
//! Every function here is a pure rearrangement (or, for
//! `pack_f32_tiled`, a per-element dequantization) of its input:
//! values are preserved exactly, so runtime accumulations are
//! bit-identical whether the layout was built at load or baked into
//! the archive.

/// GEMV layout: the same 16-byte chunks as `pack_sdot`
/// (columns 4cg..4cg+4 at depths 4jb..4jb+4, lane = column,
/// byte = depth) but stored column-group-major: chunk
/// (cg, jb) sits at ((cg * (k/4)) + jb) * 16, so one output
/// column group's whole depth is contiguous.
pub fn pack_gemv(weights: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut out = vec![0i8; k * n];
    for j in 0..k {
        for c in 0..n {
            out[(((c / 4) * (k / 4) + (j / 4)) * 16) + (c % 4) * 4 + (j % 4)] =
                weights[j * n + c];
        }
    }
    out
}

/// Tile layout for the aarch64 SDOT projection kernel: element (j, c)
/// of the natural [k][n] matrix lives at
/// `((j / 4) * (n / 4) + (c / 4)) * 16 + (c % 4) * 4 + (j % 4)`, so the
/// 16 bytes at a tile start are exactly the B operand `vdotq_s32`
/// consumes: B[4l + t] = w[j0 + t][c0 + l] against an A operand whose
/// every 4-byte group is [a[j0], a[j0+1], a[j0+2], a[j0+3]].
pub fn pack_sdot(weights: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut out = vec![0i8; k * n];
    for j in 0..k {
        for c in 0..n {
            out[((j / 4) * (n / 4) + (c / 4)) * 16 + (c % 4) * 4 + (j % 4)] = weights[j * n + c];
        }
    }
    out
}

/// Tile layout for the aarch64 SMMLA (i8mm) projection kernel:
/// a pure rearrangement of the natural [k][n] weights (values are
/// already the signed code-128 codes the accumulation consumes).
/// For an 8-column block `cb` and an 8-deep k block `jb`, the four
/// 16-byte chunks at `((cb * (k/8) + jb) * 4 + p) * 16` hold, for
/// column pair `p`, first column `cb*8 + 2p`'s codes for
/// k = jb*8..jb*8+8, then column `cb*8 + 2p + 1`'s — exactly the
/// B operand `smmla` consumes against an A operand of two rows'
/// matching 8-byte k runs.
pub fn pack_mmla(weights: &[i8], k: usize, n: usize) -> Vec<i8> {
    let mut out = vec![0i8; k * n];
    for cb in 0..n / 8 {
        for jb in 0..k / 8 {
            for p in 0..4 {
                let dst = ((cb * (k / 8) + jb) * 4 + p) * 16;
                for half in 0..2 {
                    let c = cb * 8 + p * 2 + half;
                    for kk in 0..8 {
                        out[dst + half * 8 + kk] = weights[(jb * 8 + kk) * n + c];
                    }
                }
            }
        }
    }
    out
}

/// medium.cpp `embedding_code`: the raw uint8 affine code of (token,
/// feature) in the packed64 layout, or the plain [token][feature]
/// layout when `packed` is false.
pub fn embedding_code(codes: &[u8], packed: bool, token: usize, feature: usize) -> u8 {
    if !packed {
        return codes[token * 640 + feature];
    }
    let offset = (token / 64) * (640 * 64)
        + (feature / 128) * (128 * 64)
        + (token % 64 / 16) * (128 * 16)
        + (feature % 128) * 16
        + token % 16;
    codes[offset]
}

/// The [32768][640] embedding codes in natural [token][feature]
/// order: a clone when the archive already stores them that way,
/// else the packed64 permutation undone element by element.
pub fn natural_embedding_codes(codes: &[u8], packed: bool) -> Vec<u8> {
    if !packed {
        return codes.to_vec();
    }
    let mut out = vec![0u8; 32768 * 640];
    for v in 0..32768usize {
        for j in 0..640usize {
            out[v * 640 + j] = embedding_code(codes, true, v, j);
        }
    }
    out
}

/// The output head's FP32 form: the [vocab][depth] embedding codes
/// dequantized as `(code - zero) * scale` and stored in the
/// 128-vocab-column tile layout the FP32 GEMM consumes:
/// element (v, j) lives at `(v & !127) * depth + j * 128 + (v & 127)`.
/// `codes` are the raw uint8 codes in natural [v][j] order.
/// This is exactly the `tiled` array `Model::build` materialized at
/// load before round 2 (and the x86 `HeadStore::Tiled` array).
pub fn pack_f32_tiled(codes: &[u8], vocab: usize, depth: usize, zero: i32, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab * depth];
    for v in 0..vocab {
        for j in 0..depth {
            out[(v & !127) * depth + j * 128 + (v & 127)] =
                (codes[v * depth + j] as i32 - zero) as f32 * scale;
        }
    }
    out
}
