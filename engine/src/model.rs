//! Model construction: parse the archive and materialize the weights the
//! way the C++ `Medium` constructor does (frontend int8 codes dequantized
//! to FP32 at load; VAD weights are FP32 in the archive; projection
//! matrices keep their archive int8 codes, un-packed from layout 8 into
//! natural [k][n] order — the integer projection math is exact and
//! order-free, so the layout change is value-preserving). The archive
//! directory itself is fully parsed and validated in archive.rs.

use crate::archive::{self, ParsedArchive};
use std::sync::Mutex;

pub struct FrontendWeights {
    pub linear: Vec<f32>, // [80][768]
    pub conv1: Vec<f32>,  // [1536][768][5]
    pub conv2: Vec<f32>,  // [768][1536][5]
    pub conv1_bias: Vec<f32>,
    pub conv2_bias: Vec<f32>,
}

pub struct VadWeights {
    pub stft: Vec<f32>, // [258][256]
    pub conv_w: [Vec<f32>; 4],
    pub conv_b: [Vec<f32>; 4],
    pub w_ih: Vec<f32>, // [512][128]
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>, // [512]
    pub b_hh: Vec<f32>,
    pub out_w: Vec<f32>, // [128]
    pub out_b: Vec<f32>, // [1]
}

/// One quantized projection (medium.cpp `Matrix` / projection.h): archive
/// int8 weight codes in natural [k][n] order (signed values, i.e. the
/// stored code is the uint8 code minus 128), the archive's per-column
/// int32 sums of those signed codes, per-column FP32 scales, per-column
/// weight zero points pre-centered (zero - 128), and FP32 biases.
pub struct Matrix {
    pub k: usize,
    pub n: usize,
    pub weights: Vec<i8>, // [k][n]
    pub sums: Vec<i32>,   // [n]
    pub scales: Vec<f32>, // [n]
    pub zeros: Vec<i32>,  // [n], zero point - 128
    pub bias: Vec<f32>,   // [n]
}

fn load_matrix(
    parsed: &ParsedArchive,
    name: &str,
    k: usize,
    n: usize,
) -> Result<Matrix, String> {
    let w = parsed.expect(&format!("{name}/weight"), 3, &[k, n])?;
    let packed = parsed.i8_vec(w);
    let sums_t = parsed.expect(&format!("{name}/sums"), 6, &[n])?;
    let sums = parsed.i32_vec(sums_t);
    for &s in &sums {
        if s < -128 * k as i32 || s > 127 * k as i32 {
            return Err("packed sum range".into());
        }
    }
    let scales = parsed.f32_of(&format!("{name}/scale"), &[n])?;
    let zero_t = parsed.expect(&format!("{name}/zero"), 2, &[n])?;
    let zero_u8 = parsed.u8_vec(zero_t);
    let bias = parsed.f32_of(&format!("{name}/bias"), &[n])?;
    for c in 0..n {
        if !(scales[c] > 0.0) {
            return Err("invalid projection scale or bias".into());
        }
    }
    // Un-pack layout 8 (projection.h): the packed offset of element (j, c).
    let mut weights = vec![0i8; k * n];
    for c in 0..n {
        for j in 0..k {
            let off = (c / 16) * k * 16 + (j / 8) * 128 + (c % 16 / 2) * 16 + (c % 2) * 8 + j % 8;
            weights[j * n + c] = packed[off];
        }
    }
    Ok(Matrix {
        k,
        n,
        weights,
        sums,
        scales,
        zeros: zero_u8.iter().map(|&z| z as i32 - 128).collect(),
        bias,
    })
}

pub struct Model {
    pub frontend: FrontendWeights,
    pub vad: VadWeights,
    pub encoder_matrices: Vec<Matrix>, // 84: layer * 6 + [Q, K, V, out, up, down]
    pub cross_matrices: Vec<Matrix>,   // 28: layer * 2 + [K, V]
    pub adapter_matrix: Matrix,        // 768 -> 640
    pub encoder_norms: Vec<Vec<f32>>,  // 29 x [768]
    pub adapter_positions: Vec<u8>,    // [4096][768] uint8 codes
    pub adapter_positions_scale: f32,
    pub adapter_positions_zero: i32,
    pub decoder_matrices: Vec<Matrix>, // 112: layer * 8 + [Q, K, V, out, cross Q, cross out, SwiGLU in, SwiGLU out]
    pub decoder_norms: Vec<Vec<f32>>,  // 43 x [640]
    pub rotary: Vec<f32>,              // [16]
    pub embedding_codes: Vec<u8>,      // packed64 [512][5][512][16] or quantized [32768][640]
    pub embedding_packed: bool,
    pub embedding_scale: f32,
    pub embedding_zero: i32,
    pub head: Vec<f32>, // [32768][640] FP32 in the x86 128-vocab-column tiled layout
    pub mutex: Mutex<()>,
}

/// medium.cpp `embedding_code`: the raw uint8 affine code of (token,
/// feature) in the packed64 layout, or the plain [token][feature] layout.
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

fn dequant_per_output(
    parsed: &ParsedArchive,
    name: &str,
    out_channels: usize,
    channel_size: usize,
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let q = parsed.expect(&format!("{name}/quantized"), 3, shape)?;
    let codes = parsed.i8_vec(q);
    let scale = parsed.f32_of(&format!("{name}/scale"), &[out_channels, 1, 1])?;
    let offset = parsed.f32_of(&format!("{name}/zero_point"), &[out_channels, 1, 1])?;
    let mut out = vec![0.0f32; out_channels * channel_size];
    for c in 0..out_channels {
        for i in 0..channel_size {
            out[c * channel_size + i] =
                codes[c * channel_size + i] as f32 * scale[c] + offset[c];
        }
    }
    Ok(out)
}

impl Model {
    pub fn from_bytes(data: &[u8]) -> Result<Model, String> {
        let parsed = archive::parse(data)?;

        // Frontend linear: scale/zero are [1,768] (per output column).
        let lin_q = parsed.expect("frontend/linear/quantized", 3, &[80, 768])?;
        let lin_codes = parsed.i8_vec(lin_q);
        let lin_scale = parsed.f32_of("frontend/linear/scale", &[1, 768])?;
        let lin_offset = parsed.f32_of("frontend/linear/zero_point", &[1, 768])?;
        let mut linear = vec![0.0f32; 80 * 768];
        for r in 0..80 {
            for c in 0..768 {
                linear[r * 768 + c] =
                    lin_codes[r * 768 + c] as f32 * lin_scale[c] + lin_offset[c];
            }
        }
        let conv1 = dequant_per_output(&parsed, "frontend/conv1", 1536, 768 * 5, &[1536, 768, 5])?;
        let conv2 = dequant_per_output(&parsed, "frontend/conv2", 768, 1536 * 5, &[768, 1536, 5])?;
        let conv1_bias = parsed.f32_of("frontend/conv1/bias", &[1536])?;
        let conv2_bias = parsed.f32_of("frontend/conv2/bias", &[768])?;

        // Inner VAD (pure FP32 in the archive).
        let stft = parsed.f32_of("vad/stft.forward_basis_buffer", &[258, 1, 256])?;
        let vi = [129usize, 128, 64, 64];
        let vo = [128usize, 64, 64, 128];
        let mut conv_w: [Vec<f32>; 4] = Default::default();
        let mut conv_b: [Vec<f32>; 4] = Default::default();
        for i in 0..4 {
            let prefix = format!("vad/encoder.{i}.reparam_conv.");
            conv_w[i] = parsed.f32_of(&format!("{prefix}weight"), &[vo[i], vi[i], 3])?;
            conv_b[i] = parsed.f32_of(&format!("{prefix}bias"), &[vo[i]])?;
        }
        let w_ih = parsed.f32_of("vad/decoder.rnn.weight_ih", &[512, 128])?;
        let w_hh = parsed.f32_of("vad/decoder.rnn.weight_hh", &[512, 128])?;
        let b_ih = parsed.f32_of("vad/decoder.rnn.bias_ih", &[512])?;
        let b_hh = parsed.f32_of("vad/decoder.rnn.bias_hh", &[512])?;
        let out_w_full = parsed.f32_of("vad/decoder.decoder.2.weight", &[1, 128, 1])?;
        let out_b = parsed.f32_of("vad/decoder.decoder.2.bias", &[1])?;

        // Encoder / cross / adapter projections (Gate 3). Per-layer encoder
        // (k, n) shapes are pinned by the C++ Archive::validate table.
        let ek = [768usize, 768, 768, 640, 768, 3072];
        let en = [640usize, 640, 640, 768, 3072, 768];
        let mut encoder_matrices = Vec::with_capacity(84);
        for layer in 0..14 {
            for i in 0..6 {
                encoder_matrices.push(load_matrix(
                    &parsed,
                    &format!("encoder/linear/{}", layer * 6 + i),
                    ek[i],
                    en[i],
                )?);
            }
        }
        let mut cross_matrices = Vec::with_capacity(28);
        for i in 0..28 {
            cross_matrices.push(load_matrix(&parsed, &format!("cross/{i}"), 640, 640)?);
        }
        let adapter_matrix = load_matrix(&parsed, "adapter/projection", 768, 640)?;
        let mut encoder_norms = Vec::with_capacity(29);
        for i in 0..29 {
            encoder_norms.push(parsed.f32_of(&format!("encoder/norm/{i}"), &[768])?);
        }
        let pos_t = parsed.expect("adapter/positions/quantized", 2, &[4096, 768])?;
        let adapter_positions = parsed.u8_vec(pos_t);
        let adapter_positions_scale = parsed.f32_of("adapter/positions/scale", &[1])?[0];
        if !(adapter_positions_scale > 0.0) {
            return Err("invalid embedding scale".into());
        }
        let adapter_positions_zero =
            parsed.u8_vec(parsed.expect("adapter/positions/zero_point", 2, &[1])?)[0] as i32;

        // Decoder (Gate 4). Per-slot (k, n) shapes pinned by the C++
        // Archive::validate table.
        let dk = [640usize, 640, 640, 640, 640, 640, 640, 2560];
        let dn = [640usize, 640, 640, 640, 640, 640, 5120, 640];
        let mut decoder_matrices = Vec::with_capacity(112);
        for layer in 0..14 {
            for i in 0..8 {
                decoder_matrices.push(load_matrix(
                    &parsed,
                    &format!("decoder/linear/{}", layer * 8 + i),
                    dk[i],
                    dn[i],
                )?);
            }
        }
        let mut decoder_norms = Vec::with_capacity(43);
        for i in 0..43 {
            decoder_norms.push(parsed.f32_of(&format!("decoder/norm/{i}"), &[640])?);
        }
        let rotary = parsed.f32_of("decoder/rotary", &[16])?;
        let embedding_scale = parsed.f32_of("decoder/embedding/scale", &[1])?[0];
        if !(embedding_scale > 0.0) {
            return Err("invalid embedding scale".into());
        }
        let embedding_zero =
            parsed.u8_vec(parsed.expect("decoder/embedding/zero_point", 2, &[1])?)[0] as i32;
        let (embedding_codes, embedding_packed) =
            if parsed.tensors.contains_key("decoder/embedding/packed64") {
                let t = parsed.expect("decoder/embedding/packed64", 2, &[512, 5, 512, 16])?;
                (parsed.u8_vec(t), true)
            } else {
                let t = parsed.expect("decoder/embedding/quantized", 2, &[32768, 640])?;
                (parsed.u8_vec(t), false)
            };
        // x86 head: the FP32 embedding matrix materialized in
        // 128-vocab-column tiles, offset(v, j) = (v & !127) * 640 +
        // j * 128 + (v & 127), value = float(int(code) - zero) * scale.
        let mut head = vec![0.0f32; 32768 * 640];
        for v in 0..32768usize {
            for j in 0..640usize {
                let code = embedding_code(&embedding_codes, embedding_packed, v, j);
                head[(v & !127) * 640 + j * 128 + (v & 127)] =
                    (code as i32 - embedding_zero) as f32 * embedding_scale;
            }
        }

        Ok(Model {
            frontend: FrontendWeights { linear, conv1, conv2, conv1_bias, conv2_bias },
            vad: VadWeights {
                stft,
                conv_w,
                conv_b,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                out_w: out_w_full,
                out_b,
            },
            encoder_matrices,
            cross_matrices,
            adapter_matrix,
            encoder_norms,
            adapter_positions,
            adapter_positions_scale,
            adapter_positions_zero,
            decoder_matrices,
            decoder_norms,
            rotary,
            embedding_codes,
            embedding_packed,
            embedding_scale,
            embedding_zero,
            head,
            mutex: Mutex::new(()),
        })
    }
}
