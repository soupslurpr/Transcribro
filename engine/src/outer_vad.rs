//! Outer Silero VAD: a pure-Rust port of whisper.cpp's ggml Silero 6.2
//! model (`whisper_vad_*` in whisper.cpp), the network the C++ engine runs
//! behind `nativeVad`. It is the same network shape as the inner VAD
//! (`vad.rs`) with three deltas, all verified against the whisper.cpp /
//! ggml sources:
//!
//! * Weights live in a ggml container (`ggml-silero-v6.2.0.bin`): the STFT
//!   basis, the four encoder convs and the final conv are F16; the LSTM
//!   weights and every bias are F32.
//! * There is no 64-sample context prepend. Each 512-sample frame is
//!   reflection-padded by 64 samples on BOTH sides inside the graph
//!   (`ggml_pad_reflect_1d`), and the LSTM state persists across calls
//!   (whisper.cpp's `whisper_vad_detect_speech_no_reset`); it is zeroed
//!   only by [`OuterVad::reset`].
//! * ggml numerics: accurate sigmoid / tanh (not the MLAS polynomials the
//!   inner VAD uses), and — because `ggml_conv_1d` builds its im2col in
//!   F16 whenever the weight tensor is F16 — every conv input element is
//!   rounded through F16 before the (F32-accumulated) dot products. All
//!   five convs in this network have F16 weights, so all five round.
//!
//! Container layout (whisper.cpp `whisper_vad_init_with_params`): u32
//! magic `GGML_FILE_MAGIC`, i32 string length + model-type bytes, i32
//! major/minor/patch, i32 n_window, i32 n_context, then the hparams (i32
//! n_encoder_layers, per-layer in/out/kernel triples, i32
//! lstm_input_size, lstm_hidden_size, final_conv_in, final_conv_out),
//! then tensor records until EOF: i32 n_dims, i32 name_len, i32 ggml type
//! (0 = F32, 1 = F16), ne[i32 x n_dims] (ne[0] fastest), name bytes, then
//! the payload in ggml order (ne[0] fastest).
//!
//! Layout cancellation: ggml stores a tensor with ne[0] fastest, so the
//! payload of a tensor created as (kernel, in, out) reads in file order
//! exactly as the row-major [out][in][kernel] array the inner VAD uses —
//! the linear index of (tap, in, out) is tap + kernel*in + kernel*in*out
//! under both conventions. The same holds for the LSTM matrices (created
//! (hidden, 4*hidden), read as [gate][input]) and for the STFT basis
//! (created (256, 1, 258), read as [filter][tap]). This module therefore
//! keeps every payload in file order and indexes it exactly like
//! `vad.rs` indexes the inner VAD's archive tensors.

const GGML_FILE_MAGIC: u32 = 0x6767_6d6c;
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;

/// Decode one IEEE 754 half-precision value to f32 (round-trip exact:
/// every f16 value is representable in f32).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;
    let value = if exp == 0 {
        // Zero or subnormal: value = frac * 2^-24.
        frac as f32 * 5.960_464_5e-8
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        // Normal: (1 + frac/1024) * 2^(exp-15), computed exactly in f32.
        (1.0 + frac as f32 * (1.0 / 1024.0)) * 2f32.powi(exp - 15)
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

/// Round an f32 through f16, as ggml's F32 -> F16 im2col cast does
/// (round-to-nearest-even, the hardware conversion on both x86 and ARM).
/// Implemented via the bit path so the result does not depend on the
/// host having an f16 conversion instruction.
fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // Infinity (NaN handled above).
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00; // Overflow -> infinity.
    }
    if e <= 0 {
        // Subnormal f16 or zero.
        if e < -10 {
            return sign; // Underflow to (signed) zero.
        }
        // Restore the implicit leading 1 and shift into subnormal place,
        // rounding to nearest, ties to even.
        let m = frac | 0x0080_0000;
        let shift = 14 - e;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half;
    }
    // Normal f16: truncate the mantissa to 10 bits with round-to-nearest-even.
    let mut half = ((e as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half += 1; // May carry into the exponent (or to inf); that is correct.
    }
    sign | half as u16
}

/// The value ggml's F16 im2col actually feeds the conv: `x` rounded to
/// f16 precision and converted back to f32.
fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(x))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ggml ReLU on finite values (`x < 0 ? 0 : x`).
fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Plain 1D convolution in the inner VAD's layouts: input is
/// [channel][time], weight is [filter][channel][tap] (file order), the
/// output is [filter][time]. Zero padding of `pad` on both sides, bias
/// added once per filter at the end. When `f16_inputs` is set (the weight
/// tensor is F16, so ggml's im2col is F16) each gathered input element is
/// rounded through F16 first. Accumulation is a plain ascending f32 sum;
/// ggml's mul_mat lane order differs, which costs ~1e-7 relative — the
/// Gate 6 criterion records the probability delta, it does not require
/// bit-equality.
#[allow(clippy::too_many_arguments)]
fn conv1d(
    input: &[f32],
    channels: usize,
    length: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    filters: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    f16_inputs: bool,
) -> Vec<f32> {
    let count = (length + 2 * pad - kernel) / stride + 1;
    let mut out = vec![0.0f32; filters * count];
    for f in 0..filters {
        let w_row = &weight[f * channels * kernel..(f + 1) * channels * kernel];
        for t in 0..count {
            let mut acc = 0.0f32;
            for c in 0..channels {
                for kw in 0..kernel {
                    let pos = t as isize * stride as isize + kw as isize - pad as isize;
                    if pos >= 0 && (pos as usize) < length {
                        let mut v = input[c * length + pos as usize];
                        if f16_inputs {
                            v = round_f16(v);
                        }
                        acc += w_row[c * kernel + kw] * v;
                    }
                }
            }
            out[f * count + t] = match bias {
                Some(b) => acc + b[f],
                None => acc,
            };
        }
    }
    out
}

/// Bounds-checked little-endian cursor over the ggml file bytes.
struct Reader<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .off
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| "outer VAD: truncated ggml file".to_string())?;
        let slice = &self.data[self.off..end];
        self.off = end;
        Ok(slice)
    }

    fn i32(&mut self) -> Result<i32, String> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// One parsed tensor record, payload already converted to f32 (F16
/// payloads decode exactly; F32 payloads copy verbatim), kept in file
/// order.
struct Tensor {
    name: String,
    ggml_type: i32,
    ne: Vec<i32>,
    values: Vec<f32>,
}

/// Stateful outer Silero VAD (whisper.cpp ggml Silero 6.2). One
/// [`OuterVad::probability`] call consumes exactly one 512-sample frame;
/// the LSTM state carries across calls until [`OuterVad::reset`].
pub struct OuterVad {
    stft: Vec<f32>,           // [258][256]
    conv_w: [Vec<f32>; 4],    // [out][in][3] per layer
    conv_b: [Vec<f32>; 4],    // [out] per layer
    w_ih: Vec<f32>,           // [512][128], gate-major (i, f, g, o)
    w_hh: Vec<f32>,           // [512][128]
    b_ih: Vec<f32>,           // [512]
    b_hh: Vec<f32>,           // [512]
    out_w: Vec<f32>,          // [128]
    out_b: f32,
    hidden: [f32; 128],
    cell: [f32; 128],
}

impl OuterVad {
    /// Parse the ggml Silero container (the
    /// `ggml-silero-v6.2.0.bin` asset) from memory. The network shape is
    /// pinned to the 6.2 hparams; anything else is rejected rather than
    /// run through mismatched indexing.
    pub fn from_bytes(data: &[u8]) -> Result<OuterVad, String> {
        let mut r = Reader { data, off: 0 };
        if r.u32()? != GGML_FILE_MAGIC {
            return Err("outer VAD: bad ggml magic".into());
        }
        // Model type string and version: read for framing, not validated
        // (whisper.cpp only logs them).
        let str_len = r.i32()?;
        if str_len < 0 || str_len > 4096 {
            return Err("outer VAD: bad model type length".into());
        }
        let _model_type = r.take(str_len as usize)?;
        let _major = r.i32()?;
        let _minor = r.i32()?;
        let _patch = r.i32()?;
        let n_window = r.i32()?;
        let _n_context = r.i32()?;
        if n_window != 512 {
            return Err("outer VAD: n_window is not 512".into());
        }

        // Hparams.
        let n_layers = r.i32()?;
        if n_layers != 4 {
            return Err("outer VAD: n_encoder_layers is not 4".into());
        }
        let mut layer_in = [0i32; 4];
        let mut layer_out = [0i32; 4];
        let mut layer_kernel = [0i32; 4];
        for i in 0..4 {
            layer_in[i] = r.i32()?;
            layer_out[i] = r.i32()?;
            layer_kernel[i] = r.i32()?;
        }
        let lstm_input_size = r.i32()?;
        let lstm_hidden_size = r.i32()?;
        let final_conv_in = r.i32()?;
        let final_conv_out = r.i32()?;
        if layer_in != [129, 128, 64, 64]
            || layer_out != [128, 64, 64, 128]
            || layer_kernel != [3, 3, 3, 3]
            || lstm_input_size != 128
            || lstm_hidden_size != 128
            || final_conv_in != 128
            || final_conv_out != 1
        {
            return Err("outer VAD: unexpected Silero hparams".into());
        }

        // Tensor records until EOF (whisper.cpp's load loop; unknown or
        // duplicate names are an error there and here).
        let mut tensors: Vec<Tensor> = Vec::new();
        while r.off < data.len() {
            let n_dims = r.i32()?;
            let name_len = r.i32()?;
            let ttype = r.i32()?;
            if !(0..=4).contains(&n_dims) {
                return Err("outer VAD: bad tensor n_dims".into());
            }
            if name_len < 0 || name_len > 4096 {
                return Err("outer VAD: bad tensor name length".into());
            }
            if ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16 {
                return Err("outer VAD: unsupported ggml tensor type".into());
            }
            let mut ne = Vec::with_capacity(n_dims as usize);
            let mut count: usize = 1;
            for _ in 0..n_dims {
                let d = r.i32()?;
                if d <= 0 {
                    return Err("outer VAD: bad tensor extent".into());
                }
                count = count
                    .checked_mul(d as usize)
                    .ok_or_else(|| "outer VAD: tensor size overflow".to_string())?;
                ne.push(d);
            }
            let name_bytes = r.take(name_len as usize)?;
            let name = String::from_utf8(name_bytes.to_vec())
                .map_err(|_| "outer VAD: tensor name is not UTF-8".to_string())?;
            if tensors.iter().any(|t| t.name == name) {
                return Err("outer VAD: duplicate tensor".into());
            }
            let values = if ttype == GGML_TYPE_F32 {
                let raw = r.take(count * 4)?;
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            } else {
                let raw = r.take(count * 2)?;
                raw.chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect()
            };
            tensors.push(Tensor {
                name,
                ggml_type: ttype,
                ne,
                values,
            });
        }

        let get = |name: &str, ttype: i32, ne: &[i32]| -> Result<Vec<f32>, String> {
            let t = tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("outer VAD: missing tensor {name}"))?;
            if t.ggml_type != ttype || t.ne != ne {
                return Err(format!("outer VAD: tensor {name} shape/type mismatch"));
            }
            Ok(t.values.clone())
        };
        // The final conv weight is created by whisper.cpp as the 2-D
        // tensor [128, 1] but stored with n_dims = 1 (ne = [128]); its
        // bias is created as [1] but stored as a 0-dim scalar. Accept
        // either spelling of the same 128 / 1 elements.
        let get_final_w = |name: &str| -> Result<Vec<f32>, String> {
            let t = tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("outer VAD: missing tensor {name}"))?;
            if t.ggml_type != GGML_TYPE_F16 || !(t.ne == [128] || t.ne == [128, 1]) {
                return Err(format!("outer VAD: tensor {name} shape/type mismatch"));
            }
            Ok(t.values.clone())
        };
        let get_final_b = |name: &str| -> Result<Vec<f32>, String> {
            let t = tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("outer VAD: missing tensor {name}"))?;
            if t.ggml_type != GGML_TYPE_F32 || !(t.ne.is_empty() || t.ne == [1]) {
                return Err(format!("outer VAD: tensor {name} shape/type mismatch"));
            }
            Ok(t.values.clone())
        };

        let stft = get(
            "_model.stft.forward_basis_buffer",
            GGML_TYPE_F16,
            &[256, 1, 258],
        )?;
        let mut conv_w: [Vec<f32>; 4] = Default::default();
        let mut conv_b: [Vec<f32>; 4] = Default::default();
        for i in 0..4 {
            conv_w[i] = get(
                &format!("_model.encoder.{i}.reparam_conv.weight"),
                GGML_TYPE_F16,
                &[3, layer_in[i], layer_out[i]],
            )?;
            conv_b[i] = get(
                &format!("_model.encoder.{i}.reparam_conv.bias"),
                GGML_TYPE_F32,
                &[layer_out[i]],
            )?;
        }
        let w_ih = get("_model.decoder.rnn.weight_ih", GGML_TYPE_F32, &[128, 512])?;
        let w_hh = get("_model.decoder.rnn.weight_hh", GGML_TYPE_F32, &[128, 512])?;
        let b_ih = get("_model.decoder.rnn.bias_ih", GGML_TYPE_F32, &[512])?;
        let b_hh = get("_model.decoder.rnn.bias_hh", GGML_TYPE_F32, &[512])?;
        let out_w = get_final_w("_model.decoder.decoder.2.weight")?;
        let out_b = get_final_b("_model.decoder.decoder.2.bias")?[0];
        if tensors.len() != 15 {
            return Err("outer VAD: unexpected tensor count".into());
        }

        Ok(OuterVad {
            stft,
            conv_w,
            conv_b,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            out_w,
            out_b,
            hidden: [0.0; 128],
            cell: [0.0; 128],
        })
    }

    /// Zero the LSTM state (whisper.cpp `whisper_vad_reset_state`).
    pub fn reset(&mut self) {
        self.hidden = [0.0; 128];
        self.cell = [0.0; 128];
    }

    /// Speech probability of one 512-sample frame, advancing the LSTM
    /// state. The frame is reflection-padded by 64 samples on both
    /// sides inside the call, exactly like the ggml graph; nothing is
    /// carried between calls except the LSTM state.
    pub fn probability(&mut self, audio: &[f32]) -> Result<f32, String> {
        if audio.len() != 512 {
            return Err("VAD frame extent".into());
        }
        for &v in audio {
            if !v.is_finite() {
                return Err("nonfinite VAD audio".into());
            }
        }

        // ggml_pad_reflect_1d(frame, 64, 64): mirror without repeating
        // the edge sample — padded index 0 is frame[64], index 63 is
        // frame[1], index 576 is frame[510], index 639 is frame[447].
        // (ggml-cpu pad_reflect kernel: left[-i] = left[i],
        // right[i] = right[-i].)
        let mut buf = [0.0f32; 640];
        for (i, slot) in buf.iter_mut().enumerate() {
            let q = i as isize - 64;
            let src = if q < 0 {
                -q
            } else if q >= 512 {
                1022 - q
            } else {
                q
            };
            *slot = audio[src as usize];
        }

        // STFT: 258 filters, kernel 256, stride 128 (= lstm_input_size
        // in the ggml call), no padding, no bias -> [258][4]; then the
        // magnitude over the real/imaginary halves -> [129][4].
        let proj = conv1d(&buf, 1, 640, &self.stft, None, 258, 256, 128, 0, true);
        debug_assert_eq!(proj.len(), 258 * 4);
        let mut hidden = vec![0.0f32; 129 * 4];
        for i in 0..129 * 4 {
            let a = proj[i];
            let b = proj[129 * 4 + i];
            hidden[i] = (a * a + b * b).sqrt();
        }

        // Encoder: kernel 3, pad 1, strides 1/2/2/1, ReLU after each.
        let filters = [128usize, 64, 64, 128];
        let strides = [1usize, 2, 2, 1];
        let mut channels = 129usize;
        let mut length = 4usize;
        for i in 0..4 {
            let mut next = conv1d(
                &hidden,
                channels,
                length,
                &self.conv_w[i],
                Some(&self.conv_b[i]),
                filters[i],
                3,
                strides[i],
                1,
                true,
            );
            for v in &mut next {
                *v = relu(*v);
            }
            channels = filters[i];
            length = (length - 1) / strides[i] + 1;
            hidden = next;
        }
        if length != 1 || hidden.len() != 128 {
            return Err("VAD encoder shape".into());
        }
        let x = &hidden; // [:, :, 0] == the whole [128] at length 1.

        // LSTM step, gate blocks i/f/g/o of 128. ggml association:
        // (mul_mat(W_ih, x) + b_ih) + (mul_mat(W_hh, h) + b_hh).
        let mut gates = [0.0f32; 512];
        for g in 0..512 {
            let mut ih = 0.0f32;
            let mut hh = 0.0f32;
            for k in 0..128 {
                ih += self.w_ih[g * 128 + k] * x[k];
                hh += self.w_hh[g * 128 + k] * self.hidden[k];
            }
            gates[g] = (ih + self.b_ih[g]) + (hh + self.b_hh[g]);
        }
        for i in 0..128 {
            let in_gate = sigmoid(gates[i]);
            let forget = sigmoid(gates[128 + i]);
            let cell_gate = gates[256 + i].tanh();
            let out_gate = sigmoid(gates[384 + i]);
            self.cell[i] = forget * self.cell[i] + in_gate * cell_gate;
            self.hidden[i] = out_gate * self.cell[i].tanh();
        }

        // ReLU(h), then the final conv (kernel 128 over the channel
        // axis, i.e. a dot product), bias, sigmoid.
        let relu_hidden: Vec<f32> = self.hidden.iter().map(|&v| relu(v)).collect();
        let out = conv1d(&relu_hidden, 1, 128, &self.out_w, None, 1, 128, 1, 0, true);
        Ok(sigmoid(out[0] + self.out_b))
    }
}
