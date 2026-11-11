use crate::kernels::{conv1d, gemv_t, logistic, relu_cpp, tanh_mlas};
use crate::model::Model;
use crate::TmVadState;


/// Inner Silero VAD, a line-by-line port of `Medium::vad` in medium.cpp.
/// Exactly 512 new samples per call; recurrent state in `TmVadState`.
pub fn vad(model: &Model, audio: &[f32], state: &mut TmVadState) -> Result<f32, String> {
    if audio.len() != 512 {
        return Err("VAD frame extent".into());
    }
    let vw = &model.vad;
    let _tm = crate::prof::Timer::start();
    let mut buf = vec![0.0f32; 640];
    buf[..64].copy_from_slice(&state.context);
    buf[64..576].copy_from_slice(audio);
    for i in 0..576 {
        if !buf[i].is_finite() {
            return Err("nonfinite VAD audio".into());
        }
    }
    for i in 0..64 {
        buf[576 + i] = buf[574 - i];
    }
    _tm.stop(&crate::prof::VAD_MISC_NS);
    let _ts = crate::prof::Timer::start();
    let proj = conv1d(&buf, 1, 640, &vw.stft, None, 258, 256, 128, 0);
    debug_assert_eq!(proj.len(), 258 * 4);
    let mut hidden = vec![0.0f32; 129 * 4];
    for i in 0..129 * 4 {
        let a = proj[i];
        let b = proj[129 * 4 + i];
        hidden[i] = (a * a + b * b).sqrt();
    }
    _ts.stop(&crate::prof::VAD_STFT_NS);
    let _tc = crate::prof::Timer::start();
    let filters = [128usize, 64, 64, 128];
    let strides = [1usize, 2, 2, 1];
    let mut channels = 129usize;
    let mut length = 4usize;
    for i in 0..4 {
        let mut next = conv1d(
            &hidden,
            channels,
            length,
            &vw.conv_w[i],
            Some(&vw.conv_b[i]),
            filters[i],
            3,
            strides[i],
            1,
        );
        for v in &mut next {
            *v = relu_cpp(*v);
        }
        channels = filters[i];
        length = (length - 1) / strides[i] + 1;
        hidden = next;
    }
    if length != 1 || hidden.len() != 128 {
        return Err("VAD encoder shape".into());
    }
    _tc.stop(&crate::prof::VAD_CONVS_NS);
    let _tr = crate::prof::Timer::start();
    let mut gated = [0.0f32; 512];
    gemv_t(&mut gated, &hidden, &vw.w_ih, false);
    let hidden_state = state.hidden;
    gemv_t(&mut gated, &hidden_state, &vw.w_hh, true);
    for i in 0..512 {
        gated[i] += vw.b_ih[i] + vw.b_hh[i];
    }
    for i in 256..384 {
        gated[i] = tanh_mlas(gated[i]);
    }
    let mut act = [0.0f32; 384];
    act[..256].copy_from_slice(&gated[..256]);
    act[256..].copy_from_slice(&gated[384..]);
    for v in &mut act {
        *v = logistic(*v);
    }
    // x86 oracle path: plain mul + add (the FMA contraction is aarch64-only).
    for i in 0..128 {
        state.cell[i] = state.cell[i] * act[128 + i] + act[i] * gated[256 + i];
    }
    let mut relu_hidden = [0.0f32; 128];
    for i in 0..128 {
        state.hidden[i] = tanh_mlas(state.cell[i]) * act[256 + i];
        relu_hidden[i] = relu_cpp(state.hidden[i]);
    }
    _tr.stop(&crate::prof::VAD_RNN_NS);
    let _tm2 = crate::prof::Timer::start();
    let out = conv1d(&relu_hidden, 128, 1, &vw.out_w, Some(&vw.out_b), 1, 1, 1, 0);
    let probability = logistic(out[0]);
    state.context.copy_from_slice(&audio[448..512]);
    _tm2.stop(&crate::prof::VAD_MISC_NS);
    Ok(probability)
}
