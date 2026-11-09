//! Frontend (audio -> 50 fps features), a line-by-line port of
//! `Medium::frontend` in medium.cpp (census §0.6). Chunk contract:
//! 640 <= samples <= 320,000; consumes whole 80-sample frames from the
//! concatenation of the state's carry and the new audio; at the end the
//! carry is RESET (sample_len := 0, buffer zeroed) exactly as the oracle
//! does — sub-frame leftovers are the caller's responsibility.

use crate::kernels::{conv1d, eigen_sum_80, gemm_nn, silu_inplace};
use crate::model::Model;
use crate::TmFrontendState;

pub fn frontend(
    model: &Model,
    audio: &[f32],
    state: &mut TmFrontendState,
    output: &mut [f32],
    capacity_frames: usize,
) -> Result<usize, String> {
    let samples = audio.len();
    if !(640..=320_000).contains(&samples) {
        return Err("frontend chunk extent".into());
    }
    if !(0..=79).contains(&state.sample_len)
        || state.frame_count < 0
        || state.frame_count > i64::MAX - 4096
    {
        return Err("frontend state extent".into());
    }
    let sample_len = state.sample_len as usize;
    let count = (samples + sample_len) / 80;
    let first = (count + 1) / 2;
    let second = (first + 1) / 2;
    if capacity_frames < second || output.len() < second * 768 {
        return Err("frontend output capacity".into());
    }

    let mut hidden = vec![0.0f32; count * 80];
    for i in 0..count * 80 {
        let x = if i < sample_len {
            state.sample_buffer[i]
        } else {
            audio[i - sample_len]
        };
        if !x.is_finite() {
            return Err("nonfinite audio".into());
        }
        hidden[i] = x;
    }
    for r in 0..count {
        let row: &mut [f32; 80] = (&mut hidden[r * 80..(r + 1) * 80]).try_into().unwrap();
        let mean = eigen_sum_80(row) / 80.0;
        let mut squares = [0.0f32; 80];
        for j in 0..80 {
            row[j] -= mean;
            squares[j] = row[j] * row[j];
        }
        let variance = eigen_sum_80(&squares) / 80.0;
        let stddev = (variance + 1.0e-6).sqrt();
        for j in 0..80 {
            row[j] = ((row[j] / stddev) * 0.5526678562164307f32).asinh();
        }
    }

    let fw = &model.frontend;
    let mut projected = vec![0.0f32; count * 768];
    gemm_nn(&hidden, count, 80, &fw.linear, 768, &mut projected);
    silu_inplace(&mut projected);

    let mut joined = vec![0.0f32; 768 * (count + 4)];
    for c in 0..768 {
        joined[c * (count + 4)..c * (count + 4) + 4]
            .copy_from_slice(&state.conv1[c * 4..c * 4 + 4]);
        for t in 0..count {
            joined[c * (count + 4) + 4 + t] = projected[t * 768 + c];
        }
    }
    let mut expanded = conv1d(
        &joined,
        768,
        count + 4,
        &fw.conv1,
        Some(&fw.conv1_bias),
        1536,
        5,
        2,
        0,
    );
    debug_assert_eq!(expanded.len(), 1536 * first);
    silu_inplace(&mut expanded);

    let mut hidden2 = vec![0.0f32; 1536 * (first + 4)];
    for c in 0..1536 {
        hidden2[c * (first + 4)..c * (first + 4) + 4]
            .copy_from_slice(&state.conv2[c * 4..c * 4 + 4]);
        hidden2[c * (first + 4) + 4..c * (first + 4) + 4 + first]
            .copy_from_slice(&expanded[c * first..(c + 1) * first]);
    }
    let gated = conv1d(
        &hidden2,
        1536,
        first + 4,
        &fw.conv2,
        Some(&fw.conv2_bias),
        768,
        5,
        2,
        0,
    );
    debug_assert_eq!(gated.len(), 768 * second);
    for t in 0..second {
        for c in 0..768 {
            output[t * 768 + c] = gated[c * second + t];
        }
    }

    for c in 0..768 {
        state.conv1[c * 4..c * 4 + 4]
            .copy_from_slice(&joined[c * (count + 4) + count..c * (count + 4) + count + 4]);
    }
    for c in 0..1536 {
        state.conv2[c * 4..c * 4 + 4]
            .copy_from_slice(&hidden2[c * (first + 4) + first..c * (first + 4) + first + 4]);
    }
    state.sample_buffer = [0.0; 79];
    state.sample_len = 0;
    state.frame_count += count as i64;
    Ok(second)
}
