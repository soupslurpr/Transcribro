//! Phase 2 A-2 kernel bench (feature "proj-bench"; never in the
//! product build). Decomposes project_s at in-app shapes:
//!   hot   — project_s, same matrix (product path, fresh out alloc
//!           per call, per-call tiling)
//!   rot   — project_s, 28 distinct matrices cycled
//!   ceil  — tile once + fused kernel into a REUSED out buffer
//!   ceilT — tile per rep (reused tile buf) + kernel into reused out
//!   ceilA — tile once + kernel into a FRESH alloc_uninit out per rep
//! Each cell runs 3 trials; the best GMAC/s is printed per mode.

use crate::encoder::{project_s, quantize, Quantized};
use crate::layout::{pack_gemv, pack_mmla, pack_sdot};
use crate::model::{Matrix, WeightStore};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn i8(&mut self) -> i8 {
        (self.next() % 255) as i32 as i8 - 127
    }
    fn f32(&mut self) -> f32 {
        ((self.next() % 2001) as f32 - 1000.0) * 0.01
    }
}

fn make_matrix(k: usize, n: usize, rng: &mut Rng) -> Matrix {
    let weights: Vec<i8> = (0..k * n).map(|_| rng.i8()).collect();
    let sums: Vec<i32> = (0..n)
        .map(|c| (0..k).map(|j| weights[j * n + c] as i32).sum())
        .collect();
    let scales: Vec<f32> = (0..n).map(|c| 0.001 + 0.0001 * (c % 7) as f32).collect();
    let zeros: Vec<i32> = (0..n).map(|c| (c % 5) as i32 - 2).collect();
    let bias: Vec<f32> = (0..n).map(|c| 0.01 * (c as f32 - 2.0)).collect();
    Matrix {
        k,
        n,
        packed_sdot: WeightStore::Owned(pack_sdot(&weights, k, n)),
        packed_mmla: WeightStore::Owned(pack_mmla(&weights, k, n)),
        packed_gemv: WeightStore::Owned(pack_gemv(&weights, k, n)),
        weights: WeightStore::Owned(weights),
        sums,
        scales,
        zeros,
        bias,
        int4: None,
    }
}

fn setup(k: usize, n: usize, rows: usize, count: usize) -> (Vec<Matrix>, Quantized) {
    let mut rng = Rng(0x1234_5678_9abc_def1 ^ (k as u64) << 20 ^ (n as u64) << 40 ^ rows as u64);
    let mats: Vec<Matrix> = (0..count).map(|_| make_matrix(k, n, &mut rng)).collect();
    let x: Vec<f32> = (0..rows * k).map(|_| rng.f32()).collect();
    let q: Quantized = quantize(&x, rows, k).expect("quantize");
    (mats, q)
}

fn rate(macs: f64, reps: usize, dt: f64) -> f64 {
    macs * reps as f64 / dt / 1.0e9
}

fn bench_cell(k: usize, n: usize, rows: usize) {
    let macs = rows as f64 * k as f64 * n as f64;
    let reps = ((4.0e9 / macs) as usize).clamp(10, 2000);
    let (mats, q) = setup(k, n, rows, 28);
    let ref_out = project_s(&mats[0], &q, rows, 0);
    assert!(project_s(&mats[0], &q, rows, 0) == ref_out);

    let mut best = |f: &mut dyn FnMut() -> f32| -> f64 {
        let mut b = 0.0f64;
        for _ in 0..3 {
            // warmup
            let _ = f();
            let t0 = Instant::now();
            let mut sink = 0.0f32;
            for _ in 0..reps {
                sink += f();
            }
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(sink);
            b = b.max(rate(macs, reps, dt));
        }
        b
    };

    let hot = best(&mut || project_s(&mats[0], &q, rows, 0)[0]);
    let hot_p = best(&mut || {
        let o = project_s(&mats[0], &q, rows, 0);
        let v = o[0];
        crate::scratch::put_f32(o);
        v
    });
    let mut ri = 0usize;
    let rot = best(&mut || {
        let o = project_s(&mats[ri % 28], &q, rows, 0);
        ri += 1;
        o[0]
    });
    println!("PB k={k} n={n} rows={rows} hot={hot:.1} hotP={hot_p:.1} rot={rot:.1} reps={reps}");

    #[cfg(target_arch = "aarch64")]
    {
        let m = &mats[0];
        let mut tiled: Vec<i8> = Vec::new();
        crate::encoder::tile_activations(&q, rows, k, &mut tiled);
        let mut out = vec![0.0f32; rows * n];
        unsafe {
            crate::encoder::project_block_mmla_fused(m, &q, &tiled, rows, &mut out);
        }
        assert!(out == ref_out, "ceil output != project_s output");
        let ceil = best(&mut || {
            unsafe {
                crate::encoder::project_block_mmla_fused(m, &q, &tiled, rows, &mut out);
            }
            out[0]
        });
        let mut tiled2: Vec<i8> = Vec::new();
        let ceil_t = best(&mut || {
            crate::encoder::tile_activations(&q, rows, k, &mut tiled2);
            unsafe {
                crate::encoder::project_block_mmla_fused(m, &q, &tiled2, rows, &mut out);
            }
            out[0]
        });
        let ceil_a = best(&mut || {
            let mut o = crate::scratch::alloc_uninit_f32(rows * n);
            unsafe {
                crate::encoder::project_block_mmla_fused(m, &q, &tiled, rows, &mut o);
            }
            o[0]
        });
        println!("PB k={k} n={n} rows={rows} ceil={ceil:.1} ceilT={ceil_t:.1} ceilA={ceil_a:.1}");
    }
}


// ---------------- VAD conv bench (Phase 2 B) ----------------
// Times kernels::conv1d (V0) against a bench-local copy whose
// narrow GEMM is a NEON M=4 kernel (V1) at the six fixed VAD
// shapes; asserts bitwise equality per shape.

#[cfg(target_arch = "aarch64")]
mod vad_bench {
    use std::arch::aarch64::*;

    // NEON twin of gemm_small::<4>: vectorize across the 4 output
    // columns; each lane keeps the scalar ascending-k FMA chain.
    pub unsafe fn gemm_small4_neon(
        a: &[f32],
        t0: usize,
        t1: usize,
        k: usize,
        b: &[f32],
        c: &mut [f32],
        bias: Option<&[f32]>,
    ) {
        let init_of = |t: usize| bias.map(|bb| bb[t]).unwrap_or(0.0);
        let mut t = t0;
        while t + 8 <= t1 {
            let mut acc = [vdupq_n_f32(0.0); 8];
            for i in 0..8 {
                acc[i] = vdupq_n_f32(init_of(t + i));
            }
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            for kk in 0..k {
                let bv = vld1q_f32(bp.add(kk * 4));
                for i in 0..8 {
                    let av = *ap.add((t + i) * k + kk);
                    acc[i] = vfmaq_n_f32(acc[i], bv, av);
                }
            }
            for i in 0..8 {
                vst1q_f32(c.as_mut_ptr().add((t + i - t0) * 4), acc[i]);
            }
            t += 8;
        }
        while t < t1 {
            for j in 0..4 {
                let mut total = init_of(t);
                for kk in 0..k {
                    total = a[t * k + kk].mul_add(b[kk * 4 + j], total);
                }
                c[(t - t0) * 4 + j] = total;
            }
            t += 1;
        }
    }

    // conv1d_neon clone with gemm_small4_neon for count == 4 and
    // the stock scalar narrow path otherwise (only count==4 is
    // prototyped; other counts return None).
    pub fn conv1d_v1(
        input: &[f32],
        channels: usize,
        length: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        filters: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    ) -> Option<Vec<f32>> {
        let count = (length + 2 * pad - kernel) / stride + 1;
        if count != 4 {
            return None;
        }
        let k_total = channels * kernel;
        let mut out = vec![0.0f32; filters * count];
        let mut panel = vec![0.0f32; k_total * count];
        for cix in 0..channels {
            for kw in 0..kernel {
                let row = (cix * kernel + kw) * count;
                for t in 0..count {
                    let pos = t as isize * stride as isize + kw as isize - pad as isize;
                    if pos >= 0 && (pos as usize) < length {
                        panel[row + t] = input[cix * length + pos as usize];
                    }
                }
            }
        }
        let mut pout = vec![0.0f32; filters * count];
        unsafe {
            gemm_small4_neon(weight, 0, filters, k_total, &panel, &mut pout, bias);
        }
        for f in 0..filters {
            out[f * count..(f + 1) * count].copy_from_slice(&pout[f * count..(f + 1) * count]);
        }
        Some(out)
    }
}


// Scalar reference conv: im2col + the ORIGINAL scalar
// gemm_small accumulation (ascending-k FMA chain per output).
fn conv1d_ref(
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
    let count = (length + 2 * pad - kernel) / stride + 1;
    let k_total = channels * kernel;
    let mut panel = vec![0.0f32; k_total * count];
    for cix in 0..channels {
        for kw in 0..kernel {
            let row = (cix * kernel + kw) * count;
            for t in 0..count {
                let pos = t as isize * stride as isize + kw as isize - pad as isize;
                if pos >= 0 && (pos as usize) < length {
                    panel[row + t] = input[cix * length + pos as usize];
                }
            }
        }
    }
    let mut out = vec![0.0f32; filters * count];
    for f in 0..filters {
        for t in 0..count {
            let mut total = bias.map(|bb| bb[f]).unwrap_or(0.0);
            for kk in 0..k_total {
                total = weight[f * k_total + kk].mul_add(panel[kk * count + t], total);
            }
            out[f * count + t] = total;
        }
    }
    out
}

fn bench_vad() {
    let mut rng = Rng(0xabcdef01_23456789);
    // (channels, length, filters, kernel, stride, pad, name)
    let shapes: [(usize, usize, usize, usize, usize, usize, &str); 6] = [
        (1, 640, 258, 256, 128, 0, "stft"),
        (129, 4, 128, 3, 1, 1, "conv1"),
        (128, 4, 64, 3, 2, 1, "conv2"),
        (64, 2, 64, 3, 2, 1, "conv3"),
        (64, 1, 128, 3, 1, 1, "conv4"),
        (128, 1, 1, 1, 1, 0, "out"),
    ];
    for &(ch, len, filt, kern, stride, pad, name) in &shapes {
        let input: Vec<f32> = (0..ch * len).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..filt * ch * kern).map(|_| rng.f32() * 0.1).collect();
        let bias: Vec<f32> = (0..filt).map(|_| rng.f32() * 0.1).collect();
        let v0 = crate::kernels::conv1d(&input, ch, len, &w, Some(&bias), filt, kern, stride, pad);
        #[cfg(target_arch = "aarch64")]
        {
            let r = conv1d_ref(&input, ch, len, &w, Some(&bias), filt, kern, stride, pad);
            assert!(v0 == r, "VAD conv {name}: production != scalar reference bitwise");
        }
        let macs = (filt * ch * kern * ((len + 2 * pad - kern) / stride + 1)) as f64;
        let reps = ((2.0e8 / macs) as usize).clamp(200, 200000);
        let mut best0 = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut sink = 0.0f32;
            for _ in 0..reps {
                let o = crate::kernels::conv1d(&input, ch, len, &w, Some(&bias), filt, kern, stride, pad);
                sink += o[0];
            }
            std::hint::black_box(sink);
            best0 = best0.min(t0.elapsed().as_secs_f64());
        }
        let per0 = best0 / reps as f64 * 1.0e6;
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(v1) = vad_bench::conv1d_v1(&input, ch, len, &w, Some(&bias), filt, kern, stride, pad) {
                assert!(v0 == v1, "VAD conv {name}: V1 != V0 bitwise");
                let mut best1 = f64::MAX;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    let mut sink = 0.0f32;
                    for _ in 0..reps {
                        let o = vad_bench::conv1d_v1(&input, ch, len, &w, Some(&bias), filt, kern, stride, pad).unwrap();
                        sink += o[0];
                    }
                    std::hint::black_box(sink);
                    best1 = best1.min(t0.elapsed().as_secs_f64());
                }
                let per1 = best1 / reps as f64 * 1.0e6;
                println!("VB {name} v0={per0:.2}us v1={per1:.2}us macs={macs:.0} reps={reps}");
                continue;
            }
        }
        println!("VB {name} v0={per0:.2}us macs={macs:.0} reps={reps}");
    }
    println!("VB DONE");
}

pub fn main() {
    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    if let Ok(mask_s) = std::env::var("PB_MASK") {
        if let Ok(mask) = usize::from_str_radix(mask_s.trim_start_matches("0x"), 16) {
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut set);
                for cpu in 0..64 {
                    if mask & (1usize << cpu) != 0 {
                        libc::CPU_SET(cpu, &mut set);
                    }
                }
                let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                println!("PB affinity mask={mask:#x} rc={rc}");
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("PB_PAR").is_some() { bench_par(); return; }
    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("PB_BW").is_some() { bench_bw(); return; }
    if std::env::var_os("PB_VAD").is_some() {
        bench_vad();
        return;
    }
    let shapes: [(usize, usize); 5] = [(768, 640), (640, 640), (640, 768), (768, 3072), (3072, 768)];
    let rows_list: [usize; 4] = [32, 124, 256, 512];
    for &(k, n) in &shapes {
        for &rows in &rows_list {
            bench_cell(k, n, rows);
        }
    }
    println!("PB DONE");
}

// ---- Round 6 Phase 1 ceiling probes (bench-only, never product) ----
#[cfg(target_arch = "aarch64")]
fn pin_cpu(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(target_arch = "aarch64")]
fn run_par_off(
    m: &Matrix,
    q: &Quantized,
    tiled: &[i8],
    rows: usize,
    t: usize,
    out: &mut [f32],
) {
    let (k, n) = (m.k, m.n);
    if t == 1 {
        unsafe { crate::encoder::project_block_mmla_fused(m, q, tiled, rows, out); }
        return;
    }
    let chunk = ((rows / t) / 8).max(1) * 8;
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut r0 = 0usize;
    while r0 < rows {
        let take = chunk.min(rows - r0);
        starts.push((r0, take));
        r0 += take;
    }
    std::thread::scope(|s| {
        let mut rest_out: &mut [f32] = out;
        for (i, &(start, take)) in starts.iter().enumerate() {
            let (o_head, o_tail) = std::mem::take(&mut rest_out).split_at_mut(take * n);
            rest_out = o_tail;
            let tpart = &tiled[start * k..(start + take) * k];
            let cpu = [5usize, 6, 7, 8][i];
            s.spawn(move || {
                pin_cpu(cpu);
                unsafe {
                    crate::encoder::project_block_mmla_fused_off(m, q, tpart, take, start, o_head);
                }
            });
        }
    });
}

#[cfg(target_arch = "aarch64")]
fn bench_par_cell(k: usize, n: usize, rows: usize) {
    pin_cpu(5);
    let (mats, qfull) = setup(k, n, rows, 28);
    let macs = rows as f64 * k as f64 * n as f64;
    let reps = ((4.0e9 / macs) as usize).clamp(10, 2000);
    let m = &mats[0];
    let mut tiled_full: Vec<i8> = Vec::new();
    crate::encoder::tile_activations(&qfull, rows, k, &mut tiled_full);
    let mut out = vec![0.0f32; rows * n];
    unsafe { crate::encoder::project_block_mmla_fused(m, &qfull, &tiled_full, rows, &mut out); }
    let ref_out = out.clone();
    for t in 1..=4usize {
        for rot in [false, true] {
            let mut best_rate = 0.0f64;
            for _trial in 0..3 {
                run_par_off(m, &qfull, &tiled_full, rows, t, &mut out);
                let t0 = Instant::now();
                let mut mi = 0usize;
                for _ in 0..reps {
                    let mm = if rot { &mats[mi % 28] } else { m };
                    mi += 1;
                    run_par_off(mm, &qfull, &tiled_full, rows, t, &mut out);
                }
                let dt = t0.elapsed().as_secs_f64();
                best_rate = best_rate.max(rate(macs, reps, dt));
            }
            if !rot {
                assert!(out == ref_out, "par output mismatch");
            }
            println!("PAR k={k} n={n} rows={rows} T={t} rot={} rate={best_rate:.1}", rot as u8);
        }
    }
}

fn bench_par() {
    let shapes: [(usize, usize); 5] = [(768, 640), (640, 640), (640, 768), (768, 3072), (3072, 768)];
    for &(k, n) in &shapes {
        for &rows in &[128usize, 512] {
            bench_par_cell(k, n, rows);
        }
    }
    println!("PAR DONE");
}

#[cfg(target_arch = "aarch64")]
fn sum_range(b: &[u8]) -> u64 {
    let mut a0 = 0u64;
    let mut a1 = 0u64;
    let mut a2 = 0u64;
    let mut a3 = 0u64;
    let mut chunks = b.chunks_exact(16);
    for c in &mut chunks {
        a0 += u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64;
        a1 += u32::from_le_bytes([c[4], c[5], c[6], c[7]]) as u64;
        a2 += u32::from_le_bytes([c[8], c[9], c[10], c[11]]) as u64;
        a3 += u32::from_le_bytes([c[12], c[13], c[14], c[15]]) as u64;
    }
    for &xv in chunks.remainder() {
        a0 += xv as u64;
    }
    a0 + a1 + a2 + a3
}

#[cfg(target_arch = "aarch64")]
fn sum_mt(buf: &[u8], t: usize) -> u64 {
    if t == 1 {
        return sum_range(buf);
    }
    let chunk = buf.len() / t;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for i in 0..t {
            let part = &buf[i * chunk..if i == t - 1 { buf.len() } else { (i + 1) * chunk }];
            let cpu = [5usize, 6, 7, 8][i];
            handles.push(s.spawn(move || {
                pin_cpu(cpu);
                sum_range(part)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

#[cfg(target_arch = "aarch64")]
fn bench_bw() {
    pin_cpu(5);
    let size = 512usize << 20;
    let buf: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    for t in [1usize, 2, 3, 4] {
        let mut best = 0.0f64;
        for _trial in 0..3 {
            let passes = 4;
            let t0 = Instant::now();
            let mut total = 0u64;
            for _ in 0..passes {
                total += sum_mt(&buf, t);
            }
            std::hint::black_box(total);
            let dt = t0.elapsed().as_secs_f64();
            best = best.max(size as f64 * passes as f64 / dt / 1.0e9);
        }
        println!("BW T={t} rate={best:.1} GB/s");
    }
    println!("BW DONE");
}
