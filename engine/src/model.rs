//! Model construction: parse the archive and materialize the weights the
//! way the C++ `Medium` constructor does (frontend int8 codes dequantized
//! to FP32 at load; VAD weights are FP32 in the archive; projection
//! matrices keep their archive int8 codes, un-packed from layout 8 into
//! natural [k][n] order — the integer projection math is exact and
//! order-free, so the layout change is value-preserving). The archive
//! directory itself is fully parsed and validated in archive.rs.

use crate::archive::{self, ParsedArchive};
use std::sync::{Arc, Mutex};

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

/// The owned byte source a Model was built from. For Gate 6's mmap
/// loading this is the memory-mapped model file / APK asset extent:
/// its pages are the shared OS page cache, faulted on demand. A Model
/// built with `from_backing` on a TRMED002 archive borrows its
/// projection weights directly out of this buffer (zero-copy); the
/// Arc keeps the mapping alive exactly as long as the Model.
pub struct Backing {
    kind: BackingKind,
}

enum BackingKind {
    Bytes(Vec<u8>),
    Mmap { map: memmap2::Mmap, start: usize, len: usize },
}

impl Backing {
    pub fn from_bytes(bytes: Vec<u8>) -> Backing {
        Backing { kind: BackingKind::Bytes(bytes) }
    }
    pub fn from_mmap(map: memmap2::Mmap, start: usize, len: usize) -> Backing {
        Backing { kind: BackingKind::Mmap { map, start, len } }
    }
    pub fn as_slice(&self) -> &[u8] {
        match &self.kind {
            BackingKind::Bytes(v) => v,
            BackingKind::Mmap { map, start, len } => &map[*start..*start + *len],
        }
    }
}

/// Projection weight storage. `Owned` is the Gate 1-5 path (v1 weights
/// unpacked at load, or a v2 natural-layout copy when the caller does
/// not own the archive bytes). `Borrowed` points into the Model's
/// `Backing` (v2 + mmap): the pointer is into an immutable buffer the
/// Model keeps alive via Arc, is never mutated, and is only read while
/// the Model lives -- hence the manual Send/Sync.
pub enum WeightStore {
    Owned(Vec<i8>),
    Borrowed { ptr: *const i8, len: usize },
}

unsafe impl Send for WeightStore {}
unsafe impl Sync for WeightStore {}

impl WeightStore {
    pub fn as_slice(&self) -> &[i8] {
        match self {
            WeightStore::Owned(v) => v,
            WeightStore::Borrowed { ptr, len } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub fn as_ptr(&self) -> *const i8 {
        self.as_slice().as_ptr()
    }
}

/// One quantized projection (medium.cpp `Matrix` / projection.h): archive
/// int8 weight codes in natural [k][n] order (signed values, i.e. the
/// stored code is the uint8 code minus 128), the archive's per-column
/// int32 sums of those signed codes, per-column FP32 scales, per-column
/// weight zero points pre-centered (zero - 128), and FP32 biases.
pub struct Matrix {
    pub k: usize,
    pub n: usize,
    pub weights: WeightStore, // [k][n]
    /// aarch64 SDOT tile layout of `weights` (see
    /// `crate::layout::pack_sdot`); built at load on aarch64 only,
    /// or borrowed zero-copy from a TRMED003 archive; empty elsewhere.
    pub packed_sdot: WeightStore,
    /// aarch64 SMMLA tile layout of `weights` (see
    /// `crate::layout::pack_mmla`); same provenance as `packed_sdot`.
    pub packed_mmla: WeightStore,
    /// aarch64 GEMV layout of `weights` (see `crate::layout::pack_gemv`):
    /// column-group-major, so a single row's projection streams
    /// each output column group's codes contiguously; same
    /// provenance as `packed_sdot`.
    pub packed_gemv: WeightStore,
    pub sums: Vec<i32>,       // [n]
    pub scales: Vec<f32>,     // [n]
    pub zeros: Vec<i32>,      // [n], zero point - 128
    pub bias: Vec<f32>,       // [n]
    /// Round-4 int4 twin (GEMV path only); built at load for the
    /// decoder matrices when TS_DEC_INT4 is on. See Int4Matrix.
    pub int4: Option<Int4Matrix>,
}

impl Matrix {
    pub fn weights(&self) -> &[i8] {
        self.weights.as_slice()
    }
}

/// Round-4 int4 engagement (speed program): TS_DEC_INT4 set and
/// not "0". Default OFF; adoption is a separate gated ruling.
pub fn dec_int4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("TS_DEC_INT4").map(|v| v != "0").unwrap_or(false)
    })
}

/// Group size (consecutive k values per scale) for int4 weights.
pub const INT4_GROUP: usize = 64;

/// Int4 twin of a Matrix for the rows==1 GEMV path: the
/// dequantized weights re-quantized symmetrically to 4 bits per
/// (group of INT4_GROUP consecutive k, column), packed as nibbles
/// in the flat operand order the int4 GEMV kernel consumes
/// (block b = column-group * (k/4) + depth-block; within a block,
/// t = dc*4 + lane maps to (j = (b%(k/4))*4 + lane, c = cg*4+dc);
/// flat position p -> byte p/2, low nibble if p even).
pub struct Int4Matrix {
    pub k: usize,
    pub n: usize,
    pub group: usize,
    /// Nibbles, length k*n/2 + 16 pad bytes (kernel over-read).
    pub packed: Vec<u8>,
    /// Per-(group, column) scales, [g][c] group-major.
    pub gscales: Vec<f32>,
    /// colsums[c] = sum_j W4[j][c] (dequantized int4 weights).
    pub colsums: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Int4Matrix {
    pub fn from_matrix(m: &Matrix) -> Int4Matrix {
        let (k, n, g) = (m.k, m.n, INT4_GROUP);
        assert!(k % g == 0 && n % 4 == 0 && k % 4 == 0);
        let ngroups = k / g;
        let w = m.weights();
        let mut gscales = vec![0.0f32; ngroups * n];
        let mut colsums = vec![0.0f32; n];
        // q4 codes in natural [j][c] order first.
        let mut q4 = vec![0i8; k * n];
        for c in 0..n {
            let sc = m.scales[c];
            let zc = m.zeros[c];
            for gi in 0..ngroups {
                let mut mx = 0.0f32;
                for j in gi * g..(gi + 1) * g {
                    let wv = (w[j * n + c] as i32 - zc) as f32 * sc;
                    if wv.abs() > mx {
                        mx = wv.abs();
                    }
                }
                let sg = if mx > 0.0 { mx / 7.0 } else { 1.0 };
                gscales[gi * n + c] = sg;
                let mut qsum = 0i32;
                for j in gi * g..(gi + 1) * g {
                    let wv = (w[j * n + c] as i32 - zc) as f32 * sc;
                    let q = (wv / sg).round_ties_even().clamp(-8.0, 7.0) as i8;
                    q4[j * n + c] = q;
                    qsum += q as i32;
                }
                colsums[c] += sg * qsum as f32;
            }
        }
        // Pack into the kernel's flat operand order.
        let total = k * n;
        let mut packed = vec![0u8; total / 2 + 16];
        let blocks_per_cg = k / 4;
        for b in 0..(n / 4) * blocks_per_cg {
            let cg = b / blocks_per_cg;
            let jb = b % blocks_per_cg;
            for t in 0..16usize {
                let dc = t / 4;
                let lane = t % 4;
                let j = jb * 4 + lane;
                let c = cg * 4 + dc;
                let u = (q4[j * n + c] + 8) as u8;
                let pos = b * 16 + t;
                if pos % 2 == 0 {
                    packed[pos / 2] |= u;
                } else {
                    packed[pos / 2] |= u << 4;
                }
            }
        }
        Int4Matrix { k, n, group: g, packed, gscales, colsums, bias: m.bias.clone() }
    }

    /// Dequantized int4 weight at (j, c); test/reference use.
    pub fn weight_at(&self, j: usize, c: usize) -> f32 {
        let gi = j / self.group;
        let cg = c / 4;
        let dc = c % 4;
        let jb = j / 4;
        let lane = j % 4;
        let pos = (cg * (self.k / 4) + jb) * 16 + dc * 4 + lane;
        let byte = self.packed[pos / 2];
        let u = if pos % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        (u as i32 - 8) as f32 * self.gscales[gi * self.n + c]
    }
}

/// Owned-or-borrowed byte storage (the head's natural-order embedding
/// codes); same lifetime contract as `WeightStore`.
pub enum ByteStore {
    Owned(Vec<u8>),
    Borrowed { ptr: *const u8, len: usize },
}

unsafe impl Send for ByteStore {}
unsafe impl Sync for ByteStore {}

impl ByteStore {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ByteStore::Owned(v) => v,
            ByteStore::Borrowed { ptr, len } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

/// Owned-or-borrowed f32 storage (the head's dequantized FP32 tiles);
/// same lifetime contract as `WeightStore`.
pub enum F32Store {
    Owned(Vec<f32>),
    Borrowed { ptr: *const f32, len: usize },
}

unsafe impl Send for F32Store {}
unsafe impl Sync for F32Store {}

impl F32Store {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            F32Store::Owned(v) => v,
            F32Store::Borrowed { ptr, len } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

/// The kernel-ready layouts now live in `crate::layout` (single-
/// sourced with the TRMED003 converter); re-exported here so the
/// Gate 1-6 call sites keep their paths.
pub use crate::layout::{pack_gemv, pack_mmla, pack_sdot};

fn load_matrix(
    parsed: &ParsedArchive,
    name: &str,
    k: usize,
    n: usize,
    borrow_base: Option<*const u8>,
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
    let weights = if parsed.natural_weights {
        match borrow_base {
            // TRMED002 + owned backing: borrow the natural-layout
            // weights straight out of the mapping (zero-copy).
            Some(base) => WeightStore::Borrowed {
                ptr: unsafe { base.add(w.offset) } as *const i8,
                len: k * n,
            },
            // TRMED002 without an owned backing: the stored order is
            // already natural, so this is a plain copy.
            None => WeightStore::Owned(packed),
        }
    } else {
        // Un-pack layout 8 (projection.h): the packed offset of (j, c).
        let mut weights = vec![0i8; k * n];
        for c in 0..n {
            for j in 0..k {
                let off = (c / 16) * k * 16 + (j / 8) * 128 + (c % 16 / 2) * 16 + (c % 2) * 8 + j % 8;
                weights[j * n + c] = packed[off];
            }
        }
        WeightStore::Owned(weights)
    };
    // Kernel-ready layouts: borrowed zero-copy from a TRMED003
    // archive when it carries them (they are the exact bytes the
    // pack_* permutations below produce), else packed at load on
    // aarch64 (the TRMED001/002 path), else empty. A derived tensor
    // that is present but mis-shaped is an archive error, not a
    // fallback trigger.
    let derived = |suffix: &str| -> Result<Option<WeightStore>, String> {
        if !parsed.derived_layouts {
            return Ok(None);
        }
        let Some(t) = parsed.tensors.get(&format!("{name}/{suffix}")) else {
            return Ok(None);
        };
        if t.ty != 3 || t.shape != [k, n] || t.bytes != k * n {
            return Err(format!("bad derived layout tensor {name}/{suffix}"));
        }
        let Some(base) = borrow_base else {
            return Ok(None);
        };
        Ok(Some(WeightStore::Borrowed {
            ptr: unsafe { base.add(t.offset) } as *const i8,
            len: k * n,
        }))
    };
    let packed_sdot = match derived("weight_sdot")? {
        Some(s) => s,
        None => {
            #[cfg(target_arch = "aarch64")]
            {
                if k % 4 == 0 && n % 4 == 0 {
                    WeightStore::Owned(pack_sdot(weights.as_slice(), k, n))
                } else {
                    WeightStore::Owned(Vec::new())
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                WeightStore::Owned(Vec::new())
            }
        }
    };
    let packed_mmla = match derived("weight_mmla")? {
        Some(s) => s,
        None => {
            #[cfg(target_arch = "aarch64")]
            {
                if k % 8 == 0 && n % 8 == 0 {
                    WeightStore::Owned(pack_mmla(weights.as_slice(), k, n))
                } else {
                    WeightStore::Owned(Vec::new())
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                WeightStore::Owned(Vec::new())
            }
        }
    };
    let packed_gemv = match derived("weight_gemv")? {
        Some(s) => s,
        None => {
            #[cfg(target_arch = "aarch64")]
            {
                if k % 4 == 0 && n % 4 == 0 {
                    WeightStore::Owned(pack_gemv(weights.as_slice(), k, n))
                } else {
                    WeightStore::Owned(Vec::new())
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                WeightStore::Owned(Vec::new())
            }
        }
    };
    Ok(Matrix {
        k,
        n,
        weights,
        packed_sdot,
        packed_mmla,
        packed_gemv,
        sums,
        scales,
        zeros: zero_u8.iter().map(|&z| z as i32 - 128).collect(),
        bias,
        int4: None,
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
    pub head: HeadStore,
    /// Speed round 3 (lever 8): the head as a projection
    /// Matrix over its own int8 codes ([640][32768], tied
    /// embedding), for the env-gated int8 head path in the
    /// decoder. None when the head store carries no codes.
    pub head_matrix: Option<Matrix>,
    pub mutex: Mutex<()>,
    /// The archive backing, kept alive while any matrix borrows from
    /// it (v2 zero-copy path). None for the copy-based constructors.
    pub _backing: Option<Arc<Backing>>,
}

/// The output head. On x86 the C++ materializes an FP32 matrix in
/// 128-vocab-column tiles and the Gate 4 numerics are defined against
/// it, so `Tiled` keeps that exact form. On aarch64 the C++ uses its
/// CompactHead over the uint8 codes instead; `Codes` mirrors it: the
/// embedding codes re-laid in natural [32768][640] order (21 MB), with
/// dequantization at use -- no 84 MB FP32 expansion on device.
pub enum HeadStore {
    Tiled(F32Store),
    Codes {
        codes: ByteStore,
        scale: f32,
        zero: i32,
        /// FP32 dequantized head in the same 128-column tile
        /// layout as `Tiled`, for multi-row (bulk) calls, which
        /// run as an FP32 GEMM like the C++ oracle's bulk head.
        /// Borrowed from a TRMED003 archive when available;
        /// an empty Owned store when unavailable.
        tiled: F32Store,
    },
}

/// Kept at its historical path; the definition lives in
/// `crate::layout` (single-sourced with the TRMED003 converter).
pub use crate::layout::embedding_code;

/// The embedding codes in natural [token][feature] order
/// (delegates to `crate::layout::natural_embedding_codes`).
fn natural_codes(codes: &[u8], packed: bool) -> Vec<u8> {
    crate::layout::natural_embedding_codes(codes, packed)
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

/// Build the head's projection Matrix from its u8 codes
/// ([vocab][640], tied embedding): weights stored [j][v] as
/// code - 128, per-column scale replicated, zeros = zero - 128,
/// so project_s's epilogue reproduces
/// scale * sum_j (a_j - za) * (code_j - zero) exactly in i32
/// (the only deviation from the f32 head is the activation
/// quantization of the normalized row, as in every encoder
/// projection).
fn build_head_matrix(codes: &[u8], scale: f32, zero: i32) -> Matrix {
    let (k, n) = (640usize, 32768usize);
    let mut w = vec![0i8; k * n];
    let mut sums = vec![0i32; n];
    for v in 0..n {
        let row = &codes[v * k..(v + 1) * k];
        let mut sm = 0i32;
        for j in 0..k {
            let c = (row[j] as i32 - 128) as i8;
            w[j * n + v] = c;
            sm += c as i32;
        }
        sums[v] = sm;
    }
    let packed_mmla = {
        #[cfg(target_arch = "aarch64")]
        {
            WeightStore::Owned(pack_mmla(&w, k, n))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            WeightStore::Owned(Vec::new())
        }
    };
    let packed_gemv = {
        #[cfg(target_arch = "aarch64")]
        {
            WeightStore::Owned(pack_gemv(&w, k, n))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            WeightStore::Owned(Vec::new())
        }
    };
    Matrix {
        k,
        n,
        weights: WeightStore::Owned(w),
        packed_sdot: WeightStore::Owned(Vec::new()),
        packed_mmla,
        packed_gemv,
        sums,
        scales: vec![scale; n],
        zeros: vec![zero - 128; n],
        bias: vec![0.0; n],
        int4: None,
    }
}

impl Model {
    pub fn from_bytes(data: &[u8]) -> Result<Model, String> {
        let parsed = archive::parse(data)?;
        Model::build(&parsed, None, None)
    }

    /// Build over an owned backing (mmap of the model file or asset
    /// extent). TRMED002 weights are borrowed zero-copy from the
    /// backing; TRMED001 falls back to the unpack-at-load path.
    pub fn from_backing(backing: Arc<Backing>) -> Result<Model, String> {
        // The parsed view borrows the backing; build() copies or
        // raw-pointer-borrows everything it needs, so the view is
        // dropped before the Arc moves into the Model (which then
        // keeps the buffer alive for the borrowed weights).
        let mut model = {
            let parsed = archive::parse(backing.as_slice())?;
            let base = if parsed.natural_weights {
                Some(backing.as_slice().as_ptr())
            } else {
                None
            };
            Model::build(&parsed, base, None)?
        };
        model._backing = Some(backing);
        Ok(model)
    }

    fn build(
        parsed: &ParsedArchive,
        borrow_base: Option<*const u8>,
        backing: Option<Arc<Backing>>,
    ) -> Result<Model, String> {
        let parsed = parsed;

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
        let conv1 = dequant_per_output(parsed, "frontend/conv1", 1536, 768 * 5, &[1536, 768, 5])?;
        let conv2 = dequant_per_output(parsed, "frontend/conv2", 768, 1536 * 5, &[768, 1536, 5])?;
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
                    parsed,
                    &format!("encoder/linear/{}", layer * 6 + i),
                    ek[i],
                    en[i],
                    borrow_base,
                )?);
            }
        }
        let mut cross_matrices = Vec::with_capacity(28);
        for i in 0..28 {
            cross_matrices.push(load_matrix(parsed, &format!("cross/{i}"), 640, 640, borrow_base)?);
        }
        let adapter_matrix = load_matrix(parsed, "adapter/projection", 768, 640, borrow_base)?;
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
                    parsed,
                    &format!("decoder/linear/{}", layer * 8 + i),
                    dk[i],
                    dn[i],
                    borrow_base,
                )?);
            }
        }
        // Round 4: int4 twins for the decoder GEMV path (env-gated).
        if dec_int4_enabled() {
            for m in decoder_matrices.iter_mut() {
                if m.k % INT4_GROUP == 0 && m.n % 4 == 0 {
                    m.int4 = Some(Int4Matrix::from_matrix(m));
                }
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
        let (embedding_codes, embedding_packed) = if parsed.derived_layouts
            && parsed.tensors.contains_key("decoder/embedding/quantized")
        {
            // TRMED003 carries the natural-order codes as a derived
            // tensor alongside packed64; prefer it (the head borrows
            // it verbatim below when the backing allows).
            let t = parsed.expect("decoder/embedding/quantized", 2, &[32768, 640])?;
            (parsed.u8_vec(t), false)
        } else if parsed.tensors.contains_key("decoder/embedding/packed64") {
                let t = parsed.expect("decoder/embedding/packed64", 2, &[512, 5, 512, 16])?;
                (parsed.u8_vec(t), true)
            } else {
                let t = parsed.expect("decoder/embedding/quantized", 2, &[32768, 640])?;
                (parsed.u8_vec(t), false)
            };
        // The embedding codes in natural [token][feature] order,
        // computed at most once and only if a fallback needs them
        // (TRMED003 borrows both head stores from the archive).
        let mut nat_codes: Option<Vec<u8>> = None;
        let head = if cfg!(target_arch = "aarch64") {
            // CompactHead form: natural-order uint8 codes, dequantized
            // at use (decoder::head_block). On TRMED003 both the codes
            // and the FP32 tiles are borrowed zero-copy from the
            // archive's derived tensors; otherwise they are
            // materialized here exactly as before.
            let tiled: F32Store = if parsed.derived_layouts {
                match (borrow_base, parsed.tensors.get("decoder/head_tiled")) {
                    (Some(base), Some(t)) => {
                        if t.ty != 1 || t.shape != [32768, 640] || t.bytes != 32768 * 640 * 4 {
                            return Err("bad derived tensor decoder/head_tiled".into());
                        }
                        F32Store::Borrowed {
                            ptr: unsafe { base.add(t.offset) } as *const f32,
                            len: 32768 * 640,
                        }
                    }
                    (Some(_), None) => {
                        return Err("TRMED003 archive missing decoder/head_tiled".into());
                    }
                    _ => F32Store::Owned(crate::layout::pack_f32_tiled(
                        nat_codes.get_or_insert_with(|| {
                            natural_codes(&embedding_codes, embedding_packed)
                        }),
                        32768,
                        640,
                        embedding_zero,
                        embedding_scale,
                    )),
                }
            } else {
                F32Store::Owned(crate::layout::pack_f32_tiled(
                    nat_codes.get_or_insert_with(|| {
                        natural_codes(&embedding_codes, embedding_packed)
                    }),
                    32768,
                    640,
                    embedding_zero,
                    embedding_scale,
                ))
            };
            let codes: ByteStore = if parsed.derived_layouts && !embedding_packed {
                // `embedding_codes` here IS the derived natural-order
                // tensor's bytes; borrow them from the backing.
                match borrow_base {
                    Some(base) => {
                        let t =
                            parsed.expect("decoder/embedding/quantized", 2, &[32768, 640])?;
                        ByteStore::Borrowed {
                            ptr: unsafe { base.add(t.offset) },
                            len: 32768 * 640,
                        }
                    }
                    None => ByteStore::Owned(embedding_codes.clone()),
                }
            } else if !embedding_packed {
                ByteStore::Owned(embedding_codes.clone())
            } else {
                ByteStore::Owned(
                    nat_codes
                        .get_or_insert_with(|| natural_codes(&embedding_codes, true))
                        .clone(),
                )
            };
            HeadStore::Codes { codes, scale: embedding_scale, zero: embedding_zero, tiled }
        } else {
            // x86 head: the FP32 embedding matrix materialized in
            // 128-vocab-column tiles, offset(v, j) = (v & !127) * 640 +
            // j * 128 + (v & 127), value = float(int(code) - zero) * scale.
            HeadStore::Tiled(F32Store::Owned(crate::layout::pack_f32_tiled(
                nat_codes
                    .get_or_insert_with(|| natural_codes(&embedding_codes, embedding_packed)),
                32768,
                640,
                embedding_zero,
                embedding_scale,
            )))
        };

        let head_matrix = match &head {
            HeadStore::Codes { codes, scale, zero, .. } => {
                Some(build_head_matrix(codes.as_slice(), *scale, *zero))
            }
            _ => None,
        };
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
            head_matrix,
            mutex: Mutex::new(()),
            _backing: backing,
        })
    }
}
