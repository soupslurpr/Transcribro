//! TRMED002 -> TRMED003 (format v3) conversion — Gate 6 speed round 2.
//!
//! TRMED003 is TRMED002 plus derived kernel-ready layout tensors.
//! The 1,230 census tensors are carried over BYTE-IDENTICALLY (same
//! names, dtypes, shapes, payload bytes — including the natural
//! [k][n] projection `/weight` payloads); appended after them, in
//! access order, are 675–677 derived tensors whose values are pure
//! rearrangements of census values:
//!
//!   * for each of the 225 projection weights `<name>/weight`:
//!     `<name>/weight_mmla`, `<name>/weight_gemv`, `<name>/weight_sdot`
//!     (dtype 3, dims [k, n]) — exactly the bytes the engine's
//!     `layout::pack_*` permutations produce from the natural codes,
//!     which the loader otherwise recomputes at every engine create
//!     (~209 MB of codes -> ~627 MB of packed copies, ~1 s on the
//!     Pixel 8a, plus the page-fault churn of three fresh buffers);
//!   * `decoder/embedding/quantized` (dtype 2, [32768][640]) — the
//!     embedding codes in natural order, present only when the
//!     source carries them as `decoder/embedding/packed64` (the
//!     loader's head `codes` array verbatim);
//!   * `decoder/head_tiled` (dtype 1, [32768][640]) — the head's
//!     dequantized FP32 matrix in the 128-vocab-column tile layout
//!     (`layout::pack_f32_tiled`), which the loader otherwise
//!     materializes at create (84 MB).
//!
//! With TRMED003 the engine borrows all of these zero-copy from the
//! mmap instead. The layout functions are single-sourced in the
//! engine crate (`engine::layout`); the standalone `verify_trmed003`
//! binary re-derives every derived tensor with independent code and
//! checks the census against TRMED001.
//!
//! Structural rules mirror v2: header 64 B (magic `TRMED003`,
//! version 3), 160 B directory records, every payload offset
//! 4096-aligned, payloads in directory order, deterministic output.

use crate::util::sha256_hex;
use std::path::Path;

const HEADER_SIZE: u64 = 64;
const RECORD_SIZE: u64 = 160;
const PAYLOAD_ALIGN: u64 = 4096;

struct Entry {
    name: String,
    dtype: u32,
    dims: Vec<u64>,
    offset: u64,
    len: u64,
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn rd_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Access-order rank of a tensor's group (its first path segment).
fn group_rank(name: &str) -> Result<usize, String> {
    match name.split('/').next().unwrap_or("") {
        "frontend" => Ok(0),
        "vad" => Ok(1),
        "encoder" => Ok(2),
        "adapter" => Ok(3),
        "cross" => Ok(4),
        "decoder" => Ok(5),
        other => Err(format!("tensor {name} has unknown group {other:?}")),
    }
}

fn parse_v2(data: &[u8]) -> Result<Vec<Entry>, String> {
    if data.len() < HEADER_SIZE as usize {
        return Err("input is shorter than the 64 B header".into());
    }
    if &data[0..8] != b"TRMED002" {
        return Err("input is not a TRMED002 archive (bad magic)".into());
    }
    let version = rd_u32(data, 8);
    if version != 2 {
        return Err(format!("input version is {version}, expected 2"));
    }
    let count = rd_u32(data, 12) as usize;
    let total = rd_u64(data, 16);
    let header_size = rd_u64(data, 24);
    if header_size != HEADER_SIZE {
        return Err(format!("input header size is {header_size}, expected 64"));
    }
    if total != data.len() as u64 {
        return Err(format!(
            "input header total is {total} but the file is {} B",
            data.len()
        ));
    }
    let dir_end = HEADER_SIZE + RECORD_SIZE * count as u64;
    if dir_end > total {
        return Err("input directory overruns the archive".into());
    }
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let at = (HEADER_SIZE + RECORD_SIZE * i as u64) as usize;
        let raw_name = &data[at..at + 80];
        let end = raw_name.iter().position(|&b| b == 0).unwrap_or(80);
        let name = String::from_utf8(raw_name[..end].to_vec())
            .map_err(|_| format!("directory record {i} has a non-UTF-8 name"))?;
        let dtype = rd_u32(data, at + 80);
        let rank = rd_u32(data, at + 84) as usize;
        if rank > 4 {
            return Err(format!("tensor {name} has rank {rank} > 4"));
        }
        let dims: Vec<u64> = (0..rank).map(|d| rd_u64(data, at + 88 + 8 * d)).collect();
        let offset = rd_u64(data, at + 120);
        let len = rd_u64(data, at + 128);
        if offset < dir_end || offset + len > total {
            return Err(format!(
                "tensor {name} payload [{offset}, {}) overruns the archive",
                offset + len
            ));
        }
        entries.push(Entry { name, dtype, dims, offset, len });
    }
    Ok(entries)
}

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

enum Payload<'a> {
    Borrowed(&'a [u8]),
    I8(Vec<i8>),
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl<'a> Payload<'a> {
    fn bytes(&self) -> &[u8] {
        match self {
            // SAFETY: i8/f32 and u8 share size and alignment-compatible
            // byte views on every target this tool runs on (little-
            // endian hosts); lengths are element count * size.
            Payload::Borrowed(b) => b,
            Payload::I8(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len())
            },
            Payload::U8(v) => v,
            Payload::F32(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
            },
        }
    }
    fn len(&self) -> u64 {
        self.bytes().len() as u64
    }
}

struct OutEntry<'a> {
    name: String,
    dtype: u32,
    dims: Vec<u64>,
    payload: Payload<'a>,
}

/// `convert-v3 <input.trm2> <output.trm3>`: rewrite a TRMED002 archive
/// as TRMED003 (see module docs). Refuses to overwrite, like the
/// other converters.
pub fn run_convert_v3(input: &Path, output: &Path) -> Result<String, String> {
    if output.exists() {
        return Err("Refuse to overwrite an archive".to_string());
    }
    let data = std::fs::read(input).map_err(|e| e.to_string())?;
    let entries = parse_v2(&data)?;
    if entries.len() != 1230 {
        return Err(format!(
            "TRMED002 census must have 1230 tensors, found {}",
            entries.len()
        ));
    }
    let find = |name: &str| -> Option<&Entry> { entries.iter().find(|e| e.name == name) };
    // Census payloads are borrowed straight out of the input buffer
    // (never mutated; everything below only reads it).
    let payload_of = |e: &Entry| -> &[u8] {
        &data[e.offset as usize..(e.offset + e.len) as usize]
    };

    let mut out: Vec<OutEntry> = Vec::with_capacity(1230 + 677);
    for e in &entries {
        out.push(OutEntry {
            name: e.name.clone(),
            dtype: e.dtype,
            dims: e.dims.clone(),
            payload: Payload::Borrowed(payload_of(e)),
        });
    }

    // Derived layouts for the 225 projection weights.
    let mut derived: Vec<OutEntry> = Vec::with_capacity(677);
    let mut n_weights = 0usize;
    for e in &entries {
        if !crate::convert_v2::is_projection_weight(&e.name) {
            continue;
        }
        if e.dtype != 3 || e.dims.len() != 2 {
            return Err(format!(
                "projection weight {} has dtype {} / dims {:?}, expected i8 [k, n]",
                e.name, e.dtype, e.dims
            ));
        }
        let (k, n) = (e.dims[0] as usize, e.dims[1] as usize);
        let raw = payload_of(e);
        if raw.len() != k * n {
            return Err(format!("projection weight {} payload length mismatch", e.name));
        }
        // SAFETY: the payload bytes are the i8 codes themselves.
        let natural: &[i8] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        for (suffix, packed) in [
            ("weight_mmla", engine::layout::pack_mmla(natural, k, n)),
            ("weight_gemv", engine::layout::pack_gemv(natural, k, n)),
            ("weight_sdot", engine::layout::pack_sdot(natural, k, n)),
        ] {
            derived.push(OutEntry {
                name: format!("{}/{suffix}", e.name.trim_end_matches("/weight")),
                dtype: 3,
                dims: e.dims.clone(),
                payload: Payload::I8(packed),
            });
        }
        n_weights += 1;
    }
    if n_weights != 225 {
        return Err(format!("expected 225 projection weights, found {n_weights}"));
    }

    // Derived embedding tensors.
    let scale_e = find("decoder/embedding/scale")
        .ok_or("missing decoder/embedding/scale")?;
    let scale = f32::from_le_bytes(payload_of(scale_e)[..4].try_into().unwrap());
    let zero_e = find("decoder/embedding/zero_point")
        .ok_or("missing decoder/embedding/zero_point")?;
    let zero = payload_of(zero_e)[0] as i32;
    let natural_codes: Vec<u8> = if let Some(packed) = find("decoder/embedding/packed64") {
        if find("decoder/embedding/quantized").is_some() {
            return Err(
                "archive carries both packed64 and quantized embeddings; ambiguous".into(),
            );
        }
        let nat = engine::layout::natural_embedding_codes(payload_of(packed), true);
        derived.push(OutEntry {
            name: "decoder/embedding/quantized".to_string(),
            dtype: 2,
            dims: vec![32768, 640],
            payload: Payload::U8(nat.clone()),
        });
        nat
    } else if let Some(q) = find("decoder/embedding/quantized") {
        payload_of(q).to_vec()
    } else {
        return Err("archive carries no embedding codes tensor".into());
    };
    let tiled = engine::layout::pack_f32_tiled(&natural_codes, 32768, 640, zero, scale);
    derived.push(OutEntry {
        name: "decoder/head_tiled".to_string(),
        dtype: 1,
        dims: vec![32768, 640],
        payload: Payload::F32(tiled),
    });

    // Derived tensors in access order: stable sort by (group, name).
    let mut keyed: Vec<(usize, String, OutEntry)> = Vec::with_capacity(derived.len());
    for d in derived {
        keyed.push((group_rank(&d.name)?, d.name.clone(), d));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let n_derived = keyed.len();
    for (_, _, d) in keyed {
        out.push(d);
    }

    // Assign 4096-aligned offsets in directory order.
    let total_count = out.len();
    let mut offsets: Vec<u64> = Vec::with_capacity(total_count);
    let mut cursor = HEADER_SIZE + RECORD_SIZE * total_count as u64;
    for e in &out {
        cursor = align_up(cursor, PAYLOAD_ALIGN);
        offsets.push(cursor);
        cursor += e.payload.len();
    }
    let total = cursor;

    let mut records: Vec<u8> = Vec::with_capacity((RECORD_SIZE * total_count as u64) as usize);
    for (e, &offset) in out.iter().zip(&offsets) {
        if e.name.len() >= 80 {
            return Err(format!("tensor {} exceeds the 80 B name field", e.name));
        }
        let mut name_field = [0u8; 80];
        name_field[..e.name.len()].copy_from_slice(e.name.as_bytes());
        records.extend_from_slice(&name_field);
        records.extend_from_slice(&e.dtype.to_le_bytes());
        records.extend_from_slice(&(e.dims.len() as u32).to_le_bytes());
        for i in 0..4 {
            let d = if i < e.dims.len() { e.dims[i] } else { 0 };
            records.extend_from_slice(&d.to_le_bytes());
        }
        records.extend_from_slice(&offset.to_le_bytes());
        records.extend_from_slice(&e.payload.len().to_le_bytes());
        records.extend_from_slice(&[0u8; 24]);
    }

    let mut header = Vec::with_capacity(HEADER_SIZE as usize);
    header.extend_from_slice(b"TRMED003");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(total_count as u32).to_le_bytes());
    header.extend_from_slice(&total.to_le_bytes());
    header.extend_from_slice(&HEADER_SIZE.to_le_bytes());
    header.extend_from_slice(&[0u8; 32]);

    use std::io::Write;
    let mut f = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let mut pos: u64 = 0;
    let emit = |f: &mut std::fs::File, pos: &mut u64, bytes: &[u8]| -> Result<(), String> {
        f.write_all(bytes).map_err(|e| e.to_string())?;
        *pos += bytes.len() as u64;
        Ok(())
    };
    emit(&mut f, &mut pos, &header)?;
    emit(&mut f, &mut pos, &records)?;
    let zero_pad = [0u8; 4096];
    for (e, &offset) in out.iter().zip(&offsets) {
        while pos < offset {
            let chunk = ((offset - pos) as usize).min(zero_pad.len());
            emit(&mut f, &mut pos, &zero_pad[..chunk])?;
        }
        emit(&mut f, &mut pos, e.payload.bytes())?;
    }
    drop(f);
    if pos != total {
        return Err(format!("wrote {pos} B but the header promises {total} B"));
    }

    let out_bytes = std::fs::read(output).map_err(|e| e.to_string())?;
    Ok(format!(
        "{{\"format\": \"TRMED003\", \"census_tensors\": 1230, \"derived_tensors\": {}, \"bytes\": {}, \"sha256\": \"{}\"}}",
        n_derived,
        out_bytes.len(),
        sha256_hex(&out_bytes),
    ))
}
