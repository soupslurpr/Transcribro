//! TRMED001 -> TRMED002 (format v2) conversion — Gate 6, stage S1.
//!
//! Reads a TRMED001 archive and writes the same 1,230 tensors as TRMED002:
//!   * header 64 B, magic `TRMED002`, version 2, same field layout as v1;
//!   * directory records unchanged (160 B each), but emitted in ACCESS
//!     order — groups frontend, vad, encoder, adapter, cross, decoder, with
//!     v1's name-sorted order preserved inside each group (v1's directory is
//!     globally name-sorted, so a stable sort by group yields exactly that);
//!   * every payload offset is 4096-aligned (zero padding);
//!   * the 225 projection `/weight` tensors are stored PRE-UNPACKED from
//!     v1's layout-8 packing into natural [k][n] (row-major) order, using the
//!     unpack mapping documented in engine/src/model.rs `load_matrix`:
//!     the packed offset of element (j, c) is
//!       (c/16)*k*16 + (j/8)*128 + (c%16/2)*16 + (c%2)*8 + (j%8)
//!     All other tensors are copied byte-identically (same payload bytes,
//!     same dtype, same shape, same byte length — the transform is a pure
//!     permutation, so even the weight tensors keep their byte length).
//!
//! Deterministic: the same input produces the same output bytes.

use crate::util::sha256_hex;
use std::path::Path;

const HEADER_SIZE: u64 = 64;
const RECORD_SIZE: u64 = 160;
const PAYLOAD_ALIGN: u64 = 4096;

/// One parsed directory record plus its payload's location in the source.
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

/// The 225 projection weights: `*/linear/*/weight`, `cross/*/weight`, and
/// `adapter/projection/weight`. (VAD tensors end in `.weight`, not
/// `/weight`, and the frontend/embedding/positions tensors use other
/// suffixes, so no other tensor matches.)
pub fn is_projection_weight(name: &str) -> bool {
    let parts: Vec<&str> = name.split('/').collect();
    match parts.as_slice() {
        ["adapter", "projection", "weight"] => true,
        ["cross", _, "weight"] => true,
        [_, "linear", _, "weight"] => true,
        _ => false,
    }
}

/// Inverse of v1's layout-8 packing: returns the tensor's codes in natural
/// [k][n] row-major order (out[j*n + c]), where the packed offset of
/// element (j, c) is the documented mapping (see module docs).
fn unpack_layout8(packed: &[u8], k: usize, n: usize) -> Result<Vec<u8>, String> {
    if k % 8 != 0 || n % 16 != 0 {
        return Err(format!("weight shape [{k}, {n}] is not layout-8 packable"));
    }
    if packed.len() != k * n {
        return Err(format!(
            "weight payload is {} B but shape [{k}, {n}] needs {} B",
            packed.len(),
            k * n
        ));
    }
    let mut out = vec![0u8; k * n];
    for c in 0..n {
        for j in 0..k {
            let off = (c / 16) * k * 16 + (j / 8) * 128 + (c % 16 / 2) * 16 + (c % 2) * 8 + (j % 8);
            out[j * n + c] = packed[off];
        }
    }
    Ok(out)
}

fn parse_v1(data: &[u8]) -> Result<Vec<Entry>, String> {
    if data.len() < HEADER_SIZE as usize {
        return Err("input is shorter than the 64 B header".into());
    }
    if &data[0..8] != b"TRMED001" {
        return Err("input is not a TRMED001 archive (bad magic)".into());
    }
    let version = rd_u32(data, 8);
    if version != 1 {
        return Err(format!("input version is {version}, expected 1"));
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
        entries.push(Entry {
            name,
            dtype,
            dims,
            offset,
            len,
        });
    }
    Ok(entries)
}

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

/// `convert-v2 <input.trm> <output.trm2>`: rewrite a TRMED001 archive as
/// TRMED002 (see module docs). Refuses to overwrite, like `convert`.
pub fn run_convert_v2(input: &Path, output: &Path) -> Result<String, String> {
    if output.exists() {
        return Err("Refuse to overwrite an archive".to_string());
    }
    let data = std::fs::read(input).map_err(|e| e.to_string())?;
    let mut entries = parse_v1(&data)?;

    // Access order: stable sort by group rank; v1's directory is globally
    // name-sorted, so each group keeps its v1 (name-sorted) relative order.
    let mut keyed: Vec<(usize, Entry)> = Vec::with_capacity(entries.len());
    for e in entries.drain(..) {
        let rank = group_rank(&e.name)?;
        keyed.push((rank, e));
    }
    keyed.sort_by_key(|(rank, _)| *rank);
    let entries: Vec<Entry> = keyed.into_iter().map(|(_, e)| e).collect();

    // Prepare payloads (owned only for the pre-unpacked weights) and assign
    // 4096-aligned offsets in access order.
    enum Payload<'a> {
        Borrowed(&'a [u8]),
        Owned(Vec<u8>),
    }
    let mut payloads: Vec<Payload> = Vec::with_capacity(entries.len());
    let mut offsets: Vec<u64> = Vec::with_capacity(entries.len());
    let mut cursor = HEADER_SIZE + RECORD_SIZE * entries.len() as u64;
    let mut transformed = 0usize;
    for e in &entries {
        let src = &data[e.offset as usize..(e.offset + e.len) as usize];
        if is_projection_weight(&e.name) {
            if e.dtype != 3 || e.dims.len() != 2 {
                return Err(format!(
                    "projection weight {} has dtype {} / dims {:?}, expected i8 [k, n]",
                    e.name, e.dtype, e.dims
                ));
            }
            let (k, n) = (e.dims[0] as usize, e.dims[1] as usize);
            payloads.push(Payload::Owned(unpack_layout8(src, k, n)?));
            transformed += 1;
        } else {
            payloads.push(Payload::Borrowed(src));
        }
        cursor = align_up(cursor, PAYLOAD_ALIGN);
        offsets.push(cursor);
        cursor += e.len;
    }
    let total = cursor;

    // Directory records: identical fields to v1, new offsets.
    let mut records: Vec<u8> = Vec::with_capacity((RECORD_SIZE * entries.len() as u64) as usize);
    for (e, &offset) in entries.iter().zip(&offsets) {
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
        records.extend_from_slice(&e.len.to_le_bytes());
        records.extend_from_slice(&[0u8; 24]);
    }

    let mut header = Vec::with_capacity(HEADER_SIZE as usize);
    header.extend_from_slice(b"TRMED002");
    header.extend_from_slice(&2u32.to_le_bytes());
    header.extend_from_slice(&(entries.len() as u32).to_le_bytes());
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
    for (payload, &offset) in payloads.iter().zip(&offsets) {
        while pos < offset {
            let chunk = ((offset - pos) as usize).min(zero_pad.len());
            emit(&mut f, &mut pos, &zero_pad[..chunk])?;
        }
        let bytes: &[u8] = match payload {
            Payload::Borrowed(b) => b,
            Payload::Owned(v) => v,
        };
        emit(&mut f, &mut pos, bytes)?;
    }
    drop(f);
    if pos != total {
        return Err(format!("wrote {pos} B but the header promises {total} B"));
    }

    let out_bytes = std::fs::read(output).map_err(|e| e.to_string())?;
    Ok(format!(
        "{{\"format\": \"TRMED002\", \"tensors\": {}, \"transformed_weights\": {}, \"copied_identical\": {}, \"bytes\": {}, \"sha256\": \"{}\"}}",
        entries.len(),
        transformed,
        entries.len() - transformed,
        out_bytes.len(),
        sha256_hex(&out_bytes),
    ))
}
