//! Standalone TRMED001 <-> TRMED002 value-equivalence verifier (Gate 6 S1).
//!
//! Usage: verify_trmed002 <v1.trm> <v2.trm2>
//!
//! Deliberately self-contained (std only, no shared code with the
//! `convert-v2` converter): it parses both archives from scratch and checks
//!   * same name set (expected: 1,230 tensors), same dtype+shape+length
//!     per tensor;
//!   * byte-identical payloads for the 1,005 non-projection tensors;
//!   * for the 225 projection `/weight` tensors, the v2 payload equals the
//!     unpack of the v1 payload — recomputed here by walking the packed
//!     payload sequentially and scattering each byte to its natural
//!     [k][n] position (the inverse traversal of the converter's gather);
//!   * v2 structural rules: every payload offset 4096-aligned, directory in
//!     access order (frontend, vad, encoder, adapter, cross, decoder;
//!     name-ascending inside a group), payloads non-overlapping and in
//!     directory order.
//! Prints exact counts and exits non-zero on any mismatch.

use std::collections::BTreeMap;
use std::process::ExitCode;

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

fn parse(data: &[u8], magic: &[u8; 8], version: u32, label: &str) -> Result<Vec<Entry>, String> {
    if data.len() < 64 {
        return Err(format!("{label}: shorter than the 64 B header"));
    }
    if &data[0..8] != magic {
        return Err(format!("{label}: bad magic"));
    }
    if rd_u32(data, 8) != version {
        return Err(format!("{label}: version {} != {version}", rd_u32(data, 8)));
    }
    let count = rd_u32(data, 12) as usize;
    let total = rd_u64(data, 16);
    if rd_u64(data, 24) != 64 {
        return Err(format!("{label}: header size != 64"));
    }
    if total != data.len() as u64 {
        return Err(format!("{label}: header total {total} != file size {}", data.len()));
    }
    if 64 + 160 * count as u64 > total {
        return Err(format!("{label}: directory overruns the archive"));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = 64 + 160 * i;
        let raw = &data[at..at + 80];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(80);
        let name = String::from_utf8(raw[..end].to_vec())
            .map_err(|_| format!("{label}: record {i} name is not UTF-8"))?;
        let rank = rd_u32(data, at + 84) as usize;
        if rank > 4 {
            return Err(format!("{label}: tensor {name} rank {rank} > 4"));
        }
        let dims = (0..rank).map(|d| rd_u64(data, at + 88 + 8 * d)).collect();
        out.push(Entry {
            name,
            dtype: rd_u32(data, at + 80),
            dims,
            offset: rd_u64(data, at + 120),
            len: rd_u64(data, at + 128),
        });
    }
    Ok(out)
}

fn group_rank(name: &str) -> Option<usize> {
    match name.split('/').next() {
        Some("frontend") => Some(0),
        Some("vad") => Some(1),
        Some("encoder") => Some(2),
        Some("adapter") => Some(3),
        Some("cross") => Some(4),
        Some("decoder") => Some(5),
        _ => None,
    }
}

/// Same classification as the format spec, written independently:
/// `*/linear/*/weight`, `cross/*/weight`, `adapter/projection/weight`.
fn is_projection_weight(name: &str) -> bool {
    if name == "adapter/projection/weight" {
        return true;
    }
    let seg: Vec<&str> = name.split('/').collect();
    (seg.len() == 3 && seg[0] == "cross" && seg[2] == "weight")
        || (seg.len() == 4 && seg[1] == "linear" && seg[3] == "weight")
}

/// Recompute the natural [k][n] payload from v1's packed payload by
/// walking the packed bytes in order and scattering: the packed offset
/// decomposes (mixed radix) as
///   p = (c/16)*(128*(k/8)) + (j/8)*128 + ((c%16)/2)*16 + (c%2)*8 + (j%8)
/// which is the documented layout-8 mapping read in reverse.
fn unpack_by_scatter(packed: &[u8], k: usize, n: usize) -> Vec<u8> {
    assert_eq!(packed.len(), k * n, "weight payload length vs shape");
    assert!(k % 8 == 0 && n % 16 == 0, "weight shape not layout-8 packable");
    let k_blocks = k / 8;
    let mut out = vec![0u8; packed.len()];
    for (p, &byte) in packed.iter().enumerate() {
        let j_lo = p % 8; // j % 8
        let c_lo_bit = (p / 8) % 2; // c % 2
        let c_mid = (p / 16) % 8; // (c % 16) / 2
        let j_block = (p / 128) % k_blocks; // j / 8
        let c_block = p / (128 * k_blocks); // c / 16
        let j = j_block * 8 + j_lo;
        let c = c_block * 16 + c_mid * 2 + c_lo_bit;
        out[j * n + c] = byte;
    }
    out
}

fn payload<'a>(data: &'a [u8], e: &Entry) -> &'a [u8] {
    &data[e.offset as usize..(e.offset + e.len) as usize]
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: verify_trmed002 <v1.trm> <v2.trm2>".into());
    }
    let v1_data = std::fs::read(&args[0]).map_err(|e| e.to_string())?;
    let v2_data = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let v1 = parse(&v1_data, b"TRMED001", 1, "v1")?;
    let v2 = parse(&v2_data, b"TRMED002", 2, "v2")?;
    println!("v1: {} tensors, {} bytes", v1.len(), v1_data.len());
    println!("v2: {} tensors, {} bytes", v2.len(), v2_data.len());

    let mut failures: Vec<String> = Vec::new();
    let m1: BTreeMap<&str, &Entry> = v1.iter().map(|e| (e.name.as_str(), e)).collect();
    let m2: BTreeMap<&str, &Entry> = v2.iter().map(|e| (e.name.as_str(), e)).collect();
    if m1.len() != v1.len() || m2.len() != v2.len() {
        failures.push("duplicate tensor names present".into());
    }
    if m1.keys().collect::<Vec<_>>() != m2.keys().collect::<Vec<_>>() {
        failures.push("name sets differ".into());
    }

    // v2 structural checks: alignment, access order, non-overlap.
    let mut alignment_violations = 0usize;
    let mut order_violations = 0usize;
    let mut prev_end = 64 + 160 * v2.len() as u64;
    let mut prev_group = 0usize;
    let mut prev_name = "";
    for e in &v2 {
        if e.offset % 4096 != 0 {
            alignment_violations += 1;
        }
        if e.offset < prev_end {
            order_violations += 1; // overlap or out-of-order payload
        }
        prev_end = e.offset + e.len;
        match group_rank(&e.name) {
            Some(g) => {
                if g < prev_group || (g == prev_group && e.name.as_str() <= prev_name) {
                    order_violations += 1;
                }
                prev_group = g;
                prev_name = &e.name;
            }
            None => order_violations += 1,
        }
    }
    if prev_end > v2_data.len() as u64 {
        failures.push("v2 last payload overruns the file".into());
    }

    // Per-tensor value equivalence.
    let mut transformed_checked = 0usize;
    let mut identical_checked = 0usize;
    let mut dtype_shape_mismatches = 0usize;
    let mut payload_mismatches = 0usize;
    for (name, a) in &m1 {
        let Some(b) = m2.get(name) else { continue };
        if a.dtype != b.dtype || a.dims != b.dims || a.len != b.len {
            dtype_shape_mismatches += 1;
            if failures.len() < 10 {
                failures.push(format!(
                    "{name}: dtype/shape/len differ (v1 dtype {} dims {:?} len {}; v2 dtype {} dims {:?} len {})",
                    a.dtype, a.dims, a.len, b.dtype, b.dims, b.len
                ));
            }
            continue;
        }
        let pa = payload(&v1_data, a);
        let pb = payload(&v2_data, b);
        if is_projection_weight(name) {
            transformed_checked += 1;
            if a.dims.len() != 2 {
                payload_mismatches += 1;
                failures.push(format!("{name}: weight rank != 2"));
                continue;
            }
            let expected = unpack_by_scatter(pa, a.dims[0] as usize, a.dims[1] as usize);
            if expected.as_slice() != pb {
                payload_mismatches += 1;
                if failures.len() < 10 {
                    let first = expected.iter().zip(pb).position(|(x, y)| x != y).unwrap();
                    failures.push(format!("{name}: unpack mismatch at byte {first}"));
                }
            }
        } else {
            identical_checked += 1;
            if pa != pb {
                payload_mismatches += 1;
                if failures.len() < 10 {
                    failures.push(format!("{name}: payload not byte-identical"));
                }
            }
        }
    }

    println!("total_tensors: {}", v1.len());
    println!("transformed_weights_checked: {transformed_checked}");
    println!("identical_tensors_checked: {identical_checked}");
    println!("dtype_shape_mismatches: {dtype_shape_mismatches}");
    println!("payload_mismatches: {payload_mismatches}");
    println!("alignment_violations: {alignment_violations}");
    println!("order_violations: {order_violations}");
    for f in &failures {
        println!("FAIL: {f}");
    }
    let pass = failures.is_empty()
        && v1.len() == 1230
        && v2.len() == 1230
        && transformed_checked == 225
        && identical_checked == 1005
        && dtype_shape_mismatches == 0
        && payload_mismatches == 0
        && alignment_violations == 0
        && order_violations == 0;
    println!("result: {}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
