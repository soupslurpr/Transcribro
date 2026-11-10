//! Standalone TRMED001 <-> TRMED003 value-equivalence verifier
//! (Gate 6 speed round 2), in the Gate 6 S1 verifier discipline.
//!
//! Usage: verify_trmed003 <v1.trm> <v3.trm3>
//!
//! Deliberately self-contained (std only, no shared code with the
//! converters or the engine): it parses both archives from scratch
//! and checks
//!   * the v3 census block is exactly v1's 1,230 tensors (same
//!     names, dtype, shape, length), in v2 access order, with
//!     byte-identical payloads for the 1,005 non-projection tensors
//!     and layout-8-unpack-equal payloads for the 225 projection
//!     weights (recomputed by sequential scatter, the inverse
//!     traversal of any gather);
//!   * every derived tensor in v3 against an independent recompute
//!     from the census values: the three packed projection layouts
//!     (each checked by decomposing every stored byte position back
//!     to its natural (j, c) coordinates — scatter-style, sharing no
//!     code with the packers), the natural-order embedding codes
//!     (checked against v1's packed64 payload by position
//!     decomposition), and the head's FP32 tiles (bit-exact
//!     dequantization at the tile offset);
//!   * v3 structural rules: census records first (access order),
//!     derived records after (access order), every payload offset
//!     4096-aligned, payloads non-overlapping and in directory
//!     order, no trailing bytes, no unknown tensor names.
//! Prints exact counts and exits non-zero on any mismatch.

use std::collections::{BTreeMap, BTreeSet};
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

/// Natural [k][n] payload recomputed from v1's layout-8 payload by
/// walking the packed bytes in order and scattering (the packed
/// offset decomposes as
///   p = (c/16)*(128*(k/8)) + (j/8)*128 + ((c%16)/2)*16 + (c%2)*8 + (j%8)
/// read in reverse).
fn unpack_by_scatter(packed: &[u8], k: usize, n: usize) -> Vec<u8> {
    assert_eq!(packed.len(), k * n, "weight payload length vs shape");
    assert!(k % 8 == 0 && n % 16 == 0, "weight shape not layout-8 packable");
    let k_blocks = k / 8;
    let mut out = vec![0u8; packed.len()];
    for (p, &byte) in packed.iter().enumerate() {
        let j_lo = p % 8;
        let c_lo_bit = (p / 8) % 2;
        let c_mid = (p / 16) % 8;
        let j_block = (p / 128) % k_blocks;
        let c_block = p / (128 * k_blocks);
        let j = j_block * 8 + j_lo;
        let c = c_block * 16 + c_mid * 2 + c_lo_bit;
        out[j * n + c] = byte;
    }
    out
}

/// Scatter checks for the three derived packed layouts: for every
/// stored byte position, decompose the position back to the natural
/// (j, c) it must have come from and compare the byte. Returns the
/// first mismatching position, if any.
fn check_sdot(stored: &[u8], natural: &[u8], _k: usize, n: usize) -> Option<usize> {
    // position: chunk (jb, cg) of 16 B at ((jb*(n/4) + cg)*16);
    // within a chunk, byte (c%4)*4 + (j%4) is w[4jb + j%4][4cg + c%4].
    for (p, &b) in stored.iter().enumerate() {
        let chunk = p / 16;
        let rem = p % 16;
        let j = (chunk / (n / 4)) * 4 + rem % 4;
        let c = (chunk % (n / 4)) * 4 + rem / 4;
        if natural[j * n + c] != b {
            return Some(p);
        }
    }
    None
}

fn check_gemv(stored: &[u8], natural: &[u8], k: usize, n: usize) -> Option<usize> {
    // position: chunk (cg, jb) of 16 B at ((cg*(k/4) + jb)*16);
    // within-chunk byte order as in the SDOT layout.
    for (p, &b) in stored.iter().enumerate() {
        let chunk = p / 16;
        let rem = p % 16;
        let j = (chunk % (k / 4)) * 4 + rem % 4;
        let c = (chunk / (k / 4)) * 4 + rem / 4;
        if natural[j * n + c] != b {
            return Some(p);
        }
    }
    None
}

fn check_mmla(stored: &[u8], natural: &[u8], k: usize, n: usize) -> Option<usize> {
    // position: chunk ((cb*(k/8) + jb)*4 + pair) of 16 B; within a
    // chunk, byte half*8 + kk is w[jb*8 + kk][cb*8 + 2*pair + half].
    for (p, &b) in stored.iter().enumerate() {
        let q = p / 16;
        let within = p % 16;
        let kk = within % 8;
        let half = within / 8;
        let cb = q / (4 * (k / 8));
        let r = q % (4 * (k / 8));
        let jb = r / 4;
        let pair = r % 4;
        let j = jb * 8 + kk;
        let c = cb * 8 + pair * 2 + half;
        if natural[j * n + c] != b {
            return Some(p);
        }
    }
    None
}

/// packed64 position decomposition: the offset of (token, feature) is
///   (token/64)*40960 + (feature/128)*8192 + ((token%64)/16)*2048
///     + (feature%128)*16 + token%16,
/// a mixed-radix encoding; walk the stored bytes and recover
/// (token, feature) arithmetically.
fn packed64_position(p: usize) -> (usize, usize) {
    let t64 = p / 40960;
    let r1 = p % 40960;
    let f_blk = r1 / 8192;
    let r2 = r1 % 8192;
    let t16 = r2 / 2048;
    let r3 = r2 % 2048;
    let f_lo = r3 / 16;
    let t_lo = r3 % 16;
    (t64 * 64 + t16 * 16 + t_lo, f_blk * 128 + f_lo)
}

fn payload<'a>(data: &'a [u8], e: &Entry) -> &'a [u8] {
    &data[e.offset as usize..(e.offset + e.len) as usize]
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: verify_trmed003 <v1.trm> <v3.trm3>".into());
    }
    let v1_data = std::fs::read(&args[0]).map_err(|e| e.to_string())?;
    let v3_data = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    let v1 = parse(&v1_data, b"TRMED001", 1, "v1")?;
    let v3 = parse(&v3_data, b"TRMED003", 3, "v3")?;
    println!("v1: {} tensors, {} bytes", v1.len(), v1_data.len());
    println!("v3: {} tensors, {} bytes", v3.len(), v3_data.len());

    let mut failures: Vec<String> = Vec::new();
    let fail = |failures: &mut Vec<String>, msg: String| {
        if failures.len() < 12 {
            failures.push(msg);
        }
    };
    let m1: BTreeMap<&str, &Entry> = v1.iter().map(|e| (e.name.as_str(), e)).collect();
    if m1.len() != v1.len() || v1.len() != 1230 {
        fail(&mut failures, "v1 census is not 1230 unique tensors".into());
    }
    let census_names: BTreeSet<&str> = m1.keys().copied().collect();

    // ---- Structural: order, alignment, non-overlap, block split.
    let mut alignment_violations = 0usize;
    let mut order_violations = 0usize;
    let mut prev_end = 64 + 160 * v3.len() as u64;
    // Census block must be the first 1230 records in access order;
    // the derived block follows, also in access order.
    let mut census_prev: Option<(usize, &str)> = None;
    let mut derived_prev: Option<(usize, &str)> = None;
    for (i, e) in v3.iter().enumerate() {
        if e.offset % 4096 != 0 {
            alignment_violations += 1;
        }
        if e.offset < prev_end {
            order_violations += 1;
        }
        prev_end = e.offset + e.len;
        let g = group_rank(&e.name);
        if i < 1230 {
            if !census_names.contains(e.name.as_str()) {
                order_violations += 1;
            }
            match (g, census_prev) {
                (Some(g), Some((pg, pn))) if g < pg || (g == pg && e.name.as_str() <= pn) => {
                    order_violations += 1;
                }
                (None, _) => order_violations += 1,
                _ => {}
            }
            if let Some(g) = g {
                census_prev = Some((g, &e.name));
            }
        } else {
            if census_names.contains(e.name.as_str()) {
                order_violations += 1; // census name in the derived block
            }
            match (g, derived_prev) {
                (Some(g), Some((pg, pn))) if g < pg || (g == pg && e.name.as_str() <= pn) => {
                    order_violations += 1;
                }
                (None, _) => order_violations += 1,
                _ => {}
            }
            if let Some(g) = g {
                derived_prev = Some((g, &e.name));
            }
        }
    }
    if prev_end != v3_data.len() as u64 {
        fail(
            &mut failures,
            format!("v3 last payload ends at {prev_end}, file is {} B", v3_data.len()),
        );
    }

    // ---- Census value equivalence vs v1 (and natural payloads kept
    // for the derived checks).
    let v3map: BTreeMap<&str, &Entry> = v3.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut transformed_checked = 0usize;
    let mut identical_checked = 0usize;
    let mut dtype_shape_mismatches = 0usize;
    let mut payload_mismatches = 0usize;
    let mut naturals: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    for (name, a) in &m1 {
        let Some(b) = v3map.get(name) else {
            fail(&mut failures, format!("{name}: missing from v3"));
            continue;
        };
        if a.dtype != b.dtype || a.dims != b.dims || a.len != b.len {
            dtype_shape_mismatches += 1;
            fail(
                &mut failures,
                format!(
                    "{name}: dtype/shape/len differ (v1 dtype {} dims {:?} len {}; v3 dtype {} dims {:?} len {})",
                    a.dtype, a.dims, a.len, b.dtype, b.dims, b.len
                ),
            );
            continue;
        }
        let pa = payload(&v1_data, a);
        let pb = payload(&v3_data, b);
        if is_projection_weight(name) {
            transformed_checked += 1;
            if a.dims.len() != 2 {
                payload_mismatches += 1;
                fail(&mut failures, format!("{name}: weight rank != 2"));
                continue;
            }
            let expected = unpack_by_scatter(pa, a.dims[0] as usize, a.dims[1] as usize);
            if expected.as_slice() != pb {
                payload_mismatches += 1;
                fail(&mut failures, format!("{name}: census payload != layout-8 unpack"));
            }
            naturals.insert(name, expected);
        } else {
            identical_checked += 1;
            if pa != pb {
                payload_mismatches += 1;
                fail(&mut failures, format!("{name}: census payload not byte-identical"));
            }
        }
    }

    // ---- Derived tensors.
    let mut derived_checked = 0usize;
    let mut derived_mismatches = 0usize;
    let mut expected_names: BTreeSet<String> = census_names.iter().map(|s| s.to_string()).collect();
    for (name, nat) in &naturals {
        let a = m1[*name];
        let (k, n) = (a.dims[0] as usize, a.dims[1] as usize);
        let base = name.strip_suffix("/weight").unwrap();
        for (suffix, check) in [
            ("weight_mmla", check_mmla as fn(&[u8], &[u8], usize, usize) -> Option<usize>),
            ("weight_gemv", check_gemv),
            ("weight_sdot", check_sdot),
        ] {
            let dname = format!("{base}/{suffix}");
            expected_names.insert(dname.clone());
            let Some(d) = v3map.get(dname.as_str()) else {
                derived_mismatches += 1;
                fail(&mut failures, format!("{dname}: derived tensor missing"));
                continue;
            };
            if d.dtype != 3 || d.dims != a.dims || d.len != a.len {
                derived_mismatches += 1;
                fail(&mut failures, format!("{dname}: dtype/shape/len mismatch"));
                continue;
            }
            derived_checked += 1;
            if let Some(p) = check(payload(&v3_data, d), nat, k, n) {
                derived_mismatches += 1;
                fail(&mut failures, format!("{dname}: byte mismatch at position {p}"));
            }
        }
    }

    // Embedding: natural codes (vs v1 packed64, by position
    // decomposition) and the FP32 head tiles (bit-exact dequant).
    let zero = payload(&v1_data, m1["decoder/embedding/zero_point"])[0] as i32;
    let scale = f32::from_le_bytes(
        payload(&v1_data, m1["decoder/embedding/scale"])[..4].try_into().unwrap(),
    );
    let natural_codes: Vec<u8> = if let Some(packed) = m1.get("decoder/embedding/packed64") {
        let pp = payload(&v1_data, packed);
        let mut nat = vec![0u8; pp.len()];
        for (p, &b) in pp.iter().enumerate() {
            let (t, f) = packed64_position(p);
            nat[t * 640 + f] = b;
        }
        expected_names.insert("decoder/embedding/quantized".to_string());
        match v3map.get("decoder/embedding/quantized") {
            Some(d) if d.dtype == 2 && d.dims == [32768, 640] => {
                derived_checked += 1;
                if payload(&v3_data, d) != nat.as_slice() {
                    derived_mismatches += 1;
                    fail(&mut failures, "decoder/embedding/quantized: payload mismatch".into());
                }
            }
            _ => {
                derived_mismatches += 1;
                fail(&mut failures, "decoder/embedding/quantized: missing/mis-shaped".into());
            }
        }
        nat
    } else {
        payload(&v1_data, m1["decoder/embedding/quantized"]).to_vec()
    };
    expected_names.insert("decoder/head_tiled".to_string());
    match v3map.get("decoder/head_tiled") {
        Some(d) if d.dtype == 1 && d.dims == [32768, 640] && d.len == 32768 * 640 * 4 => {
            derived_checked += 1;
            let bytes = payload(&v3_data, d);
            let mut bad: Option<usize> = None;
            for v in 0..32768usize {
                for j in 0..640usize {
                    let off = ((v & !127) * 640 + j * 128 + (v & 127)) * 4;
                    let stored = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                    let expect =
                        ((natural_codes[v * 640 + j] as i32 - zero) as f32 * scale).to_bits();
                    if stored != expect {
                        bad = Some(v * 640 + j);
                        break;
                    }
                }
                if bad.is_some() {
                    break;
                }
            }
            if let Some(at) = bad {
                derived_mismatches += 1;
                fail(&mut failures, format!("decoder/head_tiled: value mismatch at element {at}"));
            }
        }
        _ => {
            derived_mismatches += 1;
            fail(&mut failures, "decoder/head_tiled: missing/mis-shaped".into());
        }
    }

    // No unknown names anywhere in v3.
    let mut unknown_names = 0usize;
    for e in &v3 {
        if !expected_names.contains(&e.name) {
            unknown_names += 1;
            fail(&mut failures, format!("{}: unknown tensor name in v3", e.name));
        }
    }
    if v3map.len() != v3.len() {
        fail(&mut failures, "duplicate tensor names in v3".into());
    }

    println!("total_v1_tensors: {}", v1.len());
    println!("total_v3_tensors: {}", v3.len());
    println!("census_weights_checked: {transformed_checked}");
    println!("census_identical_checked: {identical_checked}");
    println!("derived_checked: {derived_checked}");
    println!("dtype_shape_mismatches: {dtype_shape_mismatches}");
    println!("payload_mismatches: {payload_mismatches}");
    println!("derived_mismatches: {derived_mismatches}");
    println!("alignment_violations: {alignment_violations}");
    println!("order_violations: {order_violations}");
    println!("unknown_names: {unknown_names}");
    for f in &failures {
        println!("FAIL: {f}");
    }
    let expected_derived = 675 + 1 + usize::from(m1.contains_key("decoder/embedding/packed64"));
    let pass = failures.is_empty()
        && transformed_checked == 225
        && identical_checked == 1005
        && derived_checked == expected_derived
        && dtype_shape_mismatches == 0
        && payload_mismatches == 0
        && derived_mismatches == 0
        && alignment_violations == 0
        && order_violations == 0
        && unknown_names == 0
        && v3.len() == 1230 + expected_derived;
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
