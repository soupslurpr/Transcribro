//! `import_model.py` reimplemented: import the pinned Medium values into the
//! fixed, graph-free TRMED001 tensor archive. Every transformation is
//! integer-exact or a byte copy; every assertion in the Python original has a
//! counterpart here (as a panic, like Python's AssertionError).

use crate::ort::{extract_hex_bytes, load_source, Source, TensorVal};
use crate::pyjson;
use crate::util::sha256_hex;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// SHA-256 of the extracted inner Silero VAD model, asserted in
/// import_model.py (and pinned as vad.inner_sha256 in the lock).
const VAD_SHA256: &str = "a620103ef77fbdae23c7798b9271506a48a883997e55f851d2884551e0947437";

/// Generic N-d byte permutation: out has dims `axes.map(|a| dims[a])` and
/// out[idx] = data[idx'] with idx'[axes[a]] = idx[a] — numpy `transpose`.
pub fn permute(data: &[u8], dims: &[usize], axes: &[usize]) -> Vec<u8> {
    let rank = dims.len();
    assert_eq!(axes.len(), rank);
    let out_dims: Vec<usize> = axes.iter().map(|&a| dims[a]).collect();
    let total: usize = dims.iter().product();
    assert_eq!(data.len(), total);
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * dims[i + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }
    let mut out = vec![0u8; total];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let mut src = 0usize;
        for a in 0..rank {
            let coord = rem / out_strides[a];
            rem %= out_strides[a];
            src += coord * in_strides[axes[a]];
        }
        *slot = data[src];
    }
    out
}

/// The archive-under-construction: insertion-ordered values (Python dict) and
/// insertion-ordered provenance.
struct Values {
    order: Vec<String>,
    map: HashMap<String, TensorVal>,
    provenance: Vec<(String, Value)>,
}

impl Values {
    fn new() -> Values {
        Values {
            order: Vec::new(),
            map: HashMap::new(),
            provenance: Vec::new(),
        }
    }
    fn add(&mut self, name: &str, mut data: TensorVal, source: Option<Value>) {
        assert!(!self.map.contains_key(name), "duplicate tensor {name}");
        // np.ascontiguousarray promotes 0-d arrays to shape (1,); the four
        // scalar scale/zero_point tensors reach the archive that way.
        if data.dims.is_empty() {
            data.dims = vec![1];
        }
        // Python: dtype must be one of DTYPES' numpy dtypes; floats finite.
        assert!(
            crate::ort::dtype_size(data.dtype).is_some(),
            "tensor {name} has foreign dtype {}",
            data.dtype
        );
        assert_eq!(
            data.bytes.len(),
            data.numel() * data.elem_size(),
            "tensor {name} byte length does not match its shape"
        );
        if data.dtype == 1 {
            for chunk in data.bytes.chunks_exact(4) {
                assert!(
                    f32::from_le_bytes(chunk.try_into().unwrap()).is_finite(),
                    "tensor {name} contains a non-finite float"
                );
            }
        }
        self.order.push(name.to_string());
        self.map.insert(name.to_string(), data);
        if let Some(s) = source {
            self.provenance.push((name.to_string(), s));
        }
    }
    fn pop(&mut self, name: &str) -> (TensorVal, Option<Value>) {
        let v = self.map.remove(name).unwrap_or_else(|| panic!("pop of missing tensor {name}"));
        self.order.retain(|n| n != name);
        let p = self
            .provenance
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| self.provenance.remove(i).1);
        (v, p)
    }
}

fn copy(values: &mut Values, sources: &HashMap<String, Source>, name: &str, graph: &str, key: &str) {
    let data = sources[graph].values[key].clone();
    values.add(name, data, Some(json!([graph, key])));
}

/// numpy broadcast_to(x, (n,)) for the shapes that occur: (), (1,), (n,).
fn broadcast_to_n(x: &TensorVal, n: usize) -> TensorVal {
    if x.dims == [n as i64] {
        return x.clone();
    }
    assert_eq!(x.numel(), 1, "cannot broadcast shape {:?} to ({n},)", x.dims);
    let esz = x.elem_size();
    let mut bytes = Vec::with_capacity(n * esz);
    for _ in 0..n {
        bytes.extend_from_slice(&x.bytes);
    }
    TensorVal {
        dtype: x.dtype,
        dims: vec![n as i64],
        bytes,
    }
}

fn projection(
    values: &mut Values,
    sources: &HashMap<String, Source>,
    name: &str,
    graph: &str,
    output: &str,
) {
    let source = &sources[graph];
    let node = source.producing(output);
    let (weight, scale, zero, bias) = if node.op == "DynamicQuantizeMatMul" {
        let bias = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
            Some(source.values[node.inputs[4].as_str()].clone())
        } else {
            None
        };
        (
            node.inputs[1].clone(),
            node.inputs[2].clone(),
            node.inputs[3].clone(),
            bias,
        )
    } else {
        assert!(
            node.op == "MatMulIntegerToFloat" && node.inputs.len() == 6,
            "unexpected producing op {} for {output}",
            node.op
        );
        (
            node.inputs[1].clone(),
            node.inputs[3].clone(),
            node.inputs[5].clone(),
            None,
        )
    };
    let w = &source.values[weight.as_str()];
    assert_eq!(w.dtype, 2, "projection weight {weight} is not uint8");
    assert_eq!(w.dims.len(), 2, "projection weight {weight} is not a matrix");
    let (k, n) = (w.dims[0] as usize, w.dims[1] as usize);
    assert!(k % 16 == 0 && n % 16 == 0, "projection weight {weight} shape not divisible by 16");
    let signed: Vec<u8> = w.bytes.iter().map(|b| ((*b as i16 - 128) as i8) as u8).collect();
    let packed = permute(&signed, &[k / 8, 8, n / 16, 8, 2], &[2, 0, 3, 4, 1]);
    let restored = permute(&packed, &[n / 16, k / 8, 8, 2, 8], &[1, 4, 0, 2, 3]);
    assert_eq!(restored, signed, "i8mm8 packing round-trip failed for {weight}");
    values.add(
        &format!("{name}/weight"),
        TensorVal {
            dtype: 3,
            dims: vec![k as i64, n as i64],
            bytes: packed,
        },
        Some(json!([graph, weight, "reversible i8mm8"])),
    );
    let mut sums = Vec::with_capacity(4 * n);
    for col in 0..n {
        let mut acc: i32 = 0;
        for row in 0..k {
            acc += (signed[row * n + col] as i8) as i32;
        }
        sums.extend_from_slice(&acc.to_le_bytes());
    }
    values.add(
        &format!("{name}/sums"),
        TensorVal {
            dtype: 6,
            dims: vec![n as i64],
            bytes: sums,
        },
        None,
    );
    let scale_tv = broadcast_to_n(&source.values[scale.as_str()], n);
    values.add(&format!("{name}/scale"), scale_tv, None);
    let zero_tv = broadcast_to_n(&source.values[zero.as_str()], n);
    values.add(&format!("{name}/zero"), zero_tv, None);
    let bias_tv = match bias {
        Some(b) => b,
        None => TensorVal {
            dtype: 1,
            dims: vec![n as i64],
            bytes: vec![0u8; 4 * n],
        },
    };
    values.add(&format!("{name}/bias"), bias_tv, None);
}

fn transpose_equal_u8(a: &TensorVal, b: &TensorVal) -> bool {
    if a.dims.len() != 2 || b.dims.len() != 2 {
        return false;
    }
    let (r, c) = (a.dims[0] as usize, a.dims[1] as usize);
    if b.dims != [c as i64, r as i64] {
        return false;
    }
    (0..r).all(|i| (0..c).all(|j| a.bytes[i * c + j] == b.bytes[j * r + i]))
}

pub struct ConvertArgs {
    pub models: PathBuf,
    pub output: PathBuf,
    pub vad_header: PathBuf,
    pub packed_head: bool,
}

/// Resolves the repository root the way import_model.py does from its own
/// file location; here the caller supplies it (see main.rs root resolution).
pub fn run_convert(root: &Path, args: &ConvertArgs) -> Result<String, String> {
    if args.output.exists() {
        return Err("Refuse to overwrite an archive".to_string());
    }
    let pinned: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("docs/moonshine-model.json"))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut sources: HashMap<String, Source> = HashMap::new();
    for f in pinned["files"].as_array().ok_or("pinned files not an array")? {
        let file = f["file"].as_str().unwrap();
        if !file.ends_with(".ort") {
            continue;
        }
        let data = std::fs::read(args.models.join(file)).map_err(|e| e.to_string())?;
        let key = file.strip_suffix(".ort").unwrap().to_string();
        sources.insert(key, load_source(&data, f["sha256"].as_str().unwrap(), None));
    }
    let header_text = std::fs::read_to_string(&args.vad_header).map_err(|e| e.to_string())?;
    let vad_data = extract_hex_bytes(&header_text);
    sources.insert(
        "silero-16k".to_string(),
        load_source(&vad_data, VAD_SHA256, Some("then_branch")),
    );

    let mut values = Values::new();
    for (graph, count, norms) in [("encoder", 84usize, 29usize), ("decoder_kv", 112, 43)] {
        let prefix = if graph == "encoder" { "encoder" } else { "decoder" };
        for i in 0..count {
            let output = if i == 0 { "linear".to_string() } else { format!("linear_{i}") };
            projection(&mut values, &sources, &format!("{prefix}/linear/{i}"), graph, &output);
        }
        for i in 0..norms {
            let output = if i == 0 { "layer_norm".to_string() } else { format!("layer_norm_{i}") };
            let node = sources[graph]
                .nodes
                .iter()
                .find(|n| n.op == "Mul" && n.inputs.iter().any(|x| x == &output))
                .unwrap_or_else(|| panic!("no Mul node consumes {output}"));
            let gain = node
                .inputs
                .iter()
                .find(|k| *k != &output)
                .unwrap_or_else(|| panic!("Mul node for {output} has no gain input"))
                .clone();
            copy(&mut values, &sources, &format!("{prefix}/norm/{i}"), graph, &gain);
        }
    }
    {
        let d = &sources["decoder_kv"].values;
        assert!(
            transpose_equal_u8(&d["embedding.weight_quantized"], &d["t_quantized"]),
            "embedding is not the tied transpose of t_quantized"
        );
        let prod = sources["decoder_kv"].producing("t");
        assert!(
            prod.op == "DequantizeLinear"
                && prod.inputs
                    == vec![
                        "t_quantized".to_string(),
                        "embedding.weight_scale".to_string(),
                        "embedding.weight_zero_point".to_string()
                    ],
            "unexpected producer for t"
        );
    }
    for suffix in ["quantized", "scale", "zero_point"] {
        copy(
            &mut values,
            &sources,
            &format!("decoder/embedding/{suffix}"),
            "decoder_kv",
            &format!("embedding.weight_{suffix}"),
        );
    }
    if args.packed_head {
        let (original, prov) = values.pop("decoder/embedding/quantized");
        assert_eq!(original.dims, [32768, 640], "embedding shape changed");
        let packed = permute(&original.bytes, &[512, 4, 16, 5, 128], &[0, 3, 1, 4, 2]);
        let restored = permute(&packed, &[512, 5, 4, 128, 16], &[0, 2, 4, 1, 3]);
        assert_eq!(restored, original.bytes, "packed64 round-trip failed");
        let mut prov = prov.expect("embedding provenance missing").as_array().unwrap().clone();
        prov.push(json!("reversible packed64 uint8 codes; no new quantization"));
        values.add(
            "decoder/embedding/packed64",
            TensorVal {
                dtype: 2,
                dims: vec![512, 5, 512, 16],
                bytes: packed,
            },
            Some(Value::Array(prov)),
        );
    }
    copy(&mut values, &sources, "decoder/rotary", "decoder_kv", "rotary.inv_freq");
    projection(&mut values, &sources, "adapter/projection", "adapter", "memory");
    for suffix in ["quantized", "scale", "zero_point"] {
        copy(
            &mut values,
            &sources,
            &format!("adapter/positions/{suffix}"),
            "adapter",
            &format!("pos_embed.weight_{suffix}"),
        );
    }
    for i in 0..28 {
        let output = if i == 0 { "linear".to_string() } else { format!("linear_{i}") };
        projection(&mut values, &sources, &format!("cross/{i}"), "cross_kv", &output);
    }
    for (label, key) in [
        ("linear", "onnx::MatMul_128_add_tensor"),
        ("conv1", "/weight/weight.0/Mul_output_0"),
        ("conv2", "/weight/weight.0_1/Mul_output_0"),
    ] {
        for suffix in ["quantized", "scale", "zero_point"] {
            copy(
                &mut values,
                &sources,
                &format!("frontend/{label}/{suffix}"),
                "frontend.weights",
                &format!("{key}_{suffix}"),
            );
        }
    }
    for label in ["conv1", "conv2"] {
        copy(
            &mut values,
            &sources,
            &format!("frontend/{label}/bias"),
            "frontend.model",
            &format!("{label}.conv.bias"),
        );
    }
    let mut vad_names: Vec<String> = [
        "stft.forward_basis_buffer",
        "decoder.decoder.2.weight",
        "decoder.decoder.2.bias",
        "decoder.rnn.weight_ih",
        "decoder.rnn.weight_hh",
        "decoder.rnn.bias_ih",
        "decoder.rnn.bias_hh",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for i in 0..4 {
        for suffix in ["weight", "bias"] {
            vad_names.push(format!("encoder.{i}.reparam_conv.{suffix}"));
        }
    }
    for name in &vad_names {
        copy(
            &mut values,
            &sources,
            &format!("vad/{name}"),
            "silero-16k",
            &format!("If_0_then_branch__Inline_0__{name}"),
        );
    }

    write_archive(&args.output, &values, &pinned)
}

fn write_archive(output: &Path, values: &Values, pinned: &Value) -> Result<String, String> {
    const HEADER_SIZE: u64 = 64;
    const RECORD_SIZE: u64 = 160;
    let mut names: Vec<&String> = values.order.iter().collect();
    names.sort();
    let mut offset = HEADER_SIZE + RECORD_SIZE * names.len() as u64;
    let mut records = Vec::new();
    let mut payloads: Vec<(u64, &[u8])> = Vec::new();
    for name in &names {
        let data = &values.map[*name];
        assert!(name.len() < 80 && data.dims.len() <= 4, "tensor {name} exceeds directory limits");
        offset = (offset + 63) & !63;
        let mut record = Vec::with_capacity(RECORD_SIZE as usize);
        let mut name_field = [0u8; 80];
        name_field[..name.len()].copy_from_slice(name.as_bytes());
        record.extend_from_slice(&name_field);
        record.extend_from_slice(&data.dtype.to_le_bytes());
        record.extend_from_slice(&(data.dims.len() as u32).to_le_bytes());
        for i in 0..4 {
            let d = if i < data.dims.len() { data.dims[i] as u64 } else { 0 };
            record.extend_from_slice(&d.to_le_bytes());
        }
        record.extend_from_slice(&offset.to_le_bytes());
        record.extend_from_slice(&(data.bytes.len() as u64).to_le_bytes());
        record.extend_from_slice(&[0u8; 24]);
        assert_eq!(record.len() as u64, RECORD_SIZE);
        records.push(record);
        payloads.push((offset, &data.bytes));
        offset += data.bytes.len() as u64;
    }
    let total = offset;
    let mut header = Vec::with_capacity(HEADER_SIZE as usize);
    header.extend_from_slice(b"TRMED001");
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&(records.len() as u32).to_le_bytes());
    header.extend_from_slice(&total.to_le_bytes());
    header.extend_from_slice(&HEADER_SIZE.to_le_bytes());
    header.extend_from_slice(&[0u8; 32]);

    fn emit(f: &mut File, pos: &mut u64, bytes: &[u8]) -> Result<(), String> {
        f.write_all(bytes).map_err(|e| e.to_string())?;
        *pos += bytes.len() as u64;
        Ok(())
    }
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    let mut pos: u64 = 0;
    emit(&mut f, &mut pos, &header)?;
    for r in &records {
        emit(&mut f, &mut pos, r)?;
    }
    for (position, raw) in &payloads {
        if *position > pos {
            let pad = vec![0u8; (position - pos) as usize];
            emit(&mut f, &mut pos, &pad)?;
        }
        emit(&mut f, &mut pos, raw)?;
    }
    drop(f);
    assert_eq!(pos, total);

    let bytes = std::fs::read(output).map_err(|e| e.to_string())?;
    let sha = sha256_hex(&bytes);
    let mut provenance = Map::new();
    for (name, source) in &values.provenance {
        provenance.insert(name.clone(), source.clone());
    }
    let mut report = Map::new();
    report.insert("format".into(), json!("TRMED001"));
    report.insert("source_model".into(), pinned.clone());
    report.insert("bytes".into(), json!(bytes.len()));
    report.insert("sha256".into(), json!(sha));
    report.insert("projection_count".into(), json!(225));
    report.insert("tensors".into(), json!(records.len()));
    report.insert("original_values_preserved".into(), json!(true));
    report.insert("tied_embedding_transpose_verified".into(), json!(true));
    report.insert("vad_source_sha256".into(), json!(VAD_SHA256));
    report.insert("provenance".into(), Value::Object(provenance));
    let report_text = pyjson::dumps_ordered_indent2(&Value::Object(report.clone())) + "\n";
    let report_path = output.with_extension("json");
    std::fs::write(&report_path, report_text).map_err(|e| e.to_string())?;

    // The printed summary drops two keys; the Python dict comprehension
    // keeps the remaining insertion order (IndexMap remove would not).
    let summary = Map::from_iter(
        report
            .iter()
            .filter(|(k, _)| *k != "provenance" && *k != "source_model")
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    Ok(pyjson::dumps_ordered_compact(&Value::Object(summary)))
}


