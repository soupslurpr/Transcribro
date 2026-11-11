//! Reading ORT flatbuffer models with the same view `import_model.py`'s
//! `Source` class exposes: initializers as (dtype, dims, raw element bytes)
//! and the node list as (op_type, inputs, outputs).
//!
//! Field map (vtable byte offsets), taken from the pinned generated Python
//! bindings (ort_flatbuffers_py/fbs, InferenceSession.py sha256 eab477f6…):
//!   InferenceSession.model      voff 6
//!   Model.graph                 voff 18
//!   Graph.initializers          voff 4   Graph.nodes        voff 8
//!   Node.op_type                voff 14  Node.inputs        voff 20
//!   Node.outputs                voff 22  Node.attributes    voff 24
//!   Attribute.name              voff 4   Attribute.g        voff 18
//!   Tensor.name                 voff 4   Tensor.dims        voff 8
//!   Tensor.data_type            voff 10  Tensor.raw_data    voff 12

use crate::fbs::Table;
use crate::util::sha256_hex;
use std::collections::HashMap;

/// ONNX tensor element type codes used by the converter (its DTYPES keys).
pub fn dtype_size(code: i32) -> Option<usize> {
    match code {
        1 => Some(4), // float32
        2 => Some(1), // uint8
        3 => Some(1), // int8
        6 => Some(4), // int32
        7 => Some(8), // int64
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct TensorVal {
    pub dtype: i32,
    pub dims: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl TensorVal {
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }
    /// The byte length one element occupies; panics on converter-foreign
    /// dtypes, standing in for the Python DTYPES KeyError.
    pub fn elem_size(&self) -> usize {
        dtype_size(self.dtype).unwrap_or_else(|| panic!("unsupported tensor dtype {}", self.dtype))
    }
}

#[derive(Clone, Debug)]
pub struct NodeRec {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub struct Source {
    pub values: HashMap<String, TensorVal>,
    /// Initializer names in graph order (the HashMap loses it; the oracle
    /// dump comparison needs it).
    pub order: Vec<String>,
    pub nodes: Vec<NodeRec>,
}

impl Source {
    /// First node (in graph order) producing `name` — `Source.producing`.
    pub fn producing(&self, name: &str) -> &NodeRec {
        self.nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| o == name))
            .unwrap_or_else(|| panic!("no node produces {name}"))
    }
}

fn model_graph<'a>(data: &'a [u8]) -> Table<'a> {
    Table::root(data)
        .table_field(6)
        .expect("InferenceSession.model missing")
        .table_field(18)
        .expect("Model.graph missing")
}

fn read_graph(g: Table) -> Source {
    let mut values = HashMap::new();
    let mut order = Vec::new();
    for j in 0..g.vec_len(4) {
        let t = g.vec_table(4, j);
        let name = t.string_field(4).expect("initializer without a name").to_string();
        let dtype = t.i32_field(10);
        let dims = t.vec_i64(8);
        let raw = t
            .vec_u8(12)
            .expect("initializer without raw_data (external data is not used by these models)");
        order.push(name.clone());
        values.insert(
            name,
            TensorVal {
                dtype,
                dims,
                bytes: raw.to_vec(),
            },
        );
    }
    let mut nodes = Vec::new();
    for j in 0..g.vec_len(8) {
        let n = g.vec_table(8, j);
        nodes.push(NodeRec {
            op: n.string_field(14).expect("node without op_type").to_string(),
            inputs: (0..n.vec_len(20)).map(|k| n.vec_string(20, k).to_string()).collect(),
            outputs: (0..n.vec_len(22)).map(|k| n.vec_string(22, k).to_string()).collect(),
        });
    }
    Source { values, order, nodes }
}

/// `Source(path_or_bytes, expected_sha256, branch)` from import_model.py.
pub fn load_source(data: &[u8], expected_sha256: &str, branch: Option<&str>) -> Source {
    assert_eq!(
        sha256_hex(data),
        expected_sha256,
        "source model hash mismatch"
    );
    let g = model_graph(data);
    let g = match branch {
        None => g,
        Some(name) => {
            let mut found = None;
            for j in 0..g.vec_len(8) {
                let n = g.vec_table(8, j);
                if n.string_field(14) == Some("If") {
                    for k in 0..n.vec_len(24) {
                        let a = n.vec_table(24, k);
                        if a.string_field(4) == Some(name) {
                            found = Some(a.table_field(18).expect("If attribute without graph"));
                        }
                    }
                }
            }
            found.unwrap_or_else(|| panic!("branch {name} not found"))
        }
    };
    read_graph(g)
}
