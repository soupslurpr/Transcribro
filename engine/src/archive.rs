//! TRMED001 archive parsing + Gate 2 weight extraction.
//! Mirrors the C++ `Archive` parser in medium.cpp (Gate 0 census §0.5):
//! header checks, directory walk, extent/monotonicity checks, and the
//! schema expectations for the frontend + inner-VAD tensor groups.
//! The remaining tensor groups are schema-checked at their own gates.

use std::collections::HashMap;

pub struct Tensor {
    pub offset: usize,
    pub ty: u32,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

pub struct ParsedArchive<'a> {
    data: &'a [u8],
    pub tensors: HashMap<String, Tensor>,
    /// TRMED002 stores projection weights pre-unpacked in natural
    /// [k][n] order (Gate 6 format v2); TRMED001 stores them in the
    /// layout-8 packing the loader must undo.
    pub natural_weights: bool,
    /// TRMED003 additionally carries derived kernel-ready layout
    /// tensors (Gate 6 speed round 2): the three packed projection
    /// layouts per matrix, the natural-order embedding codes, and
    /// the head's FP32 tiles — pure rearrangements of the census
    /// values, borrowed zero-copy by the loader when present.
    pub derived_layouts: bool,
}

fn le_u32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn le_u64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(d[o..o + 8].try_into().unwrap())
}

pub fn parse(data: &[u8]) -> Result<ParsedArchive<'_>, String> {
    let length = data.len();
    if length < 64 || length > 1024 * 1024 * 1024 {
        return Err("archive memory extent or alignment".into());
    }
    let (natural_weights, derived_layouts) =
        if &data[0..8] == b"TRMED001" && le_u32(data, 8) == 1 {
            (false, false)
        } else if &data[0..8] == b"TRMED002" && le_u32(data, 8) == 2 {
            (true, false)
        } else if &data[0..8] == b"TRMED003" && le_u32(data, 8) == 3 {
            (true, true)
        } else {
            return Err("unsupported archive".into());
        };
    if le_u64(data, 16) != length as u64 || le_u64(data, 24) != 64 {
        return Err("archive header mismatch".into());
    }
    let count = le_u32(data, 12) as usize;
    if count == 0 || count > 4096 || 64 + count * 160 > length {
        return Err("invalid archive directory".into());
    }
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut previous_end = 64 + count * 160;
    for i in 0..count {
        let p = 64 + i * 160;
        let rec = &data[p..p + 160];
        let name_len = rec[..80].iter().position(|&b| b == 0).unwrap_or(80);
        if name_len == 0 || name_len >= 80 {
            return Err("invalid tensor name".into());
        }
        let name = String::from_utf8_lossy(&rec[..name_len]).into_owned();
        let ty = le_u32(data, p + 80);
        let rank = le_u32(data, p + 84) as usize;
        if rank > 4 {
            return Err("invalid tensor rank".into());
        }
        let mut elements: usize = 1;
        let mut shape = Vec::with_capacity(rank);
        for j in 0..rank {
            let dim = le_u64(data, p + 88 + j * 8) as usize;
            if dim == 0 || dim > 16777216 || elements > length / dim.max(1) {
                return Err("invalid tensor shape".into());
            }
            shape.push(dim);
            elements *= dim;
        }
        let width = match ty {
            1 | 6 => 4,
            7 => 8,
            2 | 3 => 1,
            _ => return Err("invalid tensor type".into()),
        };
        let offset = le_u64(data, p + 120) as usize;
        let bytes = le_u64(data, p + 128) as usize;
        if offset % 64 != 0
            || offset < previous_end
            || offset > length
            || bytes != elements * width
            || bytes > length - offset
        {
            return Err("invalid tensor extent".into());
        }
        previous_end = offset + bytes;
        if tensors
            .insert(name.clone(), Tensor { offset, ty, shape, bytes })
            .is_some()
        {
            return Err("duplicate tensor name".into());
        }
    }
    if previous_end != length {
        return Err("unexpected trailing model bytes".into());
    }
    if tensors.len() != 1230 && !(derived_layouts && tensors.len() > 1230) {
        return Err("unexpected Medium tensor count".into());
    }
    Ok(ParsedArchive { data, tensors, natural_weights, derived_layouts })
}

impl<'a> ParsedArchive<'a> {
    pub fn expect(&self, name: &str, ty: u32, shape: &[usize]) -> Result<&Tensor, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.ty != ty || t.shape != shape {
            return Err(format!("unexpected tensor schema: {name}"));
        }
        if ty == 1 {
            for v in self.f32_vec(t) {
                if !v.is_finite() {
                    return Err("nonfinite model metadata".into());
                }
            }
        }
        Ok(t)
    }
    pub fn tensor_data(&self, t: &Tensor) -> &'a [u8] {
        &self.data[t.offset..t.offset + t.bytes]
    }
    pub fn f32_vec(&self, t: &Tensor) -> Vec<f32> {
        let raw = &self.data[t.offset..t.offset + t.bytes];
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    pub fn i8_vec(&self, t: &Tensor) -> Vec<i8> {
        let raw = &self.data[t.offset..t.offset + t.bytes];
        raw.iter().map(|&b| b as i8).collect()
    }
    pub fn f32_of(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.expect(name, 1, shape)?;
        Ok(self.f32_vec(t))
    }
}

impl<'a> ParsedArchive<'a> {
    pub fn i32_vec(&self, t: &Tensor) -> Vec<i32> {
        let raw = &self.data[t.offset..t.offset + t.bytes];
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    pub fn u8_vec(&self, t: &Tensor) -> Vec<u8> {
        self.data[t.offset..t.offset + t.bytes].to_vec()
    }
}
