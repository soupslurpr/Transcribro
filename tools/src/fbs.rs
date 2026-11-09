//! Minimal read-only FlatBuffers access mirroring the generated Python
//! bindings in onnxruntime's `ort_flatbuffers_py.fbs` package — the exact API
//! surface `import_model.py` uses (Table.Offset(voff) semantics included).
//! Inputs are SHA-256-pinned before parsing; out-of-bounds access panics,
//! standing in for the Python code's assertion failures.

#[derive(Clone, Copy)]
pub struct Table<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

fn rd_u16(buf: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap())
}
fn rd_u32(buf: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}
fn rd_i32(buf: &[u8], pos: usize) -> i32 {
    i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())
}

impl<'a> Table<'a> {
    pub fn root(buf: &'a [u8]) -> Table<'a> {
        Table {
            buf,
            pos: rd_u32(buf, 0) as usize,
        }
    }

    /// Absolute buffer position of a field's inline value, or None when the
    /// field is absent. `voff` is the vtable byte offset exactly as passed to
    /// `Table.Offset(...)` in the Python bindings.
    pub fn field_pos(&self, voff: u16) -> Option<usize> {
        let vt = self.pos - rd_i32(self.buf, self.pos) as usize;
        if (voff as usize) >= rd_u16(self.buf, vt) as usize {
            return None;
        }
        let off = rd_u16(self.buf, vt + voff as usize) as usize;
        if off == 0 {
            None
        } else {
            Some(self.pos + off)
        }
    }

    pub fn i32_field(&self, voff: u16) -> i32 {
        match self.field_pos(voff) {
            Some(p) => rd_i32(self.buf, p),
            None => 0,
        }
    }

    pub fn table_field(&self, voff: u16) -> Option<Table<'a>> {
        let p = self.field_pos(voff)?;
        Some(Table {
            buf: self.buf,
            pos: p + rd_u32(self.buf, p) as usize,
        })
    }

    pub fn string_field(&self, voff: u16) -> Option<&'a str> {
        let p = self.field_pos(voff)?;
        Some(str_at(self.buf, p + rd_u32(self.buf, p) as usize))
    }

    fn vec_pos(&self, voff: u16) -> Option<usize> {
        let p = self.field_pos(voff)?;
        Some(p + rd_u32(self.buf, p) as usize)
    }

    pub fn vec_len(&self, voff: u16) -> usize {
        match self.vec_pos(voff) {
            Some(vp) => rd_u32(self.buf, vp) as usize,
            None => 0,
        }
    }

    pub fn vec_table(&self, voff: u16, j: usize) -> Table<'a> {
        let vp = self.vec_pos(voff).expect("vector field absent");
        let ep = vp + 4 + 4 * j;
        Table {
            buf: self.buf,
            pos: ep + rd_u32(self.buf, ep) as usize,
        }
    }

    pub fn vec_string(&self, voff: u16, j: usize) -> &'a str {
        let vp = self.vec_pos(voff).expect("vector field absent");
        let ep = vp + 4 + 4 * j;
        str_at(self.buf, ep + rd_u32(self.buf, ep) as usize)
    }

    pub fn vec_i64(&self, voff: u16) -> Vec<i64> {
        match self.vec_pos(voff) {
            None => Vec::new(),
            Some(vp) => {
                let len = rd_u32(self.buf, vp) as usize;
                (0..len)
                    .map(|j| i64::from_le_bytes(self.buf[vp + 4 + 8 * j..vp + 12 + 8 * j].try_into().unwrap()))
                    .collect()
            }
        }
    }

    pub fn vec_u8(&self, voff: u16) -> Option<&'a [u8]> {
        let vp = self.vec_pos(voff)?;
        let len = rd_u32(self.buf, vp) as usize;
        Some(&self.buf[vp + 4..vp + 4 + len])
    }
}

fn str_at(buf: &[u8], pos: usize) -> &str {
    let len = rd_u32(buf, pos) as usize;
    std::str::from_utf8(&buf[pos + 4..pos + 4 + len]).expect("flatbuffer string is not UTF-8")
}
