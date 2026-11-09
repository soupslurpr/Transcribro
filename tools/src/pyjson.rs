//! Serialization that reproduces CPython `json.dumps` byte-for-byte for the
//! value shapes this project uses (strings, integers, booleans, null, arrays,
//! objects). Two modes are needed:
//!   * `sort_keys=True` (compact separators ", " / ": ") — cache identities
//!     and `inputs.json` (the latter with indent=2).
//!   * insertion-ordered with indent=2 — the converter's report JSON.
//! String escaping follows `ensure_ascii=True`: short escapes for
//! \b \f \n \r \t, \" and \\, \u00XX for other control characters, and
//! \uXXXX (with surrogate pairs) for everything outside ASCII space..tilde.
//! Floats never occur in the pinned inputs (verified: no float tokens in
//! native-inputs.lock.json / moonshine-model.json); a float would be rendered
//! with serde_json's Ryu formatting, which can differ from Python repr in
//! exponent style — the dumper asserts against floats on the sorted path so a
//! future float in a pin fails loudly instead of silently changing an
//! identity hash.

use serde_json::Value;

fn escape_into(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ' '..='~' => out.push(ch),
            c if (c as u32) < 0x1_0000 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => {
                let v = (c as u32) - 0x1_0000;
                let hi = 0xD800 + (v >> 10);
                let lo = 0xDC00 + (v & 0x3FF);
                out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
            }
        }
    }
    out.push('"');
}

fn scalar_into(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => escape_into(s, out),
        _ => unreachable!("scalar_into called on a container"),
    }
}

/// json.dumps(value, sort_keys=True) — compact, separators (", ", ": ").
pub fn dumps_sorted_compact(v: &Value) -> String {
    let mut out = String::new();
    sorted_into(v, false, 0, &mut out);
    out
}

/// json.dumps(value, sort_keys=True, indent=2).
pub fn dumps_sorted_indent2(v: &Value) -> String {
    let mut out = String::new();
    sorted_into(v, true, 0, &mut out);
    out
}

/// json.dumps(value, indent=2) preserving object insertion order.
pub fn dumps_ordered_indent2(v: &Value) -> String {
    let mut out = String::new();
    ordered_into(v, 0, &mut out);
    out
}

/// json.dumps(value) preserving object insertion order (", " / ": ").
pub fn dumps_ordered_compact(v: &Value) -> String {
    let mut out = String::new();
    ordered_compact_into(v, &mut out);
    out
}

fn check_no_float(v: &Value) {
    match v {
        Value::Number(n) => assert!(
            n.is_i64() || n.is_u64(),
            "float in a pinned JSON document: canonical dump would diverge from Python"
        ),
        Value::Array(a) => a.iter().for_each(check_no_float),
        Value::Object(m) => m.values().for_each(check_no_float),
        _ => {}
    }
}

fn sorted_into(v: &Value, indent: bool, level: usize, out: &mut String) {
    match v {
        Value::Object(m) => {
            check_no_float(v);
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if indent {
                    out.push('\n');
                    out.push_str(&"  ".repeat(level + 1));
                } else if i > 0 {
                    out.push(' ');
                }
                escape_into(k, out);
                out.push_str(": ");
                sorted_into(&m[*k], indent, level + 1, out);
            }
            if indent {
                out.push('\n');
                out.push_str(&"  ".repeat(level));
            }
            out.push('}');
        }
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if indent {
                    out.push('\n');
                    out.push_str(&"  ".repeat(level + 1));
                } else if i > 0 {
                    out.push(' ');
                }
                sorted_into(item, indent, level + 1, out);
            }
            if indent {
                out.push('\n');
                out.push_str(&"  ".repeat(level));
            }
            out.push(']');
        }
        _ => scalar_into(v, out),
    }
}

fn ordered_into(v: &Value, level: usize, out: &mut String) {
    match v {
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&"  ".repeat(level + 1));
                escape_into(k, out);
                out.push_str(": ");
                ordered_into(val, level + 1, out);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(level));
            out.push('}');
        }
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&"  ".repeat(level + 1));
                ordered_into(item, level + 1, out);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(level));
            out.push(']');
        }
        _ => scalar_into(v, out),
    }
}

fn ordered_compact_into(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                escape_into(k, out);
                out.push_str(": ");
                ordered_compact_into(val, out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                ordered_compact_into(item, out);
            }
            out.push(']');
        }
        _ => scalar_into(v, out),
    }
}
