//! transcribro-prepare: Rust replacement for speech/tools/prepare.py and
//! speech/tools/import_model.py (Transcribro Kotlin+Rust rewrite, Gate 1).
//!
//! Subcommands mirror the Python CLIs:
//!   transcribro-prepare sources --cache DIR --output DIR [--offline]
//!   transcribro-prepare model   --cache DIR --output DIR --sources DIR [--offline]
//!   transcribro-prepare convert --models DIR --output FILE --vad-header FILE [--packed-head]
//!   transcribro-prepare convert-v2 [--input] IN.trm [--output] OUT.trm2
//!                                 (TRMED001 -> TRMED002, Gate 6 S1; a separate
//!                                 verify_trmed002 binary checks the result)
//!   transcribro-prepare dump    --models DIR --vad-header FILE   (dev aid:
//!                                 per-initializer/node JSON for oracle diffs)
//!
//! Download transport shells out to curl: it is the transport already proven
//! against every pinned host in this environment (the model CDN rejects the
//! Python-urllib user agent; the sandbox egress proxy is curl-friendly). All
//! policy — HTTPS-only, size + SHA-256 verification, temp file + atomic
//! replace, offline cache-miss failure — is enforced here, as in prepare.py.

mod convert;
mod convert_v2;
mod fbs;
mod ort;
mod prepare;
mod pyjson;
mod util;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(msg) => {
            if !msg.is_empty() {
                println!("{msg}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Resolves the repository root: --root, $TRANSCRIBRO_ROOT, then ancestors of
/// the executable and of the CWD that contain docs/moonshine-model.json
/// (mirroring how the Python tools derive it from their own file location).
pub fn resolve_root(explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(r) = explicit {
        return Ok(PathBuf::from(r));
    }
    if let Ok(r) = std::env::var("TRANSCRIBRO_ROOT") {
        return Ok(PathBuf::from(r));
    }
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    for start in starts {
        let mut dir: Option<&std::path::Path> = Some(start.as_path());
        while let Some(d) = dir {
            if d.join("docs/moonshine-model.json").is_file() && d.join("speech/native-inputs.lock.json").is_file() {
                return Ok(d.to_path_buf());
            }
            dir = d.parent();
        }
    }
    Err("could not locate the repository root (docs/moonshine-model.json); pass --root".into())
}

fn dispatch(args: &[String]) -> Result<String, String> {
    let (cmd, rest) = args.split_first().ok_or("no subcommand given")?;
    match cmd.as_str() {
        "convert" => {
            let a = convert::ConvertArgs {
                models: PathBuf::from(flag(rest, "--models").ok_or("--models required")?),
                output: PathBuf::from(flag(rest, "--output").ok_or("--output required")?),
                vad_header: PathBuf::from(flag(rest, "--vad-header").ok_or("--vad-header required")?),
                packed_head: rest.iter().any(|a| a == "--packed-head"),
            };
            let root = resolve_root(flag(rest, "--root"))?;
            convert::run_convert(&root, &a)
        }
        "sources" | "model" => {
            let p = prepare::PrepareArgs {
                cache: PathBuf::from(flag(rest, "--cache").ok_or("--cache required")?),
                output: PathBuf::from(flag(rest, "--output").ok_or("--output required")?),
                sources: flag(rest, "--sources").map(PathBuf::from),
                offline: rest.iter().any(|a| a == "--offline"),
                root: resolve_root(flag(rest, "--root"))?,
            };
            if cmd == "model" && p.sources.is_none() {
                return Err("model preparation requires --sources".into());
            }
            prepare::run(cmd, &p)
        }
        "convert-v2" => {
            // Path args, positional or flagged:
            //   convert-v2 IN.trm OUT.trm2
            //   convert-v2 --input IN.trm --output OUT.trm2
            let (input, output) = match (flag(rest, "--input"), flag(rest, "--output")) {
                (Some(i), Some(o)) => (PathBuf::from(i), PathBuf::from(o)),
                (None, None) => {
                    let mut pos: Vec<&str> = Vec::new();
                    let mut it = rest.iter();
                    while let Some(a) = it.next() {
                        if a == "--root" {
                            it.next();
                        } else if !a.starts_with("--") {
                            pos.push(a.as_str());
                        }
                    }
                    if pos.len() != 2 {
                        return Err("usage: convert-v2 [--input] IN.trm [--output] OUT.trm2".into());
                    }
                    (PathBuf::from(pos[0]), PathBuf::from(pos[1]))
                }
                _ => return Err("convert-v2 needs both --input and --output, or two positional paths".into()),
            };
            convert_v2::run_convert_v2(&input, &output)
        }
        "dump" => dump(rest),
        _ => Err(format!("unknown subcommand {cmd}")),
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].as_str())
}

/// Dev/verification aid: dump every initializer (name, dtype, dims, byte
/// length, SHA-256 of the element bytes) and node (op, inputs, outputs) for
/// each pinned .ort in --models and for the Silero then_branch, in the same
/// JSON shape as the Python oracle dump used to validate the fbs reader.
fn dump(args: &[String]) -> Result<String, String> {
    let models = PathBuf::from(flag(args, "--models").ok_or("--models required")?);
    let root = resolve_root(flag(args, "--root"))?;
    let pinned: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("docs/moonshine-model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut out = serde_json::Map::new();
    for f in pinned["files"].as_array().ok_or("files not an array")? {
        let file = f["file"].as_str().unwrap();
        if !file.ends_with(".ort") {
            continue;
        }
        let data = std::fs::read(models.join(file)).map_err(|e| e.to_string())?;
        let src = ort::load_source(&data, f["sha256"].as_str().unwrap(), None);
        out.insert(
            file.strip_suffix(".ort").unwrap().to_string(),
            dump_source(&src),
        );
    }
    if let Some(h) = flag(args, "--vad-header") {
        let text = std::fs::read_to_string(h).map_err(|e| e.to_string())?;
        let data = ort::extract_hex_bytes(&text);
        let src = ort::load_source(
            &data,
            "a620103ef77fbdae23c7798b9271506a48a883997e55f851d2884551e0947437",
            Some("then_branch"),
        );
        out.insert("silero-16k".to_string(), dump_source(&src));
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(out)).map_err(|e| e.to_string())
}

fn dump_source(src: &ort::Source) -> serde_json::Value {
    let inits: Vec<serde_json::Value> = src
        .order
        .iter()
        .map(|name| {
            let t = src.values.get(name.as_str()).unwrap();
            serde_json::json!({
                "name": name,
                "dtype": t.dtype,
                "dims": t.dims,
                "sha256": util::sha256_hex(&t.bytes),
                "bytes": t.bytes.len(),
            })
        })
        .collect();
    let nodes: Vec<serde_json::Value> = src
        .nodes
        .iter()
        .map(|n| serde_json::json!({"op": n.op, "inputs": n.inputs, "outputs": n.outputs}))
        .collect();
    serde_json::json!({"initializers": inits, "nodes": nodes})
}
