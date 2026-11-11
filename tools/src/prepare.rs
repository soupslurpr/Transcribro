//! `prepare.py` reimplemented: checksum-locked acquisition and preparation of
//! native sources and the original Medium assets. Same cache layout
//! (.gradle/native-inputs), same lock file, same identity hashes (Python
//! json.dumps(sort_keys=True) canonical form), same flock serialization.
//!
//! Differences from the Python original, by design:
//!   * No Python venv is built — the converter is this binary (`convert`).
//!   * Download transport is curl (see main.rs note); every verification and
//!     atomicity rule is unchanged.

use crate::convert::{run_convert, ConvertArgs};
use crate::pyjson;
use crate::util::{sha256_file, TempDir};
use serde_json::Value;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct PrepareArgs {
    pub cache: PathBuf,
    pub output: PathBuf,
    pub sources: Option<PathBuf>,
    pub offline: bool,
    pub root: PathBuf,
}

struct Ctx {
    cache: PathBuf,
    offline: bool,
    root: PathBuf,
    speech: PathBuf,
    lock: Value,
}

fn sha256_of(path: &Path) -> Result<String, String> {
    sha256_file(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// prepare.py `matches`: size and SHA-256 both equal the record's.
fn matches(path: &Path, record: &Value) -> bool {
    if !path.is_file() {
        return false;
    }
    let size_ok = fs::metadata(path).map(|m| m.len()).ok() == record["bytes"].as_u64();
    size_ok
        && record["bytes"].as_u64().is_some()
        && sha256_of(path).map(|h| h == record["sha256"].as_str().unwrap_or("")).unwrap_or(false)
}

fn acquire(ctx: &Ctx, record: &Value) -> Result<PathBuf, String> {
    let name = record
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| record["sha256"].as_str().unwrap())
        .to_string();
    let dir = ctx.cache.join("downloads");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let target = dir.join(&name);
    if matches(&target, record) {
        return Ok(target);
    }
    if ctx.offline {
        return Err(format!(
            "Missing or invalid pinned input {name}; run once without --offline"
        ));
    }
    let url = record["url"].as_str().ok_or("pinned record without url")?;
    if !url.starts_with("https://") {
        return Err("Pinned downloads require HTTPS".to_string());
    }
    println!("Downloading pinned input: {name}");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let partial = dir.join(format!("tmp{}-{}", std::process::id(), nanos));
    let status = Command::new("curl")
        .args([
            "-fSL",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--speed-limit",
            "1024",
            "--speed-time",
            "180",
            "-o",
        ])
        .arg(&partial)
        .arg(url)
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    let result = (|| {
        if !status.success() {
            return Err(format!("download failed for {name}"));
        }
        if !matches(&partial, record) {
            return Err(format!("Size or SHA-256 mismatch: {name}"));
        }
        fs::rename(&partial, &target).map_err(|e| e.to_string())?;
        Ok(target.clone())
    })();
    if partial.exists() {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// prepare.py `unpack`: strip the archive's single top-level component,
/// reject `..`, apply the include filter, write regular files (mode & 0o777)
/// and directories; links and devices are ignored.
fn unpack(archive_path: &Path, output: &Path, includes: Option<&[&str]>) -> Result<(), String> {
    fs::create_dir(output).map_err(|e| e.to_string())?;
    use std::io::{BufRead, Read};
    let raw = fs::File::open(archive_path).map_err(|e| e.to_string())?;
    let mut reader = std::io::BufReader::new(raw);
    let is_gzip = {
        let head = reader.fill_buf().map_err(|e| e.to_string())?;
        head.len() >= 2 && head[0] == 0x1f && head[1] == 0x8b
    };
    let inner: Box<dyn Read> = if is_gzip {
        Box::new(flate2::read::GzDecoder::new(reader))
    } else {
        Box::new(reader)
    };
    let mut tar = tar::Archive::new(inner);
    let mut entries = tar.entries().map_err(|e| e.to_string())?;
    while let Some(entry) = entries.next() {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.to_path_buf();
        let parts: Vec<String> = path
            .components()
            .filter_map(|c| match c {
                std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect();
        if parts.len() < 2 {
            continue;
        }
        let relative_parts = &parts[1..];
        if relative_parts.iter().any(|p| p == "..") {
            return Err("Unsafe native archive path".to_string());
        }
        let relative = relative_parts.join("/");
        if let Some(includes) = includes {
            if !includes.iter().any(|p| relative == *p || relative.starts_with(&format!("{p}/"))) {
                continue;
            }
        }
        let destination = output.join(&relative);
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
        } else if kind.is_file() {
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut f = fs::File::create(&destination).map_err(|e| e.to_string())?;
            std::io::copy(&mut entry, &mut f).map_err(|e| e.to_string())?;
            let mode = entry.header().mode().unwrap_or(0o644) & 0o777;
            fs::set_permissions(&destination, fs::Permissions::from_mode(mode))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn sources_mode(ctx: &Ctx, output: &Path) -> Result<String, String> {
    let native = ctx.lock["native"].as_object().ok_or("lock.native not an object")?;
    let mut archives: Vec<(String, PathBuf)> = Vec::new();
    for (name, record) in native {
        archives.push((name.clone(), acquire(ctx, record)?));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp = TempDir::new_in(output.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    let prepared = temp.path().join("sources");
    fs::create_dir(&prepared).map_err(|e| e.to_string())?;
    for (name, archive) in &archives {
        let includes: Option<Vec<&str>> = match name.as_str() {
            "moonshine" => Some(vec!["LICENSE", "core/silero-vad-model-data.h", "core/moonshine-c-api.h"]),
            _ => None,
        };
        unpack(archive, &prepared.join(name), includes.as_deref())?;
    }
    let header = prepared.join("moonshine/core/silero-vad-model-data.h");
    if sha256_of(&header)? != ctx.lock["vad"]["header_sha256"].as_str().unwrap_or("") {
        return Err("Pinned inner VAD source changed".to_string());
    }
    let inputs = pyjson::dumps_sorted_indent2(&ctx.lock["native"]) + "\n";
    fs::write(prepared.join("inputs.json"), inputs).map_err(|e| e.to_string())?;
    if output.exists() {
        fs::remove_dir_all(output).map_err(|e| e.to_string())?;
    }
    fs::rename(&prepared, output).map_err(|e| e.to_string())?;
    Ok("Pinned native sources prepared".to_string())
}

fn zstd_program(ctx: &Ctx, sources_dir: &Path) -> Result<PathBuf, String> {
    let dir = ctx
        .cache
        .join("zstd-host")
        .join(ctx.lock["native"]["zstd"]["sha256"].as_str().unwrap_or(""));
    let executable = dir.join("programs/zstd");
    if !executable.is_file() {
        let src = sources_dir.join("zstd/build/cmake");
        let run = |args: &[&str]| -> Result<(), String> {
            let st = Command::new("cmake").args(args).status().map_err(|e| e.to_string())?;
            if st.success() {
                Ok(())
            } else {
                Err("cmake failed while building the pinned zstd".to_string())
            }
        };
        run(&[
            "-S",
            src.to_str().unwrap(),
            "-B",
            dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DZSTD_BUILD_TESTS=OFF",
            "-DZSTD_BUILD_SHARED=OFF",
            "-DZSTD_BUILD_PROGRAMS=ON",
            "-DZSTD_MULTITHREAD_SUPPORT=ON",
        ])?;
        run(&["--build", dir.to_str().unwrap(), "--target", "zstd", "--parallel", "2"])?;
    }
    Ok(executable)
}

fn model_mode(ctx: &Ctx, output: &Path, sources_dir: &Path) -> Result<String, String> {
    let outer = &ctx.lock["outer_vad"];
    if sha256_of(&ctx.root.join(outer["path"].as_str().unwrap()))? != outer["sha256"].as_str().unwrap_or("") {
        return Err("Pinned outer Silero model changed".to_string());
    }
    for (name, expected) in ctx.lock["converter_inputs"].as_object().ok_or("lock.converter_inputs")? {
        if sha256_of(&ctx.root.join(name))? != expected.as_str().unwrap_or("") {
            return Err(format!("Converter input differs from native-inputs.lock.json: {name}"));
        }
    }
    let pinned: Value = serde_json::from_str(
        &fs::read_to_string(ctx.root.join("docs/moonshine-model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(ctx.speech.join("medium-model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let identity = crate::util::sha256_hex(pyjson::dumps_sorted_compact(&ctx.lock).as_bytes());
    let artifacts = ctx.cache.join("models").join(&identity);
    let files = manifest["files"].as_array().ok_or("manifest files")?;
    let mut valid = files.iter().all(|f| matches(&artifacts.join(f["file"].as_str().unwrap()), f));
    valid = valid
        && matches(
            &artifacts.join("medium.trm"),
            &serde_json::json!({"bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}),
        );
    if !valid {
        let base_url = pinned["base_url"].as_str().unwrap();
        let mut originals: Vec<(Value, PathBuf)> = Vec::new();
        for f in pinned["files"].as_array().ok_or("pinned files")? {
            let mut record = f.clone();
            record
                .as_object_mut()
                .unwrap()
                .insert("url".into(), Value::String(format!("{base_url}/{}", f["file"].as_str().unwrap())));
            record
                .as_object_mut()
                .unwrap()
                .insert("filename".into(), f["sha256"].clone());
            let path = acquire(ctx, &record)?;
            originals.push((f.clone(), path));
        }
        let zstd = zstd_program(ctx, sources_dir)?;
        fs::create_dir_all(artifacts.parent().unwrap()).map_err(|e| e.to_string())?;
        let temp = TempDir::new_in(artifacts.parent().unwrap()).map_err(|e| e.to_string())?;
        let originals_dir = temp.path().join("originals");
        fs::create_dir(&originals_dir).map_err(|e| e.to_string())?;
        for (record, path) in &originals {
            let dest = originals_dir.join(record["file"].as_str().unwrap());
            if fs::hard_link(path, &dest).is_err() {
                fs::copy(path, &dest).map_err(|e| e.to_string())?;
            }
        }
        let converted = temp.path().join("converted");
        fs::create_dir(&converted).map_err(|e| e.to_string())?;
        let archive = converted.join("medium.trm");
        run_convert(
            &ctx.root,
            &ConvertArgs {
                models: originals_dir.clone(),
                output: archive.clone(),
                vad_header: sources_dir.join("moonshine/core/silero-vad-model-data.h"),
                packed_head: true,
            },
        )?;
        if !matches(
            &archive,
            &serde_json::json!({"bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}),
        ) {
            return Err("Original Medium archive changed".to_string());
        }
        let st = Command::new(&zstd)
            .args(["-T1", "-9"])
            .arg(&archive)
            .arg("-o")
            .arg(converted.join("medium.trm.zst"))
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            return Err("zstd compression failed".to_string());
        }
        fs::copy(originals_dir.join("tokenizer.bin"), converted.join("tokenizer.bin"))
            .map_err(|e| e.to_string())?;
        for record in files {
            if !matches(&converted.join(record["file"].as_str().unwrap()), record) {
                return Err(format!(
                    "Converted artifact differs from pinned model: {}",
                    record["file"].as_str().unwrap()
                ));
            }
        }
        if artifacts.exists() {
            fs::remove_dir_all(&artifacts).map_err(|e| e.to_string())?;
        }
        fs::rename(&converted, &artifacts).map_err(|e| e.to_string())?;
    }
    if output.exists() {
        fs::remove_dir_all(output).map_err(|e| e.to_string())?;
    }
    let destination = output.join("models/standalone");
    fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
    for record in files {
        // Gate 7: the compressed v1 asset is no longer published. The
        // runtime memory-maps the uncompressed archive (proven on
        // device at Gate 6), so the .zst remains only in the verified
        // artifacts cache, where its pinned hash is the Gate 1 proof.
        if record["file"].as_str() == Some("medium.trm.zst") {
            continue;
        }
        fs::copy(
            artifacts.join(record["file"].as_str().unwrap()),
            destination.join(record["file"].as_str().unwrap()),
        )
        .map_err(|e| e.to_string())?;
    }
    fs::copy(ctx.speech.join("medium-model.json"), destination.join("model.json"))
        .map_err(|e| e.to_string())?;
    // Gate 6: also publish the TRMED002 (format v2) archive. The
    // Android runtime memory-maps this uncompressed file and borrows
    // its weights zero-copy. Conversion is value-verified
    // against the v1 archive by construction (see convert_v2.rs and
    // the verify_trmed002 tool).
    crate::convert_v2::run_convert_v2(
        &artifacts.join("medium.trm"),
        &destination.join("medium.trm"),
    )?;
    Ok(format!(
        "Exact original Medium assets prepared: {}",
        manifest["archive_sha256"].as_str().unwrap_or("")
    ))
}

pub fn run(mode: &str, args: &PrepareArgs) -> Result<String, String> {
    let cache = args.cache.canonicalize().or_else(|_| {
        fs::create_dir_all(&args.cache).map_err(|e| e.to_string())?;
        args.cache.canonicalize().map_err(|e| e.to_string())
    })?;
    let output = absolutize(&args.output);
    let lock_text = fs::read_to_string(args.root.join("speech/native-inputs.lock.json"))
        .map_err(|e| e.to_string())?;
    let lock: Value = serde_json::from_str(&lock_text).map_err(|e| e.to_string())?;
    fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
    // Serialize the shared cache across variant builds, as prepare.py does.
    let lock_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(cache.join("prepare.lock"))
        .map_err(|e| e.to_string())?;
    use std::os::unix::io::AsRawFd;
    if unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err("could not flock the prepare cache".to_string());
    }
    let ctx = Ctx {
        cache,
        offline: args.offline,
        speech: args.root.join("speech"),
        root: args.root.clone(),
        lock,
    };
    match mode {
        "sources" => sources_mode(&ctx, &output),
        "model" => {
            let sources = args
                .sources
                .as_ref()
                .map(|s| absolutize(s))
                .ok_or("model preparation requires --sources")?;
            model_mode(&ctx, &output, &sources)
        }
        _ => unreachable!(),
    }
}

fn absolutize(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap().join(p)
    }
}

