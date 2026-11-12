//! Generator for the packaged package-level dependency notices
//! (fix round 2 residual R2 under the third-party-attribution skill:
//! declared dependencies are not hand-recorded in CREDITS; their
//! required notices are generated from the lockfiles with license
//! tooling and packaged with the app).
//!
//! It emits two files:
//!
//! * `Rust-Dependency-NOTICES.txt` (speech licenses): every crate
//!   compiled into `libtranscribro.so` for the shipped Android ABIs,
//!   computed from `cargo metadata` for `engine/` by walking normal
//!   dependency edges whose cfg matches the target, stopping at
//!   proc-macro crates (host-only, like build and dev dependencies).
//!   License expressions are cross-checked against `cargo about`
//!   (config: `engine/about.toml`); license texts are the verbatim
//!   license files from each crate's published package.
//! * `Android-Dependency-NOTICES.txt` (app licenses): every module on
//!   the app's release runtime classpath whose artifact (AAR/JAR) is
//!   packaged, with the license declared in its POM (following parent
//!   POMs where a module declares none). Modules that publish only
//!   metadata (Kotlin Multiplatform parents, BOM platforms) are not
//!   listed; their code ships in the listed `-android`/`-jvm`
//!   variants.
//!
//! The generator fails loudly on anything it cannot settle: a shipped
//! crate with no license files and no `--license-fallback`, a license
//! expression mismatch between `cargo metadata` and `cargo about`, an
//! unrecognized cfg atom, a packaged module with no POM license, or a
//! license it does not know how to bucket. Nothing is improvised.
//!
//! Usage: see tools/notices/README.md for the full recipe.
//!
//! Deliberately uses only serde_json (already a tools dependency)
//! beyond std; POM files are read with a small block scanner because
//! the generator needs only their <licenses> and <parent> blocks.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::Value;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok((rust_count, android_count)) => {
            println!("wrote notices for {rust_count} Rust crates and {android_count} Android modules");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gen-dependency-notices: error: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    metadata: PathBuf,
    about_json: PathBuf,
    registry_src: PathBuf,
    fallbacks: HashMap<String, PathBuf>,
    gradle_components: PathBuf,
    gradle_cache: PathBuf,
    out_rust: PathBuf,
    out_android: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut fallbacks: HashMap<String, PathBuf> = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = args.get(i).ok_or("missing argument")?;
        let val = args.get(i + 1).ok_or_else(|| format!("missing value for {key}"))?;
        if key == "--license-fallback" {
            let (name, dir) = val
                .split_once('=')
                .ok_or_else(|| format!("--license-fallback expects <crate>=<dir>, got {val}"))?;
            fallbacks.insert(name.to_string(), PathBuf::from(dir));
        } else {
            map.insert(key.clone(), val.clone());
        }
        i += 2;
    }
    let get = |k: &str| -> Result<PathBuf, String> {
        map.get(k).map(PathBuf::from).ok_or_else(|| format!("missing required argument {k}"))
    };
    Ok(Args {
        metadata: get("--metadata")?,
        about_json: get("--about-json")?,
        registry_src: get("--registry-src")?,
        fallbacks,
        gradle_components: get("--gradle-components")?,
        gradle_cache: get("--gradle-cache")?,
        out_rust: get("--out-rust")?,
        out_android: get("--out-android")?,
    })
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn as_str(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

// ---------------------------------------------------------------------
// cfg evaluation for the two shipped Android targets.

struct TargetEnv {
    os: &'static str,
    arch: &'static str,
}

fn target_env(triple: &str) -> Result<TargetEnv, String> {
    match triple {
        "aarch64-linux-android" => Ok(TargetEnv { os: "android", arch: "aarch64" }),
        "x86_64-linux-android" => Ok(TargetEnv { os: "android", arch: "x86_64" }),
        other => Err(format!("unsupported target triple {other}")),
    }
}

struct CfgParser<'a> {
    chars: Vec<char>,
    pos: usize,
    env: &'a TargetEnv,
}

impl<'a> CfgParser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn skip_ws(&mut self) {
        while self.peek() == Some(' ') {
            self.pos += 1;
        }
    }
    fn ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.chars[start..self.pos].iter().collect()
    }
    fn string(&mut self) -> Result<String, String> {
        if !self.eat('"') {
            return Err("expected string in cfg".into());
        }
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '"' {
                break;
            }
            self.pos += 1;
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        if !self.eat('"') {
            return Err("unterminated string in cfg".into());
        }
        Ok(s)
    }
    fn expr(&mut self) -> Result<bool, String> {
        self.skip_ws();
        let name = self.ident();
        self.skip_ws();
        match name.as_str() {
            "all" | "any" => {
                if !self.eat('(') {
                    return Err(format!("expected ( after {name}"));
                }
                let mut acc = name == "all";
                loop {
                    let v = self.expr()?;
                    if name == "all" {
                        acc = acc && v;
                    } else {
                        acc = acc || v;
                    }
                    self.skip_ws();
                    if self.eat(',') {
                        continue;
                    }
                    break;
                }
                if !self.eat(')') {
                    return Err(format!("expected ) to close {name}"));
                }
                Ok(acc)
            }
            "not" => {
                if !self.eat('(') {
                    return Err("expected ( after not".into());
                }
                let v = self.expr()?;
                if !self.eat(')') {
                    return Err("expected ) to close not".into());
                }
                Ok(!v)
            }
            "unix" => Ok(true),
            "windows" => Ok(false),
            "" => Err("empty cfg expression".into()),
            key => {
                self.skip_ws();
                if self.eat('=') {
                    self.skip_ws();
                    let value = self.string()?;
                    Ok(match key {
                        "target_os" => value == self.env.os,
                        "target_arch" => value == self.env.arch,
                        "target_family" => value == "unix",
                        "target_pointer_width" => value == "64",
                        "target_endian" => value == "little",
                        "target_env" => value.is_empty(),
                        "target_abi" => value.is_empty(),
                        "target_vendor" => value == "unknown",
                        other => return Err(format!("unknown cfg key {other}")),
                    })
                } else {
                    Err(format!("unknown bare cfg flag {key}"))
                }
            }
        }
    }
}

fn cfg_matches(cfg: &str, env: &TargetEnv) -> Result<bool, String> {
    let inner = cfg
        .strip_prefix("cfg(")
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("malformed cfg expression {cfg}"))?;
    let mut parser = CfgParser { chars: inner.chars().collect(), pos: 0, env };
    let v = parser.expr()?;
    parser.skip_ws();
    if parser.pos != parser.chars.len() {
        return Err(format!("trailing characters in cfg {cfg}"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------
// Rust side: shipped crate set from cargo metadata.

struct Pkg {
    name: String,
    version: String,
    license: String,
    authors: Vec<String>,
    proc_macro: bool,
}

fn shipped_crates(metadata: &Value) -> Result<BTreeMap<String, Pkg>, String> {
    let packages = metadata["packages"].as_array().ok_or("metadata: no packages")?;
    let mut pkgs: HashMap<String, Pkg> = HashMap::new();
    for p in packages {
        let id = as_str(&p["id"]).to_string();
        let proc_macro = p["targets"]
            .as_array()
            .map(|ts| {
                ts.iter().any(|t| {
                    t["kind"].as_array().map(|ks| ks.iter().any(|k| as_str(k) == "proc-macro")).unwrap_or(false)
                })
            })
            .unwrap_or(false);
        pkgs.insert(
            id,
            Pkg {
                name: as_str(&p["name"]).to_string(),
                version: as_str(&p["version"]).to_string(),
                license: as_str(&p["license"]).to_string(),
                authors: p["authors"].as_array().map(|a| a.iter().map(|x| as_str(x).to_string()).collect()).unwrap_or_default(),
                proc_macro,
            },
        );
    }
    let resolve = &metadata["resolve"];
    let root = as_str(&resolve["root"]).to_string();
    if root.is_empty() {
        return Err("metadata: resolve.root is empty".into());
    }
    let mut nodes: HashMap<String, &Value> = HashMap::new();
    for n in resolve["nodes"].as_array().ok_or("metadata: no resolve nodes")? {
        nodes.insert(as_str(&n["id"]).to_string(), n);
    }
    let mut per_target: Vec<BTreeSet<String>> = Vec::new();
    for triple in ["aarch64-linux-android", "x86_64-linux-android"] {
        let env = target_env(triple)?;
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(root.clone());
        seen.insert(root.clone());
        while let Some(id) = queue.pop_front() {
            let node = nodes.get(&id).ok_or_else(|| format!("metadata: missing node {id}"))?;
            for dep in node["deps"].as_array().cloned().unwrap_or_default() {
                let dep_id = as_str(&dep["pkg"]).to_string();
                let dep_pkg = pkgs.get(&dep_id).ok_or_else(|| format!("metadata: missing package {dep_id}"))?;
                if dep_pkg.proc_macro {
                    continue; // host-only: not compiled into the shipped library
                }
                let mut qualifies = false;
                for dk in dep["dep_kinds"].as_array().cloned().unwrap_or_default() {
                    let kind_ok = dk["kind"].is_null() || as_str(&dk["kind"]) == "normal";
                    if !kind_ok {
                        continue;
                    }
                    let target_ok = if dk["target"].is_null() {
                        true
                    } else {
                        cfg_matches(as_str(&dk["target"]), &env)?
                    };
                    if target_ok {
                        qualifies = true;
                        break;
                    }
                }
                if qualifies && seen.insert(dep_id.clone()) {
                    queue.push_back(dep_id);
                }
            }
        }
        seen.remove(&root);
        per_target.push(seen);
    }
    if per_target[0] != per_target[1] {
        let only_a: Vec<&String> = per_target[0].difference(&per_target[1]).collect();
        let only_b: Vec<&String> = per_target[1].difference(&per_target[0]).collect();
        eprintln!("warning: shipped crate sets differ per ABI (aarch64-only: {only_a:?}, x86_64-only: {only_b:?}); using the union");
    }
    let ids: BTreeSet<String> = per_target.into_iter().flatten().collect();
    let mut out: BTreeMap<String, Pkg> = BTreeMap::new();
    for id in ids {
        let p = pkgs.remove(&id).ok_or("internal: package vanished")?;
        out.insert(format!("{} {}", p.name, p.version), p);
    }
    Ok(out)
}

fn is_license_filename(name: &str) -> bool {
    let upper = name.to_uppercase();
    upper.starts_with("LICENSE") || upper.starts_with("COPYING") || upper.starts_with("UNLICENSE") || upper.starts_with("NOTICE")
}

fn license_files_in(dir: &Path) -> Result<Vec<(String, String)>, String> {
    let mut files = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(files),
    };
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_license_filename(&name) && entry.path().is_file() {
            let text = std::fs::read_to_string(entry.path()).map_err(|e| format!("{}: {e}", entry.path().display()))?;
            files.push((name, text));
        }
    }
    files.sort();
    Ok(files)
}

fn registry_pkg_dir(registry_src: &Path, name: &str, version: &str) -> Result<PathBuf, String> {
    let want = format!("{name}-{version}");
    for index in std::fs::read_dir(registry_src).map_err(|e| format!("{}: {e}", registry_src.display()))? {
        let index = index.map_err(|e| e.to_string())?;
        let candidate = index.path().join(&want);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(format!("registry package dir not found for {want}"))
}

fn normalize_expr(expr: &str) -> String {
    expr.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rust_notices(args: &Args, metadata: &Value, about: &Value) -> Result<(String, usize), String> {
    let shipped = shipped_crates(metadata)?;
    // Cross-check license expressions against cargo-about.
    let mut about_exprs: HashMap<(String, String), String> = HashMap::new();
    for c in about["crates"].as_array().ok_or("about.json: no crates")? {
        let p = &c["package"];
        about_exprs.insert(
            (as_str(&p["name"]).to_string(), as_str(&p["version"]).to_string()),
            as_str(&c["license"]).to_string(),
        );
    }
    let mut out = String::new();
    out.push_str("Rust library dependencies\n\n");
    out.push_str(
        "The crates below are compiled into libtranscribro.so, the speech\n\
         engine's native library, for both shipped ABIs. This file is\n\
         generated from engine/Cargo.lock (via cargo metadata and\n\
         cargo-about) and the crates' published packages; do not edit it\n\
         by hand. The regeneration recipe is in tools/notices/README.md.\n\
         Host-only crates (proc-macro crates, build and dev dependencies)\n\
         are not compiled into the library and are not listed.\n\n",
    );
    out.push_str(
        "Each entry gives the crate, its version, its declared license\n\
         expression, and the license texts from its published package,\n\
         verbatim.\n",
    );
    for pkg in shipped.values() {
        let key = (pkg.name.clone(), pkg.version.clone());
        match about_exprs.get(&key) {
            Some(expr) if normalize_expr(expr) == normalize_expr(&pkg.license) => {}
            Some(expr) => {
                return Err(format!(
                    "license mismatch for {} {}: metadata {:?} vs cargo-about {:?}",
                    pkg.name, pkg.version, pkg.license, expr
                ))
            }
            None => return Err(format!("{} {} missing from cargo-about output", pkg.name, pkg.version)),
        }
        let dir = registry_pkg_dir(&args.registry_src, &pkg.name, &pkg.version)?;
        let mut files = license_files_in(&dir)?;
        let mut fallback_note = false;
        if files.is_empty() {
            match args.fallbacks.get(&pkg.name) {
                Some(fb) => {
                    files = license_files_in(fb)?;
                    if files.is_empty() {
                        return Err(format!("fallback dir for {} has no license files", pkg.name));
                    }
                    fallback_note = true;
                }
                None => {
                    return Err(format!(
                        "{} {} ships no license files in its published package and no --license-fallback was given",
                        pkg.name, pkg.version
                    ))
                }
            }
        }
        out.push_str("\n================================================================\n");
        out.push_str(&format!("{} {}\n", pkg.name, pkg.version));
        out.push_str(&format!("License: {}\n", pkg.license));
        if !pkg.authors.is_empty() {
            out.push_str(&format!("Authors: {}\n", pkg.authors.join(", ")));
        }
        if fallback_note {
            out.push_str(
                "\nNote: the published crate package omits its license files;\n\
                 the texts below are the crate's repository license files\n\
                 (see tools/notices/README.md).\n",
            );
        }
        for (name, text) in &files {
            out.push_str(&format!("\n--- {name} ---\n"));
            out.push_str(text);
            if !text.ends_with('\n') {
                out.push('\n');
            }
        }
    }
    Ok((out, shipped.len()))
}

// ---------------------------------------------------------------------
// Android side: packaged Gradle modules and their POM licenses.

fn collect_files(dir: &Path, acc: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, acc);
            } else {
                acc.push(path);
            }
        }
    }
}

fn block(text: &str, open: &str, close: &str) -> Option<String> {
    let start = text.find(open)? + open.len();
    let end = text[start..].find(close)? + start;
    Some(text[start..end].to_string())
}

fn tag_text(text: &str, tag: &str) -> Option<String> {
    let inner = block(text, &format!("<{tag}>"), &format!("</{tag}>"))?;
    Some(inner.trim().to_string())
}

fn pom_licenses(cache: &Path, g: &str, a: &str, v: &str, depth: usize) -> Result<Vec<(String, String)>, String> {
    if depth > 4 {
        return Err(format!("{g}:{a}:{v}: parent POM chain too deep"));
    }
    let dir = cache.join(g).join(a).join(v);
    let mut files = Vec::new();
    collect_files(&dir, &mut files);
    let pom_name = format!("{a}-{v}.pom");
    let pom = files
        .iter()
        .find(|f| f.file_name().map(|n| n == pom_name.as_str()).unwrap_or(false))
        .ok_or_else(|| format!("{g}:{a}:{v}: POM not found in Gradle cache"))?;
    let text = std::fs::read_to_string(pom).map_err(|e| format!("{}: {e}", pom.display()))?;
    if let Some(licenses) = block(&text, "<licenses>", "</licenses>") {
        let mut out = Vec::new();
        for chunk in licenses.split("<license>").skip(1) {
            let name = tag_text(chunk, "name").unwrap_or_default();
            let url = tag_text(chunk, "url").unwrap_or_default();
            if !name.is_empty() {
                out.push((name, url));
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    if let Some(parent) = block(&text, "<parent>", "</parent>") {
        let pg = tag_text(&parent, "groupId").ok_or("parent without groupId")?;
        let pa = tag_text(&parent, "artifactId").ok_or("parent without artifactId")?;
        let pv = tag_text(&parent, "version").ok_or("parent without version")?;
        return pom_licenses(cache, &pg, &pa, &pv, depth + 1);
    }
    Err(format!("{g}:{a}:{v}: no licenses in POM and no parent POM"))
}

fn license_bucket(name: &str, url: &str, module: &str) -> Result<&'static str, String> {
    if name.contains("Apache") && (name.contains("2.0") || url.contains("LICENSE-2.0")) {
        Ok("Apache-2.0")
    } else if name.contains("BSD-3-Clause") || url.contains("BSD-3-Clause") {
        Ok("BSD-3-Clause")
    } else if name == "MIT" || name.contains("MIT License") {
        Ok("MIT")
    } else {
        Err(format!("{module}: unrecognized license {name:?} <{url}>; refusing to improvise"))
    }
}

fn android_notices(args: &Args) -> Result<(String, usize), String> {
    let text = std::fs::read_to_string(&args.gradle_components)
        .map_err(|e| format!("{}: {e}", args.gradle_components.display()))?;
    let mut components: BTreeSet<(String, String, String)> = BTreeSet::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("COMPONENT ") {
            let parts: Vec<&str> = rest.rsplitn(3, ':').collect();
            if parts.len() == 3 {
                components.insert((parts[2].to_string(), parts[1].to_string(), parts[0].to_string()));
            }
        }
    }
    let mut by_bucket: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
    let mut packaged = 0usize;
    for (g, a, v) in &components {
        let dir = args.gradle_cache.join(g).join(a).join(v);
        let mut files = Vec::new();
        collect_files(&dir, &mut files);
        let code = files.iter().any(|f| {
            f.extension().map(|e| e == "aar" || e == "jar").unwrap_or(false)
        });
        if !code {
            continue; // metadata-only module; its code ships in a variant module
        }
        packaged += 1;
        let module = format!("{g}:{a}:{v}");
        for (name, url) in pom_licenses(&args.gradle_cache, g, a, v, 0)? {
            let bucket = license_bucket(&name, &url, &module)?;
            by_bucket.entry(bucket).or_default().insert(module.clone());
        }
    }
    let mut out = String::new();
    out.push_str("Android library dependencies\n\n");
    out.push_str(
        "The libraries below are packaged in the app (its release runtime\n\
         classpath). This file is generated from the Gradle resolution and\n\
         the licenses declared in each module's POM (following parent POMs\n\
         where a module declares none); do not edit it by hand. The\n\
         regeneration recipe is in tools/notices/README.md. Modules that\n\
         publish only metadata (Kotlin Multiplatform parents and BOM\n\
         platforms) are not listed; their code ships in the listed\n\
         -android and -jvm variants.\n",
    );
    for bucket in ["Apache-2.0", "BSD-3-Clause", "MIT"] {
        if let Some(modules) = by_bucket.get(bucket) {
            out.push_str(&format!("\n================================================================\n{bucket}\n"));
            match bucket {
                "Apache-2.0" => out.push_str("The full license text is packaged in NOTICE.txt in the speech licenses folder.\n"),
                "BSD-3-Clause" => out.push_str("The full license text is packaged in Protobuf-LICENSE.txt in the app licenses folder.\n"),
                _ => out.push_str("The full license text is packaged in NOTICE.txt in the speech licenses folder.\n"),
            }
            out.push_str("\nModules:\n");
            for m in modules {
                out.push_str(&format!("{m}\n"));
            }
        }
    }
    Ok((out, packaged))
}

fn run(args: &[String]) -> Result<(usize, usize), String> {
    let args = parse_args(args)?;
    let metadata = read_json(&args.metadata)?;
    let about = read_json(&args.about_json)?;
    let (rust, rust_count) = rust_notices(&args, &metadata, &about)?;
    let (android, android_count) = android_notices(&args)?;
    std::fs::write(&args.out_rust, rust).map_err(|e| format!("{}: {e}", args.out_rust.display()))?;
    std::fs::write(&args.out_android, android).map_err(|e| format!("{}: {e}", args.out_android.display()))?;
    Ok((rust_count, android_count))
}
