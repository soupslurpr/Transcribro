//! Gate 6: the JNI bridge -- the pure-Rust replacement for the C++
//! `engine.cpp`. Exports exactly the six `NativeEngine` symbols the
//! unmodified Kotlin class binds (`System.loadLibrary("transcribro")`),
//! with the C++ file's observable semantics copied point for point:
//! analysis interval 16,000 samples with previews / 8,000 without,
//! deferred analysis bounded at 32,768 samples, an 8,192-sample silent
//! drain on phrase finish (recognizer only), the delivered/preview
//! segment rule (last array entry is always the revisable phrase), the
//! 16,000-byte result cap, 2,000-segment cap, and pause/cancel via a
//! mutex/condvar pair + atomic.
//!
//! Model loading follows the mmap ruling: the medium archive ships
//! UNCOMPRESSED (`models/standalone/medium.trm`, TRMED001/TRMED002) in a
//! noCompress asset slot, so `openFd` yields a file extent that is
//! memory-mapped in place -- the OS page cache is shared across the
//! dictation/file processes and pages are faulted on demand, never a
//! ~238 MiB resident inflate at engine create. The legacy `.trm.zst`
//! asset is the fallback: mmapped (or streamed via `open()`) and decoded
//! with pure-Rust ruzstd only when no plain asset exists.

#![allow(deprecated)] // jni 0.22 Env array helpers; typed wrappers are noise here.

use crate::outer_vad::OuterVad;
use crate::stream::Stream;
use jni::objects::{JClass, JFloatArray, JObject, JObjectArray, JString, JValue};
use jni::sys::{jboolean, jfloat, jlong, JNI_TRUE};
use jni::strings::JNIString;
use jni::{jni_sig, jni_str, Env, EnvUnowned};
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

const MODEL_PLAIN: &str = "models/standalone/medium.trm";
const MODEL_ZST: &str = "models/standalone/medium.trm.zst";
const TOKENIZER: &str = "models/standalone/tokenizer.bin";
const VAD_MODEL: &str = "models/silero_vad/ggml-silero-v6.2.0.bin";

/// The engine's hold on the model archive: the shared
/// `crate::model::Backing` (mmap of the asset extent, or owned bytes
/// for the decoded/streamed fallback), reference-counted because the
/// Model borrows its TRMED002 weights from it zero-copy.
type SharedBacking = std::sync::Arc<crate::model::Backing>;

fn fail(env: &mut Env, message: &str) {
    if !env.exception_check() {
        let _ = env.throw_new(
            jni_str!("java/lang/IllegalStateException"),
            JNIString::from(message),
        );
    }
}

fn read_asset_stream<'a>(env: &mut Env<'a>, assets: &JObject<'a>, name: &str) -> Result<Vec<u8>, String> {
    let jname: JString = env.new_string(name).map_err(|_| "string".to_string())?;
    let stream = env
        .call_method(
            assets,
            jni_str!("open"),
            jni_sig!("(Ljava/lang/String;)Ljava/io/InputStream;"),
            &[JValue::Object(&jname)],
        )
        .and_then(|v| v.l())
        .map_err(|_| format!("cannot open asset {name}"))?;
    let buffer = env.new_byte_array(65536).map_err(|_| "buffer".to_string())?;
    let mut out = Vec::new();
    let mut chunk = [0i8; 65536];
    loop {
        let n = env
            .call_method(&stream, jni_str!("read"), jni_sig!("([B)I"), &[JValue::Object(&buffer)])
            .and_then(|v| v.i())
            .map_err(|_| format!("cannot read asset {name}"))?;
        if n < 0 {
            break;
        }
        if n == 0 {
            continue;
        }
        env.get_byte_array_region(&buffer, 0, &mut chunk[..n as usize])
            .map_err(|_| "region".to_string())?;
        out.extend(chunk[..n as usize].iter().map(|&b| b as u8));
    }
    let _ = env.call_method(&stream, jni_str!("close"), jni_sig!("()V"), &[]);
    Ok(out)
}

/// mmap the asset's fd extent, if Android exposes one (stored,
/// uncompressed assets). The returned backing's `bytes()` is exactly
/// the asset extent.
fn mmap_asset<'a>(env: &mut Env<'a>, assets: &JObject<'a>, name: &str) -> Result<SharedBacking, String> {
    let jname: JString = env.new_string(name).map_err(|_| "string".to_string())?;
    let afd = env
        .call_method(
            assets,
            jni_str!("openFd"),
            jni_sig!("(Ljava/lang/String;)Landroid/content/res/AssetFileDescriptor;"),
            &[JValue::Object(&jname)],
        )
        .and_then(|v| v.l())
        .map_err(|_| format!("no fd for asset {name}"))?;
    let result = (|| -> Result<SharedBacking, String> {
        let pfd = env
            .call_method(
                &afd,
                jni_str!("getParcelFileDescriptor"),
                jni_sig!("()Landroid/os/ParcelFileDescriptor;"),
                &[],
            )
            .and_then(|v| v.l())
            .map_err(|_| "pfd".to_string())?;
        let fd = env
            .call_method(&pfd, jni_str!("getFd"), jni_sig!("()I"), &[])
            .and_then(|v| v.i())
            .map_err(|_| "fd".to_string())?;
        let start = env
            .call_method(&afd, jni_str!("getStartOffset"), jni_sig!("()J"), &[])
            .and_then(|v| v.j())
            .map_err(|_| "start".to_string())?;
        let length = env
            .call_method(&afd, jni_str!("getLength"), jni_sig!("()J"), &[])
            .and_then(|v| v.j())
            .map_err(|_| "length".to_string())?;
        if fd < 0 || start < 0 || length <= 0 || length > 512 * 1024 * 1024 {
            return Err("invalid model asset extent".into());
        }
        let prefix = (start as usize) % 4096;
        let extent = length as usize + prefix;
        // The fd stays owned by the Java side; the mapping persists
        // after it closes, so borrow it without taking ownership.
        let file = std::mem::ManuallyDrop::new(unsafe {
            <std::fs::File as std::os::unix::io::FromRawFd>::from_raw_fd(fd)
        });
        let map = unsafe {
            memmap2::MmapOptions::new()
                .offset((start as usize - prefix) as u64)
                .len(extent)
                .map(&*file)
                .map_err(|_| "mmap failed".to_string())?
        };
        Ok(std::sync::Arc::new(crate::model::Backing::from_mmap(
            map,
            prefix,
            length as usize,
        )))
    })();
    let _ = env.call_method(&afd, jni_str!("close"), jni_sig!("()V"), &[]);
    result
}

fn decode_zstd(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|_| "zstd frame".to_string())?;
    let mut out = Vec::with_capacity(260 * 1024 * 1024);
    decoder.read_to_end(&mut out).map_err(|_| "zstd decode".to_string())?;
    Ok(out)
}

fn triage(raw: SharedBacking) -> Result<SharedBacking, String> {
    let bytes = raw.as_slice();
    if bytes.starts_with(b"TRMED001") || bytes.starts_with(b"TRMED002") {
        return Ok(raw);
    }
    if bytes.len() >= 4 && bytes[..4] == [0x28, 0xb5, 0x2f, 0xfd] {
        return Ok(std::sync::Arc::new(crate::model::Backing::from_bytes(
            decode_zstd(bytes)?,
        )));
    }
    Err("unrecognized model asset format".into())
}

fn load_model<'a>(env: &mut Env<'a>, assets: &JObject<'a>) -> Result<SharedBacking, String> {
    for name in [MODEL_PLAIN, MODEL_ZST] {
        if let Ok(backing) = mmap_asset(env, assets, name) {
            return triage(backing);
        }
    }
    for name in [MODEL_PLAIN, MODEL_ZST] {
        if let Ok(raw) = read_asset_stream(env, assets, name) {
            return triage(std::sync::Arc::new(crate::model::Backing::from_bytes(raw)));
        }
    }
    Err("standalone model is missing".into())
}

struct StreamState {
    stream: Stream,
    previews: bool,
    delivered: usize,
    has_audio: bool,
    samples_since_analysis: usize,
    analysis_was_deferred: bool,
    analysis_interval_samples: usize,
}

impl StreamState {
    fn new_stream(&mut self) {
        // engine.cpp new_stream(): ts_reset(stream, 1) -- the stream's
        // inner VAD state is KEPT across phrases in one session.
        self.stream.reset(true);
        self.delivered = 0;
        self.has_audio = false;
        self.samples_since_analysis = 0;
        self.analysis_was_deferred = false;
    }
}

struct Engine {
    state: Mutex<StreamState>,
    vad: Mutex<OuterVad>,
    cancelled: AtomicBool,
    pause: (Mutex<bool>, Condvar),
    /// Keeps the model mapping alive for the engine's lifetime (the
    /// Model also holds it; this is belt-and-braces for the mapping's
    /// extent). Its pages are the shared OS page cache -- lazily
    /// faulted and reclaimable, not a private resident copy.
    _backing: SharedBacking,
}

impl Engine {
    fn wait(&self) -> bool {
        let (lock, cvar) = &self.pause;
        let mut paused = lock.lock().unwrap();
        while *paused && !self.cancelled.load(Ordering::SeqCst) {
            paused = cvar.wait(paused).unwrap();
        }
        !self.cancelled.load(Ordering::SeqCst)
    }
}

fn engine<'a>(handle: jlong) -> Result<&'a Engine, String> {
    if handle == 0 {
        return Err("speech engine is closed".into());
    }
    Ok(unsafe { &*(handle as *const Engine) })
}

fn strings<'a>(env: &mut Env<'a>, texts: &[String]) -> Result<JObjectArray<'a>, String> {
    for t in texts {
        if t.len() > 16000 {
            return Err("recognition result too long".into());
        }
    }
    let class = env.find_class(jni_str!("[B")).map_err(|_| "class".to_string())?;
    let array = env
        .new_object_array(texts.len() as i32, &class, JObject::null())
        .map_err(|_| "array".to_string())?;
    for (i, t) in texts.iter().enumerate() {
        let item = env.new_byte_array(t.len()).map_err(|_| "item".to_string())?;
        let bytes: &[i8] = unsafe { std::slice::from_raw_parts(t.as_ptr() as *const i8, t.len()) };
        env.set_byte_array_region(&item, 0, bytes).map_err(|_| "region".to_string())?;
        env.set_object_array_element(&array, i, &item).map_err(|_| "element".to_string())?;
    }
    Ok(array)
}

fn create_impl<'a>(env: &mut Env<'a>, assets: &JObject<'a>, previews: bool) -> Result<jlong, String> {
    let backing = load_model(env, assets)?;
    let tokenizer = read_asset_stream(env, assets, TOKENIZER)?;
    if tokenizer.len() != 249_974 {
        return Err("invalid tokenizer".into());
    }
    let stream = Stream::from_backing(backing.clone(), &tokenizer)?;
    let vad_bytes = read_asset_stream(env, assets, VAD_MODEL)?;
    let vad = OuterVad::from_bytes(&vad_bytes)?;
    let mut state = StreamState {
        stream,
        previews,
        delivered: 0,
        has_audio: false,
        samples_since_analysis: 0,
        analysis_was_deferred: false,
        analysis_interval_samples: if previews { 16000 } else { 8000 },
    };
    state.new_stream();
    let engine = Engine {
        state: Mutex::new(state),
        vad: Mutex::new(vad),
        cancelled: AtomicBool::new(false),
        pause: (Mutex::new(false), Condvar::new()),
        _backing: backing,
    };
    Ok(Box::into_raw(Box::new(engine)) as jlong)
}

fn stream_impl<'a>(
    env: &mut Env<'a>,
    handle: jlong,
    input: &JFloatArray<'a>,
    finish: bool,
    defer: bool,
) -> Result<Vec<String>, String> {
    let value = engine(handle)?;
    if !value.wait() {
        return Ok(vec![String::new()]);
    }
    let count = env.get_array_length(input).map_err(|_| "length".to_string())?;
    if !(0..=8192).contains(&count) {
        return Err("invalid audio chunk".into());
    }
    let mut samples = vec![0.0f32; count as usize];
    if count > 0 {
        env.get_float_array_region(input, 0, &mut samples)
            .map_err(|_| "region".to_string())?;
    }
    let mut state = value.state.lock().unwrap();
    if count > 0 {
        state.stream.push(&samples)?;
        state.has_audio = true;
        state.samples_since_analysis += count as usize;
    }
    if defer {
        state.analysis_was_deferred = true;
    }
    if !finish && defer && state.samples_since_analysis < 32768 {
        return Ok(Vec::new());
    }
    if finish && state.has_audio {
        // The C++ drain: 8,192 silent samples force a final pass over
        // every real sample. Recognizer only -- never the outer VAD or
        // the listening clock.
        state.stream.push(&[0.0f32; 8192])?;
    }
    if finish || state.samples_since_analysis >= state.analysis_interval_samples {
        let previews = state.previews && !state.analysis_was_deferred;
        state.stream.update(finish, previews)?;
        state.samples_since_analysis = 0;
        state.analysis_was_deferred = false;
    }
    if value.cancelled.load(Ordering::SeqCst) {
        return Ok(vec![String::new()]);
    }
    let total = state.stream.segment_count();
    if total > 2000 {
        return Err("too many speech segments".into());
    }
    let mut result: Vec<String> = Vec::new();
    let mut preview = String::new();
    for i in state.delivered..total {
        let text = state
            .stream
            .segment_text(i)
            .ok_or("missing speech segment")?
            .to_str()
            .map_err(|_| "segment text".to_string())?
            .to_owned();
        if state.stream.segment_complete(i) {
            result.push(text);
            state.delivered = i + 1;
        } else {
            preview = text;
        }
    }
    // The final array entry is always the revisable phrase.
    result.push(preview);
    if finish {
        state.new_stream();
    }
    Ok(result)
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeCreate<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    assets: JObject<'caller>,
    previews: jboolean,
) -> jlong {
    unowned
        .with_env(|env| -> Result<jlong, jni::errors::Error> {
            Ok(match create_impl(env, &assets, previews == JNI_TRUE) {
                Ok(handle) => handle,
                Err(_) => {
                    fail(env, "Could not initialize on-device recognition");
                    0
                }
            })
        })
        .resolve::<jni::errors::ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeVad<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    handle: jlong,
    input: JFloatArray<'caller>,
) -> jfloat {
    unowned
        .with_env(|env| -> Result<jfloat, jni::errors::Error> {
            let run = (|| -> Result<jfloat, String> {
                let value = engine(handle)?;
                if env.get_array_length(&input).map_err(|_| "length".to_string())? != 512 {
                    return Err("invalid audio frame".into());
                }
                let mut samples = [0.0f32; 512];
                env.get_float_array_region(&input, 0, &mut samples)
                    .map_err(|_| "region".to_string())?;
                let probability = value.vad.lock().unwrap().probability(&samples)?;
                if !probability.is_finite() {
                    return Err("invalid voice probability".into());
                }
                Ok(probability)
            })();
            Ok(match run {
                Ok(p) => p,
                Err(_) => {
                    fail(env, "Voice detection failed");
                    0.0
                }
            })
        })
        .resolve::<jni::errors::ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeStream<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    handle: jlong,
    input: JFloatArray<'caller>,
    finish_phrase: jboolean,
    defer_analysis: jboolean,
) -> JObjectArray<'caller> {
    unowned
        .with_env(|env| -> Result<JObjectArray<'_>, jni::errors::Error> {
            let texts = match stream_impl(
                env,
                handle,
                &input,
                finish_phrase == JNI_TRUE,
                defer_analysis == JNI_TRUE,
            ) {
                Ok(texts) => texts,
                Err(_) => {
                    fail(env, "On-device recognition failed");
                    Vec::new()
                }
            };
            match strings(env, &texts) {
                Ok(array) => Ok(array),
                Err(_) => {
                    fail(env, "On-device recognition failed");
                    Ok(strings(env, &[]).expect("empty array"))
                }
            }
        })
        .resolve::<jni::errors::ThrowRuntimeExAndDefault>()
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeCancel<'caller>(
    _unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let value = unsafe { &*(handle as *const Engine) };
    {
        let _lock = value.pause.0.lock().unwrap();
        value.cancelled.store(true, Ordering::SeqCst);
    }
    value.pause.1.notify_all();
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativePause<'caller>(
    _unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    handle: jlong,
    paused: jboolean,
) {
    if handle == 0 {
        return;
    }
    let value = unsafe { &*(handle as *const Engine) };
    {
        let mut lock = value.pause.0.lock().unwrap();
        *lock = paused == JNI_TRUE;
    }
    value.pause.1.notify_all();
}

#[no_mangle]
pub extern "system" fn Java_dev_soupslurpr_transcribro_speech_NativeEngine_nativeDestroy<'caller>(
    _unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut Engine));
        }
    }
}
