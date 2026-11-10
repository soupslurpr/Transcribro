//! Rust component engine behind the `tm_*` standalone C ABI (medium.h).
//! Gates 2-4: archive parsing, frontend, inner VAD, encoder, adapter,
//! cross-KV, decoder and the `ts_*` streaming layer are implemented.

mod affinity;
mod archive;
mod decoder;
mod encoder;
mod frontend;
#[cfg(target_os = "android")]
mod jni_bridge;
mod kernels;
mod model;
mod outer_vad;
mod pool;
mod prof;
mod scratch;
mod stream;
mod vad;

pub use outer_vad::OuterVad;

use model::{Backing, Model};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

#[repr(C)]
pub struct TmFrontendState {
    pub sample_buffer: [f32; 79],
    pub sample_len: i64,
    pub conv1: [f32; 768 * 4],
    pub conv2: [f32; 1536 * 4],
    pub frame_count: i64,
}

#[repr(C)]
pub struct TmVadState {
    pub context: [f32; 64],
    pub hidden: [f32; 128],
    pub cell: [f32; 128],
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(message: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(message).unwrap_or_else(|_| CString::new("error").unwrap());
    });
}
fn clear_error() {
    set_error("");
}

pub(crate) fn write_error_buffer(error: *mut c_char, bytes: usize, message: &str) {
    if error.is_null() || bytes == 0 {
        return;
    }
    let raw = message.as_bytes();
    let n = raw.len().min(bytes - 1);
    unsafe {
        ptr::copy_nonoverlapping(raw.as_ptr(), error as *mut u8, n);
        *error.add(n) = 0;
    }
}

fn guard(body: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            clear_error();
            0
        }
        Ok(Err(message)) => {
            set_error(&message);
            1
        }
        Err(_) => {
            set_error("rust engine panic");
            1
        }
    }
}

fn model_ref<'a>(handle: *mut c_void) -> Result<&'a Model, String> {
    if handle.is_null() {
        return Err("null model".into());
    }
    Ok(unsafe { &*(handle as *const Model) })
}

#[no_mangle]
pub extern "C" fn tm_create(
    path: *const c_char,
    error: *mut c_char,
    bytes: usize,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Model, String> {
        let _big_cores = crate::affinity::big_cores();
        if path.is_null() {
            return Err("null archive path".into());
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|_| "invalid archive path".to_string())?;
        // Gate 6: map the archive file instead of reading it; a
        // TRMED002 model then borrows its weights from the mapping.
        let file = std::fs::File::open(path)
            .map_err(|_| "cannot open model archive".to_string())?;
        match unsafe { memmap2::Mmap::map(&file) } {
            Ok(map) => {
                let len = map.len();
                Model::from_backing(std::sync::Arc::new(Backing::from_mmap(map, 0, len)))
            }
            Err(_) => {
                let data = std::fs::read(path)
                    .map_err(|_| "cannot open model archive".to_string())?;
                Model::from_bytes(&data)
            }
        }
    }));
    match result {
        Ok(Ok(model)) => Box::into_raw(Box::new(model)) as *mut c_void,
        Ok(Err(message)) => {
            write_error_buffer(error, bytes, &message);
            ptr::null_mut()
        }
        Err(_) => {
            write_error_buffer(error, bytes, "rust engine panic");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tm_create_memory(
    archive: *const c_void,
    size: usize,
    error: *mut c_char,
    error_bytes: usize,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Model, String> {
        if archive.is_null() {
            return Err("null archive memory".into());
        }
        if !(64..=1024 * 1024 * 1024).contains(&size) || (archive as usize) % 64 != 0 {
            return Err("archive memory extent or alignment".into());
        }
        let data = unsafe { std::slice::from_raw_parts(archive as *const u8, size) };
        Model::from_bytes(data)
    }));
    match result {
        Ok(Ok(model)) => Box::into_raw(Box::new(model)) as *mut c_void,
        Ok(Err(message)) => {
            write_error_buffer(error, error_bytes, &message);
            ptr::null_mut()
        }
        Err(_) => {
            write_error_buffer(error, error_bytes, "rust engine panic");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tm_destroy(model: *mut c_void) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model as *mut Model));
        }
    }
}

#[no_mangle]
pub extern "C" fn tm_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn tm_frontend(
    model: *mut c_void,
    audio: *const f32,
    samples: usize,
    state: *mut TmFrontendState,
    features: *mut f32,
    capacity: usize,
    frames: *mut usize,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        if state.is_null() {
            return Err("null frontend state".into());
        }
        if audio.is_null() || features.is_null() || frames.is_null() {
            return Err("frontend chunk extent".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let audio = unsafe { std::slice::from_raw_parts(audio, samples) };
        let state = unsafe { &mut *state };
        let output = unsafe { std::slice::from_raw_parts_mut(features, capacity * 768) };
        let produced = frontend::frontend(model, audio, state, output, capacity)?;
        unsafe {
            *frames = produced;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tm_vad(
    model: *mut c_void,
    audio: *const f32,
    state: *mut TmVadState,
    probability: *mut f32,
) -> i32 {
    guard(|| {
        let model = model_ref(model)?;
        if state.is_null() {
            return Err("null VAD state".into());
        }
        if audio.is_null() || probability.is_null() {
            return Err("null VAD input".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let audio = unsafe { std::slice::from_raw_parts(audio, 512) };
        let state = unsafe { &mut *state };
        let p = vad::vad(model, audio, state)?;
        unsafe {
            *probability = p;
        }
        Ok(())
    })
}

// ---- Decoder cache handle ----

pub use decoder::Cache;

#[no_mangle]
pub extern "C" fn tm_cache_create() -> *mut c_void {
    Box::into_raw(Box::new(Cache::new())) as *mut c_void
}

#[no_mangle]
pub extern "C" fn tm_cache_destroy(cache: *mut c_void) {
    if !cache.is_null() {
        unsafe {
            drop(Box::from_raw(cache as *mut Cache));
        }
    }
}

#[no_mangle]
pub extern "C" fn tm_cache_reset(cache: *mut c_void) {
    if !cache.is_null() {
        unsafe {
            (*(cache as *mut Cache)).count = 0;
        }
    }
}

// ---- Decoder (Gate 4) ----

#[no_mangle]
pub extern "C" fn tm_decoder_cached(
    model: *mut c_void,
    cache: *mut c_void,
    tokens: *const i64,
    count: usize,
    cross_keys: *const f32,
    cross_values: *const f32,
    frames: usize,
    predictions: *mut i64,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        if cache.is_null() || predictions.is_null() || count == 0 || count > 257 {
            return Err("cached decoder input".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let cache = unsafe { &mut *(cache as *mut Cache) };
        let tokens = unsafe { std::slice::from_raw_parts(tokens, count) };
        let cross_keys = if cross_keys.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cross_keys, 14 * frames * 640) })
        };
        let cross_values = if cross_values.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cross_values, 14 * frames * 640) })
        };
        let predictions = unsafe { std::slice::from_raw_parts_mut(predictions, count) };
        decoder::decoder_cached(model, cache, tokens, cross_keys, cross_values, frames, predictions)
    })
}

#[no_mangle]
pub extern "C" fn tm_encoder(
    model: *mut c_void,
    input: *const f32,
    frames: usize,
    output: *mut f32,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        if input.is_null() || output.is_null() {
            return Err("null encoder buffer".into());
        }
        if frames == 0 || frames > 1024 {
            return Err("encoder frame limit".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let input = unsafe { std::slice::from_raw_parts(input, frames * 768) };
        let encoded = encoder::encoder(model, input, frames)?;
        let output = unsafe { std::slice::from_raw_parts_mut(output, frames * 768) };
        output.copy_from_slice(&encoded);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tm_adapter(
    model: *mut c_void,
    input: *const f32,
    frames: usize,
    offset: usize,
    output: *mut f32,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        if input.is_null() || output.is_null() {
            return Err("null adapter buffer".into());
        }
        if frames == 0 || frames > 1024 || offset > 4096 || frames > 4096 - offset {
            return Err("adapter extent".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let input = unsafe { std::slice::from_raw_parts(input, frames * 768) };
        let memory = encoder::adapter(model, input, frames, offset)?;
        let output = unsafe { std::slice::from_raw_parts_mut(output, frames * 640) };
        output.copy_from_slice(&memory);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tm_cross(
    model: *mut c_void,
    input: *const f32,
    frames: usize,
    keys: *mut f32,
    values: *mut f32,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        if input.is_null() || keys.is_null() || values.is_null() {
            return Err("null cross-attention buffer".into());
        }
        if frames == 0 || frames > 1024 {
            return Err("cross-attention frame limit".into());
        }
        let _lock = model.mutex.lock().unwrap();
        let input = unsafe { std::slice::from_raw_parts(input, frames * 640) };
        let (k, v) = encoder::cross(model, input, frames)?;
        let keys = unsafe { std::slice::from_raw_parts_mut(keys, 14 * frames * 640) };
        let values = unsafe { std::slice::from_raw_parts_mut(values, 14 * frames * 640) };
        keys.copy_from_slice(&k);
        values.copy_from_slice(&v);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn tm_decoder(
    model: *mut c_void,
    tokens: *const i64,
    count: usize,
    keys: *const f32,
    values: *const f32,
    past: usize,
    cross_keys: *const f32,
    cross_values: *const f32,
    frames: usize,
    logits: *mut f32,
    out_keys: *mut f32,
    out_values: *mut f32,
) -> i32 {
    let _big_cores = crate::affinity::big_cores();
    guard(|| {
        let model = model_ref(model)?;
        let _lock = model.mutex.lock().unwrap();
        let tokens = if tokens.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(tokens, count) })
        };
        let old = if past == 0 || keys.is_null() || values.is_null() {
            None
        } else {
            Some((
                unsafe { std::slice::from_raw_parts(keys, 14 * past * 640) },
                unsafe { std::slice::from_raw_parts(values, 14 * past * 640) },
            ))
        };
        let cross_keys = if cross_keys.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cross_keys, 14 * frames * 640) })
        };
        let cross_values = if cross_values.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(cross_values, 14 * frames * 640) })
        };
        let logits = if logits.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(logits, count * 32768) })
        };
        let out = if out_keys.is_null() || out_values.is_null() {
            None
        } else {
            Some((
                unsafe { std::slice::from_raw_parts_mut(out_keys, 14 * (past + count) * 640) },
                unsafe { std::slice::from_raw_parts_mut(out_values, 14 * (past + count) * 640) },
            ))
        };
        decoder::decoder_uncached(model, tokens, old, past, cross_keys, cross_values, frames, logits, out)
    })
}
