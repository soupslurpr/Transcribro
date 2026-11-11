//! Gate 4: the streaming layer behind the `ts_*` ABI (stream.h), a
//! line-by-line port of stream.cpp. `Sequence` owns the frontend state,
//! the adapter memory, the cross-KV and the decoder cache, and runs the
//! reference update schedule: 1280-sample frontend blocks (paired only
//! when already available), a 16-frame lookahead holdback on previews, a
//! 224-frame rewind window for re-encoding, whole-memory cross-KV
//! refresh, and speculative greedy decoding against the previous token
//! draft. `Stream` owns the inner-VAD segmentation over 512-sample
//! chunks with the 32-probability ring, the 8192-sample lookbehind, and
//! detokenization + UTF-8 sanitization (text.h).
//!
//! The C++ drives the components through the tm_* ABI (each call locking
//! the model mutex); the port calls the same component functions
//! directly, which is unobservable in the outputs.

use crate::decoder::{self, Cache};
use crate::encoder;
use crate::frontend;
use crate::model::Model;
use crate::vad;
use crate::{TmFrontendState, TmVadState};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

thread_local! {
    static TS_LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_ts_error(message: &str) {
    TS_LAST_ERROR.with(|e| {
        *e.borrow_mut() =
            CString::new(message).unwrap_or_else(|_| CString::new("error").unwrap());
    });
}

fn zero_frontend() -> TmFrontendState {
    TmFrontendState {
        sample_buffer: [0.0; 79],
        sample_len: 0,
        conv1: [0.0; 768 * 4],
        conv2: [0.0; 1536 * 4],
        frame_count: 0,
    }
}

fn zero_vad() -> TmVadState {
    TmVadState {
        context: [0.0; 64],
        hidden: [0.0; 128],
        cell: [0.0; 128],
    }
}

fn empty_cstring() -> CString {
    CString::new("").unwrap()
}

// ---------------------------------------------------------------------
// Sequence (stream.cpp)
// ---------------------------------------------------------------------

struct Sequence {
    cache: Cache,
    frontend: TmFrontendState,
    processed: usize,
    emitted: usize,
    features: Vec<f32>,
    memory: Vec<f32>,
    cross_keys: Vec<f32>,
    cross_values: Vec<f32>,
    tokens: Vec<i64>,
    cross_valid: bool,
}

impl Sequence {
    fn new() -> Sequence {
        Sequence {
            cache: Cache::new(),
            frontend: zero_frontend(),
            processed: 0,
            emitted: 0,
            features: Vec::new(),
            memory: Vec::new(),
            cross_keys: Vec::new(),
            cross_values: Vec::new(),
            tokens: Vec::new(),
            cross_valid: false,
        }
    }

    fn reset(&mut self) {
        self.frontend = zero_frontend();
        self.processed = 0;
        self.emitted = 0;
        self.features.clear();
        self.memory.clear();
        self.tokens.clear();
        self.cross_keys.clear();
        self.cross_values.clear();
        self.cross_valid = false;
        self.cache.count = 0;
    }

    fn decode(&mut self, model: &Model, input: &[i64]) -> Result<Vec<i64>, String> {
        let mut predictions = vec![0i64; input.len()];
        decoder::decoder_cached(
            model,
            &mut self.cache,
            input,
            Some(&self.cross_keys),
            Some(&self.cross_values),
            self.memory.len() / 640,
            &mut predictions,
        )?;
        Ok(predictions)
    }

    fn update(
        &mut self,
        model: &Model,
        audio: &[f32],
        final_: bool,
        previews: bool,
    ) -> Result<(), String> {
        let is_new = self.processed == 0;
        if self.processed < audio.len() {
            let mut output = [0.0f32; 8 * 768];
            while self.processed + 1280 <= audio.len() {
                let blocks = if audio.len() - self.processed >= 2560 {
                    2
                } else {
                    1
                };
                let _ptf = crate::prof::Timer::start();
                let count = frontend::frontend(
                    model,
                    &audio[self.processed..self.processed + blocks * 1280],
                    &mut self.frontend,
                    &mut output,
                    8,
                )?;
                _ptf.stop(&crate::prof::FRONTEND_NS);
                if count != blocks * 4 {
                    return Err("frontend block shape".into());
                }
                self.features.extend_from_slice(&output[..count * 768]);
                self.processed += blocks * 1280;
            }
            if final_ {
                self.emitted = 0;
                self.memory.clear();
                self.cross_valid = false;
            }
            let total = self.features.len() / 768;
            let stable = if final_ {
                total
            } else if total > 16 {
                total - 16
            } else {
                0
            };
            if stable > self.emitted {
                let start = if self.emitted > 224 {
                    self.emitted - 224
                } else {
                    0
                };
                let encoded =
                    encoder::encoder(model, &self.features[start * 768..total * 768], total - start)?;
                let _pta = crate::prof::Timer::start();
                let added = encoder::adapter(
                    model,
                    &encoded[(self.emitted - start) * 768..(stable - start) * 768],
                    stable - self.emitted,
                    self.emitted,
                )?;
                _pta.stop(&crate::prof::ADAPTER_NS);
                self.memory.extend_from_slice(&added);
                crate::scratch::put_f32(encoded);
                crate::scratch::put_f32(added);
                self.emitted = stable;
                self.cross_valid = false;
            }
        }
        if self.memory.is_empty() || (!final_ && !previews) {
            return Ok(());
        }
        if !self.cross_valid {
            let _ptx = crate::prof::Timer::start();
            let (k, v) = encoder::cross(model, &self.memory, self.memory.len() / 640)?;
            _ptx.stop(&crate::prof::CROSS_NS);
            crate::scratch::put_f32(std::mem::take(&mut self.cross_keys));
            crate::scratch::put_f32(std::mem::take(&mut self.cross_values));
            self.cross_keys = k;
            self.cross_values = v;
            self.cross_valid = true;
        }
        let budget = ((audio.len() as f32 / 16000.0f32)
            * if final_ { 20.0f32 } else { 6.5f32 })
        .ceil() as usize;
        let budget = budget.min(256);
        self.cache.count = 0;
        let mut draft: Vec<i64> = Vec::new();
        for &token in &self.tokens {
            if token != 1 && token != 2 && draft.len() < budget {
                draft.push(token);
            }
        }
        if !draft.is_empty() && !is_new {
            let mut input = vec![1i64];
            input.extend_from_slice(&draft);
            let predictions = self.decode(model, &input)?;
            let mut accepted = 0;
            while accepted < draft.len() && predictions[accepted] == draft[accepted] {
                accepted += 1;
            }
            self.tokens = vec![1];
            self.tokens.extend_from_slice(&draft[..accepted]);
            let mut next;
            if accepted == draft.len() {
                next = *predictions.last().unwrap();
            } else {
                self.cache.count = 0;
                let tokens = self.tokens.clone();
                next = *self.decode(model, &tokens)?.last().unwrap();
            }
            while next != 2 && self.tokens.len() - 1 < budget {
                self.tokens.push(next);
                next = *self.decode(model, &[next])?.last().unwrap();
            }
        } else {
            self.tokens = vec![1];
            for _ in 0..budget {
                let last = *self.tokens.last().unwrap();
                let next = *self.decode(model, &[last])?.last().unwrap();
                self.tokens.push(next);
                if next == 2 {
                    break;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Text helpers (stream.cpp detokenize + text.h valid_text)
// ---------------------------------------------------------------------

fn trim_bytes(text: &[u8]) -> &[u8] {
    fn is_set(b: u8) -> bool {
        matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c)
    }
    let mut first = 0;
    let mut last = text.len();
    while first < last && is_set(text[first]) {
        first += 1;
    }
    while last > first && is_set(text[last - 1]) {
        last -= 1;
    }
    &text[first..last]
}

/// text.h `valid_text`: keep valid UTF-8 sequences unchanged, replace
/// every invalid byte with '?' (one byte at a time).
fn valid_text(source: &[u8]) -> String {
    let mut output: Vec<u8> = Vec::new();
    let mut at = 0;
    while at < source.len() {
        let c = source[at];
        let length = if c < 0x80 {
            1
        } else if (0xc2..=0xdf).contains(&c) {
            2
        } else if (0xe0..=0xef).contains(&c) {
            3
        } else if (0xf0..=0xf4).contains(&c) {
            4
        } else {
            0
        };
        let mut valid = c != 0 && length != 0 && length <= source.len() - at;
        if valid {
            for i in 1..length {
                if source[at + i] & 0xc0 != 0x80 {
                    valid = false;
                    break;
                }
            }
        }
        if valid && length >= 3 {
            let second = source[at + 1];
            valid = !(c == 0xe0 && second < 0xa0)
                && !(c == 0xed && second >= 0xa0)
                && !(c == 0xf0 && second < 0x90)
                && !(c == 0xf4 && second >= 0x90);
        }
        if valid {
            output.extend_from_slice(&source[at..at + length]);
            at += length;
        } else {
            output.push(b'?');
            at += 1;
        }
    }
    String::from_utf8(output).unwrap_or_default()
}

fn parse_tokenizer(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut vocabulary = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let mut length = bytes[at] as usize;
        at += 1;
        if length >= 128 {
            if at >= bytes.len() {
                return Err("tokenizer length truncated".into());
            }
            length = length - 128 + bytes[at] as usize * 128;
            at += 1;
        }
        if length > bytes.len() - at {
            return Err("tokenizer entry extent".into());
        }
        vocabulary.push(bytes[at..at + length].to_vec());
        at += length;
    }
    if vocabulary.len() != 32768 {
        return Err("tokenizer vocabulary size".into());
    }
    Ok(vocabulary)
}

// ---------------------------------------------------------------------
// Stream (stream.cpp)
// ---------------------------------------------------------------------

struct Segment {
    audio: Vec<f32>,
    complete: bool,
    text: CString,
}

pub struct Stream {
    model: Model,
    sequence: Sequence,
    vocabulary: Vec<Vec<u8>>,
    vad: TmVadState,
    probabilities: [f32; 32],
    lookbehind: [f32; 8192],
    pending: Vec<f32>,
    segments: Vec<Segment>,
    probability_index: usize,
    processed: usize,
    last_end: usize,
    sequence_index: usize,
    active: bool,
    finished: bool,
    failed: bool,
    result: CString,
}

impl Stream {
    fn assemble(model: Model, tokenizer_bytes: &[u8]) -> Result<Stream, String> {
        let vocabulary = parse_tokenizer(tokenizer_bytes)?;
        Ok(Stream {
            model,
            sequence: Sequence::new(),
            vocabulary,
            vad: zero_vad(),
            probabilities: [0.0; 32],
            lookbehind: [0.0; 8192],
            pending: Vec::new(),
            segments: Vec::new(),
            probability_index: 0,
            processed: 0,
            last_end: 0,
            sequence_index: usize::MAX,
            active: false,
            finished: false,
            failed: false,
            result: empty_cstring(),
        })
    }

    pub(crate) fn from_backing(
        backing: std::sync::Arc<crate::model::Backing>,
        tokenizer: &[u8],
    ) -> Result<Stream, String> {
        let model = Model::from_backing(backing)?;
        Stream::assemble(model, tokenizer)
    }

    fn from_paths(archive: &str, tokenizer: &str) -> Result<Stream, String> {
        let file = std::fs::File::open(archive)
            .map_err(|_| "cannot open model archive".to_string())?;
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|_| "cannot map model archive".to_string())?;
        let len = map.len();
        let backing = std::sync::Arc::new(crate::model::Backing::from_mmap(map, 0, len));
        let model = Model::from_backing(backing)?;
        let metadata =
            std::fs::metadata(tokenizer).map_err(|_| "tokenizer extent".to_string())?;
        if metadata.len() == 0 || metadata.len() >= 1024 * 1024 {
            return Err("tokenizer extent".into());
        }
        let bytes = std::fs::read(tokenizer).map_err(|_| "tokenizer read".to_string())?;
        Stream::assemble(model, &bytes)
    }

    pub(crate) fn from_memory(archive: &[u8], tokenizer: &[u8]) -> Result<Stream, String> {
        let model = Model::from_bytes(archive)?;
        Stream::assemble(model, tokenizer)
    }

    fn detokenize(&self, tokens: &[i64]) -> Result<String, String> {
        let mut text: Vec<u8> = Vec::new();
        for &token in tokens {
            if token < 0 || token as usize >= self.vocabulary.len() {
                return Err("invalid decoded token".into());
            }
            let piece = &self.vocabulary[token as usize];
            if piece.is_empty() {
                return Err("empty decoded token".into());
            }
            if piece.len() > 2 && piece[0] == b'<' && piece[piece.len() - 1] == b'>' {
                continue;
            }
            text.extend_from_slice(piece);
        }
        // Replace every U+2581 byte triple with one space; the search
        // resumes one byte past each replacement.
        let mut at = 0;
        while at + 3 <= text.len() {
            if text[at] == 0xe2 && text[at + 1] == 0x96 && text[at + 2] == 0x81 {
                text[at] = b' ';
                text.drain(at + 1..at + 3);
                at += 1;
            } else {
                at += 1;
            }
        }
        Ok(valid_text(trim_bytes(&text)))
    }

    pub(crate) fn reset(&mut self, keep_vad: bool) {
        if !keep_vad {
            self.vad = zero_vad();
        }
        self.probabilities = [0.0; 32];
        self.lookbehind = [0.0; 8192];
        self.pending.clear();
        self.segments.clear();
        self.probability_index = 0;
        self.processed = 0;
        self.last_end = 0;
        self.sequence_index = usize::MAX;
        self.active = false;
        self.finished = false;
        self.failed = false;
        self.result = empty_cstring();
        self.sequence.reset();
    }

    pub(crate) fn push(&mut self, audio: &[f32]) -> Result<(), String> {
        let _t_ingest = crate::prof::Timer::start();
        if self.finished || self.failed {
            return Err("stream is not accepting audio".into());
        }
        let samples = audio.len();
        if samples > 64000 || self.pending.len() > 64000 - samples {
            return Err("pending audio limit; update stream first".into());
        }
        for &x in audio {
            if !x.is_finite() || x.abs() > 16.0 {
                return Err("invalid PCM sample".into());
            }
        }
        self.pending.extend_from_slice(audio);
        _t_ingest.stop(&crate::prof::INGEST_NS);
        Ok(())
    }

    pub(crate) fn update(&mut self, final_: bool, previews: bool) -> Result<(), String> {
        let _big_cores = crate::affinity::big_cores();
        let _t_stream = crate::prof::Timer::start();
        crate::prof::add(&crate::prof::STREAM_CALLS, 1);
        if self.finished || self.failed {
            return Err("stream is finished or failed".into());
        }
        let result = match self.update_impl(final_, previews) {
            Ok(()) => Ok(()),
            Err(message) => {
                self.failed = true;
                Err(message)
            }
        };
        _t_stream.stop(&crate::prof::STREAM_NS);
        result
    }

    pub(crate) fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub(crate) fn segment_text(&self, index: usize) -> Option<&std::ffi::CStr> {
        self.segments.get(index).map(|s| s.text.as_c_str())
    }

    pub(crate) fn segment_complete(&self, index: usize) -> bool {
        self.segments.get(index).map(|s| s.complete).unwrap_or(false)
    }

    fn update_impl(&mut self, final_: bool, previews: bool) -> Result<(), String> {
        if final_ && self.pending.len() % 512 != 0 {
            let padded = (self.pending.len() + 511) / 512 * 512;
            self.pending.resize(padded, 0.0);
        }
        let count = self.pending.len() / 512 * 512;
        let mut updated: Vec<usize> = Vec::new();
        let mark = |updated: &mut Vec<usize>, index: usize| {
            if updated.last() != Some(&index) {
                updated.push(index);
            }
        };
        let mut at = 0;
        while at < count {
            if self.processed > 16000 * 60 * 60 {
                return Err("research session duration limit".into());
            }
            let chunk: [f32; 512] = self.pending[at..at + 512].try_into().unwrap();
            self.processed += 512;
            self.lookbehind.copy_within(512..8192, 0);
            self.lookbehind[8192 - 512..].copy_from_slice(&chunk);
            let _ptv = crate::prof::Timer::start();
            let probability = vad::vad(&self.model, &chunk, &mut self.vad)?;
            _ptv.stop(&crate::prof::VAD_NS);
            self.probabilities[self.probability_index] = probability;
            self.probability_index = (self.probability_index + 1) % 32;
            let mut total = 0.0f32;
            for value in self.probabilities {
                total += value;
            }
            let mut smoothed = total / 32.0f32;
            let length = if self.active {
                self.segments.last().unwrap().audio.len()
            } else {
                0
            };
            if length > 160000 {
                smoothed *= (length - 160000) as f32 / 160000.0f32;
            }
            let voice = smoothed > 0.3f32;
            if voice && !self.active {
                if self.segments.len() >= 4096 {
                    return Err("segment limit".into());
                }
                let size = (self.processed - self.last_end).min(8192);
                let audio = self.lookbehind[8192 - size..].to_vec();
                self.segments.push(Segment {
                    audio,
                    complete: false,
                    text: empty_cstring(),
                });
                mark(&mut updated, self.segments.len() - 1);
            } else if self.active {
                let index = self.segments.len() - 1;
                let segment = self.segments.last_mut().unwrap();
                if segment.audio.len() > 320000 - 512 {
                    return Err("utterance duration limit".into());
                }
                segment.audio.extend_from_slice(&chunk);
                segment.complete = !voice;
                mark(&mut updated, index);
                if !voice {
                    self.last_end = self.processed;
                }
            }
            self.active = voice;
            at += 512;
        }
        self.pending.drain(..count);
        if final_ && self.active {
            self.segments.last_mut().unwrap().complete = true;
            mark(&mut updated, self.segments.len() - 1);
            self.active = false;
        }
        for index in updated {
            if self.sequence_index != index {
                self.sequence.reset();
                self.sequence_index = index;
            }
            let complete = self.segments[index].complete;
            self.sequence
                .update(&self.model, &self.segments[index].audio, complete, previews)?;
            let text = self.detokenize(&self.sequence.tokens)?;
            let segment = &mut self.segments[index];
            segment.text = CString::new(text).unwrap_or_default();
            if segment.complete {
                segment.audio = Vec::new();
            }
        }
        let mut result: Vec<u8> = Vec::new();
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                result.push(b' ');
            }
            result.extend_from_slice(segment.text.as_bytes());
        }
        self.result = CString::new(trim_bytes(&result).to_vec()).unwrap_or_default();
        self.finished = final_;
        Ok(())
    }
}

// ---------------------------------------------------------------------
// ts_* ABI
// ---------------------------------------------------------------------

fn ts_guard(handle: *mut c_void, body: impl FnOnce(&mut Stream) -> Result<(), String>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null() {
            return Err("null stream".into());
        }
        let stream = unsafe { &mut *(handle as *mut Stream) };
        body(stream)
    }));
    match result {
        Ok(Ok(())) => {
            set_ts_error("");
            0
        }
        Ok(Err(message)) => {
            set_ts_error(&message);
            1
        }
        Err(_) => {
            set_ts_error("rust engine panic");
            1
        }
    }
}

fn create_result(
    result: Result<Stream, String>,
    error: *mut c_char,
    size: usize,
) -> *mut c_void {
    match result {
        Ok(stream) => Box::into_raw(Box::new(stream)) as *mut c_void,
        Err(message) => {
            crate::write_error_buffer(error, size, &message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn ts_create(
    archive: *const c_char,
    tokenizer: *const c_char,
    error: *mut c_char,
    size: usize,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Stream, String> {
        if archive.is_null() || tokenizer.is_null() {
            return Err("null model sources".into());
        }
        let archive = unsafe { CStr::from_ptr(archive) }
            .to_str()
            .map_err(|_| "invalid archive path".to_string())?;
        let tokenizer = unsafe { CStr::from_ptr(tokenizer) }
            .to_str()
            .map_err(|_| "invalid tokenizer path".to_string())?;
        Stream::from_paths(archive, tokenizer)
    }));
    match result {
        Ok(result) => create_result(result, error, size),
        Err(_) => {
            crate::write_error_buffer(error, size, "rust engine panic");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn ts_create_memory(
    archive: *const c_void,
    archive_size: usize,
    tokenizer: *const c_void,
    tokenizer_size: usize,
    error: *mut c_char,
    error_size: usize,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Stream, String> {
        if archive.is_null() || tokenizer.is_null() {
            return Err("null model sources".into());
        }
        if !(64..=1024 * 1024 * 1024).contains(&archive_size) || (archive as usize) % 64 != 0 {
            return Err("archive memory extent or alignment".into());
        }
        if tokenizer_size == 0 || tokenizer_size >= 1024 * 1024 {
            return Err("tokenizer memory extent".into());
        }
        let archive = unsafe { std::slice::from_raw_parts(archive as *const u8, archive_size) };
        let tokenizer =
            unsafe { std::slice::from_raw_parts(tokenizer as *const u8, tokenizer_size) };
        Stream::from_memory(archive, tokenizer)
    }));
    match result {
        Ok(result) => create_result(result, error, error_size),
        Err(_) => {
            crate::write_error_buffer(error, error_size, "rust engine panic");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn ts_destroy(stream: *mut c_void) {
    crate::prof::dump();
    if !stream.is_null() {
        unsafe {
            drop(Box::from_raw(stream as *mut Stream));
        }
    }
}

#[no_mangle]
pub extern "C" fn ts_push(stream: *mut c_void, audio: *const f32, count: usize) -> i32 {
    ts_guard(stream, |s| {
        if audio.is_null() && count > 0 {
            return Err("stream is not accepting audio".into());
        }
        let audio = if audio.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(audio, count) }
        };
        s.push(audio)
    })
}

#[no_mangle]
pub extern "C" fn ts_update(stream: *mut c_void, final_: i32) -> i32 {
    ts_guard(stream, |s| s.update(final_ != 0, true))
}

#[no_mangle]
pub extern "C" fn ts_update_ex(stream: *mut c_void, final_: i32, previews: i32) -> i32 {
    ts_guard(stream, |s| s.update(final_ != 0, previews != 0))
}

#[no_mangle]
pub extern "C" fn ts_segment_count(stream: *mut c_void) -> usize {
    if stream.is_null() {
        return 0;
    }
    unsafe { (*(stream as *const Stream)).segments.len() }
}

#[no_mangle]
pub extern "C" fn ts_segment_text(stream: *mut c_void, index: usize) -> *const c_char {
    if stream.is_null() {
        return ptr::null();
    }
    let stream = unsafe { &*(stream as *const Stream) };
    match stream.segments.get(index) {
        Some(segment) => segment.text.as_ptr(),
        None => ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn ts_segment_complete(stream: *mut c_void, index: usize) -> i32 {
    if stream.is_null() {
        return 0;
    }
    let stream = unsafe { &*(stream as *const Stream) };
    match stream.segments.get(index) {
        Some(segment) => segment.complete as i32,
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn ts_reset(stream: *mut c_void, keep_vad: i32) -> i32 {
    ts_guard(stream, |s| {
        s.reset(keep_vad != 0);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn ts_text(stream: *mut c_void) -> *const c_char {
    if stream.is_null() {
        return b"\0".as_ptr() as *const c_char;
    }
    unsafe { (*(stream as *const Stream)).result.as_ptr() }
}

#[no_mangle]
pub extern "C" fn ts_error() -> *const c_char {
    TS_LAST_ERROR.with(|e| e.borrow().as_ptr())
}
