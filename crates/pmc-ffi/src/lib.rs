// SPDX-License-Identifier: GPL-3.0-only

//! A C ABI over the trainer.
//!
//! The Python side does not require this library: the Python reference
//! implementation is the oracle, and the parity test compares the two. This
//! ABI exists so that a deployment can call the Rust trainer without
//! reimplementing it, and so the boundary between the two is explicit.
//!
//! Every entry point returns a JSON string and writes its length to an
//! out-parameter, so no allocation crosses the boundary except the string
//! itself, which the caller frees with `pmc_string_free`.

use std::ffi::{c_char, CStr, CString};

use pmc_train::{train, TrainConfig};

/// Train a model and return a JSON report.
///
/// # Safety
///
/// `out_len` must point to a writable `usize`.
#[no_mangle]
pub unsafe extern "C" fn pmc_train_json(
    seed: u64,
    steps: u32,
    learning_rate: f32,
    hidden: u32,
    keys: u32,
    out_len: *mut usize,
) -> *mut c_char {
    let config = TrainConfig {
        seed,
        steps: steps as usize,
        learning_rate,
        hidden: hidden as usize,
        keys: keys as usize,
    };
    let (_, report) = train(&config);
    let text = report.to_json().to_string();
    into_c_string(text, out_len)
}

/// The version of the artifact format this library produces.
#[no_mangle]
pub extern "C" fn pmc_artifact_format_version() -> u32 {
    pmc_core::ARTIFACT_FORMAT_VERSION
}

/// Free a string returned by this library.
///
/// # Safety
///
/// `text` must have come from this library and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn pmc_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

/// Read a C string, for tests.
#[allow(dead_code)]
pub unsafe fn read_c_string(text: *const c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    CStr::from_ptr(text).to_str().ok().map(|value| value.to_string())
}

unsafe fn into_c_string(text: String, out_len: *mut usize) -> *mut c_char {
    let bytes = text.into_bytes();
    let length = bytes.len();
    let mut buffer: Vec<u8> = bytes;
    buffer.push(0);
    let pointer = buffer.as_mut_ptr() as *mut c_char;
    std::mem::forget(buffer);
    if !out_len.is_null() {
        *out_len = length;
    }
    pointer
}
