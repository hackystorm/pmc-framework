// SPDX-License-Identifier: GPL-3.0-only

//! Fusion of a vocabulary and memory-state delta into a frozen GGUF
//! checkpoint.
//!
//! # Inputs
//!
//! `rows.bin` is the delta embedding block:
//!
//! ```text
//! magic   "PMCR"          4 bytes
//! version u32 little-endian   = 1
//! n_rows  u32 little-endian
//! dim     u32 little-endian
//! data    n_rows * dim f32 little-endian
//! ```
//!
//! The file must be exactly `16 + n_rows * dim * 4` bytes. Zero rows, zero
//! dimensions, truncation, trailing bytes and non-finite values are rejected.
//!
//! `meta.json` names the appended tokens and carries the optional state
//! banks, dream id and provenance list. See [`parse_meta`] for the schema.
//!
//! # Guarantees
//!
//! * Every byte of every base tensor is hashed before the write and compared
//!   with the corresponding range of the fused file afterwards. For
//!   `token_embd.weight` and `output.weight` only the base rows are compared,
//!   because fusion appends rows to them.
//! * The fused file is written to a staging path, verified, and only then
//!   renamed over the requested output path. A failed fusion leaves an
//!   existing output file untouched.
//! * State banks are written as `pmc.state.*` metadata arrays and read back
//!   bit-exact during verification. They add no tensors.
//! * `pmc.version`, `pmc.manifest` and (when present) `pmc.dream_id` are
//!   written into the GGUF metadata, and read back during verification.
//!
//! The manifest records the **file name** of the base and the output, their
//! sizes, their integrity identities and their SHA-256 digests. It never
//! records an absolute path, a user name or a machine-specific location, so a
//! fused artifact can leave this machine with its manifest intact.
//!
//! # llama.cpp compatibility
//!
//! Fused files load in stock llama.cpp. Vocabulary rows are appended to
//! `token_embd.weight` (and `output.weight` when the model has an untied
//! output), and state banks are stored as `pmc.state.*` metadata arrays.
//! Stock loaders ignore unknown metadata and load the file; the PMC runtime
//! reads the arrays back through [`GgufReader`]. Extra tensors are not used:
//! stock llama.cpp requires every tensor to belong to the architecture and
//! aborts with `done_getting_tensors: wrong number of tensors`.
//!
//! # Identity
//!
//! [`tensor_stream_identity`] folds every tensor's name, declared offset,
//! computed byte length and content hash into a 64-bit FNV-1a value together
//! with the file size. It is an **integrity fingerprint, not a cryptographic
//! signature**: a collision is possible in principle and the value must never
//! be used to authenticate a model. The delta bytes and the manifest use
//! SHA-256, which is cryptographic.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use pmc_core::{sha256_hex, Json, Sha256};
use pmc_gguf::{
    quant_geometry, quant_name, GgufReader, GgufValue, GgufWriter, TensorInfo, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
};

/// Magic bytes of the delta embedding block.
pub const DELTA_MAGIC: [u8; 4] = *b"PMCR";
/// Delta format version understood by this crate.
pub const DELTA_VERSION: u32 = 1;
/// Bytes before the first f32 value in a delta file.
pub const DELTA_HEADER_BYTES: usize = 16;
/// Value of the `pmc.version` metadata key written by fusion.
pub const PMC_VERSION: u32 = 1;
/// Version string of the manifest written to disk and into the GGUF.
pub const MANIFEST_FORMAT: &str = "pmc-fuse/1";
/// GGUF token type for user-defined tokens; every appended PMC token uses it.
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// Bytes read per I/O chunk when hashing or copying.
const CHUNK: usize = 4 * 1024 * 1024;
/// Deepest JSON nesting accepted in `meta.json`.
const MAX_JSON_DEPTH: u32 = 32;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Paths and options for one fusion.
#[derive(Clone, Debug)]
pub struct FuseRequest {
    /// Frozen source GGUF.
    pub base: PathBuf,
    /// `rows.bin` delta embedding block.
    pub rows: PathBuf,
    /// `meta.json` token and state description.
    pub meta: PathBuf,
    /// Destination GGUF.
    pub out: PathBuf,
    /// Optional external manifest path.
    pub manifest: Option<PathBuf>,
}

/// One state bank copied from `meta.json` into the fused GGUF.
#[derive(Clone, Debug, PartialEq)]
pub struct StateTensor {
    /// Metadata namespace, for example `pmc.state.bank`.
    pub name: String,
    /// GGUF wire dimensions: `dims[0]` is the row length.
    pub dims: Vec<u64>,
    /// Flat little-endian f32 payload, length equal to the product of `dims`.
    pub data: Vec<f32>,
}

/// One row of the delta block.
#[derive(Clone, Debug)]
pub struct Delta {
    /// Number of appended embedding rows.
    pub n_rows: usize,
    /// Elements per row; must equal the base `token_embd.weight` row length.
    pub dim: usize,
    /// `n_rows` rows of `dim` values.
    pub rows: Vec<Vec<f32>>,
}

/// Parsed `meta.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct FuseMeta {
    /// New token strings, in vocabulary order.
    pub tokens: Vec<String>,
    /// Token scores appended to `tokenizer.ggml.scores`.
    pub scores: Vec<f32>,
    /// Token types appended to `tokenizer.ggml.token_type`.
    pub token_types: Vec<i32>,
    /// Optional f32 state tensors.
    pub state_tensors: Vec<StateTensor>,
    /// Optional dream identifier stored as `pmc.dream_id`.
    pub dream_id: Option<String>,
    /// Optional source digests carried forward from the dream.
    pub provenance: Vec<String>,
}

/// Summary of one state tensor in a report or manifest.
#[derive(Clone, Debug)]
pub struct StateTensorSummary {
    /// Tensor name.
    pub name: String,
    /// GGUF wire dimensions.
    pub dims: Vec<u64>,
    /// Number of f32 elements.
    pub elements: usize,
}

/// Size and streaming identity of a GGUF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileFingerprint {
    /// File size in bytes.
    pub size: u64,
    /// `fnv1a64:<16 hex digits>` identity, see [`tensor_stream_identity`].
    pub identity: String,
}

/// Successful fusion result.
#[derive(Clone, Debug)]
pub struct FuseReport {
    /// Base path as given by the caller (report only, never embedded).
    pub base_path: String,
    /// Base file size.
    pub base_size: u64,
    /// Base streaming identity.
    pub base_identity: String,
    /// SHA-256 of the whole base file.
    pub base_sha256: String,
    /// Vocabulary size of the base.
    pub old_vocab: usize,
    /// Vocabulary size of the fused file.
    pub new_vocab: usize,
    /// Index of the first appended token.
    pub token_first: usize,
    /// Index of the last appended token.
    pub token_last: usize,
    /// Number of appended rows.
    pub rows: usize,
    /// Elements per appended row.
    pub dim: usize,
    /// SHA-256 of the delta file.
    pub rows_sha256: String,
    /// SHA-256 of the meta file.
    pub meta_sha256: String,
    /// Appended state tensors.
    pub state_tensors: Vec<StateTensorSummary>,
    /// Fused path as given by the caller (report only, never embedded).
    pub output_path: String,
    /// Fused file size.
    pub output_size: u64,
    /// Fused streaming identity.
    pub output_identity: String,
    /// SHA-256 of the whole fused file.
    pub output_sha256: String,
    /// Dream identifier, when the delta carried one.
    pub dream_id: Option<String>,
    /// Provenance digests carried forward.
    pub provenance: Vec<String>,
    /// External manifest path, when requested.
    pub manifest_path: Option<String>,
}

impl FuseReport {
    /// Render the report as JSON. Token strings and tensor payloads are never
    /// included: only counts, paths and digests.
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            (
                "base",
                Json::obj(vec![
                    ("path", Json::str(self.base_path.clone())),
                    ("size", Json::num(self.base_size as f64)),
                    ("identity", Json::str(self.base_identity.clone())),
                    ("sha256", Json::str(self.base_sha256.clone())),
                ]),
            ),
            (
                "delta",
                Json::obj(vec![
                    ("rows", Json::num(self.rows as f64)),
                    ("dim", Json::num(self.dim as f64)),
                    ("sha256", Json::str(self.rows_sha256.clone())),
                    ("meta_sha256", Json::str(self.meta_sha256.clone())),
                ]),
            ),
            (
                "tokens",
                Json::obj(vec![
                    ("old_count", Json::num(self.old_vocab as f64)),
                    ("new_count", Json::num(self.new_vocab as f64)),
                    ("first_index", Json::num(self.token_first as f64)),
                    ("last_index", Json::num(self.token_last as f64)),
                ]),
            ),
            ("state_tensors", state_summary_json(&self.state_tensors)),
            (
                "output",
                Json::obj(vec![
                    ("path", Json::str(self.output_path.clone())),
                    ("size", Json::num(self.output_size as f64)),
                    ("identity", Json::str(self.output_identity.clone())),
                    ("sha256", Json::str(self.output_sha256.clone())),
                ]),
            ),
            (
                "dream_id",
                match &self.dream_id {
                    Some(value) => Json::str(value.clone()),
                    None => Json::Null,
                },
            ),
            (
                "provenance",
                Json::Arr(
                    self.provenance
                        .iter()
                        .map(|value| Json::str(value.clone()))
                        .collect(),
                ),
            ),
            (
                "manifest",
                match &self.manifest_path {
                    Some(value) => Json::str(value.clone()),
                    None => Json::Null,
                },
            ),
        ])
    }
}

// ---------------------------------------------------------------------------
// Delta block
// ---------------------------------------------------------------------------

/// Parse and validate a `rows.bin` payload.
///
/// The length must be exactly `16 + n_rows * dim * 4`. All values must be
/// finite; a NaN in an embedding row would be quantized into the model and
/// silently poison inference.
pub fn parse_delta(bytes: &[u8]) -> Result<Delta, String> {
    if bytes.len() < DELTA_HEADER_BYTES {
        return Err(format!(
            "delta is truncated: {} bytes, need at least {}",
            bytes.len(),
            DELTA_HEADER_BYTES
        ));
    }
    if bytes[0..4] != DELTA_MAGIC {
        return Err(format!(
            "delta has bad magic {:02x?}, expected {:?}",
            &bytes[0..4], DELTA_MAGIC
        ));
    }
    let version = read_u32_le(bytes, 4);
    if version != DELTA_VERSION {
        return Err(format!(
            "unsupported delta version {version}, expected {DELTA_VERSION}"
        ));
    }
    let n_rows = read_u32_le(bytes, 8) as usize;
    let dim = read_u32_le(bytes, 12) as usize;
    if n_rows == 0 {
        return Err("delta declares zero rows".to_string());
    }
    if dim == 0 {
        return Err("delta declares zero dimensions".to_string());
    }

    let values = n_rows
        .checked_mul(dim)
        .ok_or_else(|| "delta element count overflows usize".to_string())?;
    let expected = values
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|body| body.checked_add(DELTA_HEADER_BYTES))
        .ok_or_else(|| "delta byte length overflows usize".to_string())?;
    if bytes.len() < expected {
        return Err(format!(
            "delta is truncated: expected {expected} bytes, got {}",
            bytes.len()
        ));
    }
    if bytes.len() > expected {
        return Err(format!(
            "delta has {} extra bytes: expected {expected}, got {}",
            bytes.len() - expected,
            bytes.len()
        ));
    }

    let mut rows = Vec::with_capacity(n_rows);
    let mut cursor = DELTA_HEADER_BYTES;
    for row in 0..n_rows {
        let mut values = Vec::with_capacity(dim);
        for column in 0..dim {
            let value = f32::from_bits(read_u32_le(bytes, cursor));
            if !value.is_finite() {
                return Err(format!(
                    "delta row {row} column {column} is not a finite f32"
                ));
            }
            values.push(value);
            cursor += std::mem::size_of::<f32>();
        }
        rows.push(values);
    }

    Ok(Delta {
        n_rows,
        dim,
        rows,
    })
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

// ---------------------------------------------------------------------------
// meta.json
// ---------------------------------------------------------------------------

/// Parse and validate a `meta.json` payload against a delta row count.
///
/// The top-level object accepts exactly these keys:
///
/// * `tokens`: array of strings, length `n_rows`;
/// * `scores`: array of finite numbers, length `n_rows`;
/// * `token_types`: array of integers, length `n_rows`, each equal to 4
///   (`USER_DEFINED`), because this delta format exists for PMC memory tokens;
/// * `state_tensors`: optional object mapping a state name under
///   `pmc.state.` to `{"dims": [u64...], "data": [f32...]}`. The bank is
///   stored as `pmc.state.*` metadata arrays, not as a tensor. `dims` has 1
///   to 8 positive entries and the product must equal `data.len()`;
/// * `dream_id`: optional non-empty string;
/// * `provenance`: optional array of non-empty strings.
///
/// Unknown keys, duplicate keys and trailing bytes are rejected so a typo in
/// provenance cannot be dropped silently.
pub fn parse_meta(bytes: &[u8], n_rows: usize) -> Result<FuseMeta, String> {
    let document = JsonParser::new(bytes).parse_document()?;
    let entries = json_object(&document, "meta.json")?;

    for (key, _) in entries {
        match key.as_str() {
            "tokens" | "scores" | "token_types" | "state_tensors" | "dream_id" | "provenance" => {}
            other => return Err(format!("unknown meta field {other:?}")),
        }
    }

    let tokens = json_string_array(
        required_field(entries, "tokens", "meta.json")?,
        "tokens",
        true,
    )?;
    let scores = json_f32_array(
        required_field(entries, "scores", "meta.json")?,
        "meta scores",
    )?;
    let token_types = json_token_types(required_field(entries, "token_types", "meta.json")?)?;

    if tokens.len() != n_rows {
        return Err(format!(
            "meta tokens has {} entries, delta has {n_rows} rows",
            tokens.len()
        ));
    }
    if scores.len() != n_rows {
        return Err(format!(
            "meta scores has {} entries, delta has {n_rows} rows",
            scores.len()
        ));
    }
    if token_types.len() != n_rows {
        return Err(format!(
            "meta token_types has {} entries, delta has {n_rows} rows",
            token_types.len()
        ));
    }
    for (index, token_type) in token_types.iter().enumerate() {
        if *token_type != TOKEN_TYPE_USER_DEFINED {
            return Err(format!(
                "meta token_types[{index}] is {token_type}; PMC memory tokens must use type 4 (USER_DEFINED)"
            ));
        }
    }

    let state_tensors = match json_field(entries, "state_tensors") {
        None => Vec::new(),
        Some(value) => {
            let object = json_object(value, "meta state_tensors")?;
            let mut tensors = Vec::with_capacity(object.len());
            for (name, specification) in object {
                tensors.push(parse_state_tensor(name, specification)?);
            }
            tensors
        }
    };

    let dream_id = match json_field(entries, "dream_id") {
        None => None,
        Some(JsonValue::String(value)) if value.is_empty() => {
            return Err("meta dream_id must not be empty".to_string());
        }
        Some(JsonValue::String(value)) => Some(value.clone()),
        Some(_) => return Err("meta dream_id must be a string".to_string()),
    };

    let provenance = match json_field(entries, "provenance") {
        None => Vec::new(),
        Some(value) => json_string_array(value, "meta provenance", false)?,
    };

    Ok(FuseMeta {
        tokens,
        scores,
        token_types,
        state_tensors,
        dream_id,
        provenance,
    })
}

fn parse_state_tensor(name: &str, value: &JsonValue) -> Result<StateTensor, String> {
    if name.is_empty() {
        return Err("meta state_tensors contains an empty tensor name".to_string());
    }
    if !name.starts_with("pmc.state.") {
        return Err(format!(
            "meta state tensor {name:?} must start with pmc.state."
        ));
    }
    let entries = json_object(value, &format!("state_tensors[{name:?}]"))?;
    for (key, _) in entries {
        match key.as_str() {
            "dims" | "data" => {}
            other => {
                return Err(format!(
                    "state_tensors[{name:?}] has unknown field {other:?}"
                ))
            }
        }
    }

    let dims_value = required_field(entries, "dims", &format!("state_tensors[{name:?}]"))?;
    let dims = match dims_value {
        JsonValue::Array(items) => {
            if items.is_empty() {
                return Err(format!("state tensor {name:?} has no dimensions"));
            }
            if items.len() > 8 {
                return Err(format!(
                    "state tensor {name:?} has {} dimensions; GGUF allows at most 8",
                    items.len()
                ));
            }
            let mut dims = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                let dim = json_u64(
                    item,
                    &format!("state_tensors[{name:?}].dims[{index}]"),
                )?;
                if dim == 0 {
                    return Err(format!(
                        "state tensor {name:?} has a zero dimension at index {index}"
                    ));
                }
                dims.push(dim);
            }
            dims
        }
        _ => return Err(format!("state tensor {name:?} dims must be an array")),
    };

    let data_value = required_field(entries, "data", &format!("state_tensors[{name:?}]"))?;
    let data = json_f32_array(data_value, &format!("state_tensors[{name:?}].data"))?;

    let mut expected: u64 = 1;
    for dim in &dims {
        expected = expected.checked_mul(*dim).ok_or_else(|| {
            format!("state tensor {name:?} declares an element count that overflows u64")
        })?;
    }
    if expected != data.len() as u64 {
        return Err(format!(
            "state tensor {name:?} declares {expected} elements but carries {}",
            data.len()
        ));
    }

    Ok(StateTensor {
        name: name.to_string(),
        dims,
        data,
    })
}

fn json_token_types(value: &JsonValue) -> Result<Vec<i32>, String> {
    let items = match value {
        JsonValue::Array(items) => items,
        _ => return Err("meta token_types must be an array".to_string()),
    };
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        out.push(json_i32(item, &format!("meta token_types[{index}]"))?);
    }
    Ok(out)
}

fn json_string_array(
    value: &JsonValue,
    what: &str,
    allow_empty: bool,
) -> Result<Vec<String>, String> {
    let items = match value {
        JsonValue::Array(items) => items,
        _ => return Err(format!("{what} must be an array")),
    };
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            JsonValue::String(value) => {
                if !allow_empty && value.is_empty() {
                    return Err(format!("{what}[{index}] must not be empty"));
                }
                out.push(value.clone());
            }
            _ => return Err(format!("{what}[{index}] must be a string")),
        }
    }
    Ok(out)
}

fn json_f32_array(value: &JsonValue, what: &str) -> Result<Vec<f32>, String> {
    let items = match value {
        JsonValue::Array(items) => items,
        _ => return Err(format!("{what} must be an array")),
    };
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let number = match item {
            JsonValue::Number(number) => *number,
            _ => return Err(format!("{what}[{index}] must be a number")),
        };
        let value = number as f32;
        if !value.is_finite() {
            return Err(format!("{what}[{index}] is not a finite f32"));
        }
        out.push(value);
    }
    Ok(out)
}

fn json_i32(value: &JsonValue, what: &str) -> Result<i32, String> {
    match value {
        JsonValue::Number(number)
            if number.is_finite()
                && number.fract() == 0.0
                && *number >= i32::MIN as f64
                && *number <= i32::MAX as f64 =>
        {
            Ok(*number as i32)
        }
        _ => Err(format!("{what} must be a 32-bit integer")),
    }
}

fn json_u64(value: &JsonValue, what: &str) -> Result<u64, String> {
    // 2^64 exactly; every f64 below it converts to a representable u64.
    const U64_LIMIT: f64 = 18_446_744_073_709_551_616.0;
    match value {
        JsonValue::Number(number)
            if number.is_finite()
                && *number >= 0.0
                && *number < U64_LIMIT
                && number.fract() == 0.0 =>
        {
            Ok(*number as u64)
        }
        _ => Err(format!("{what} must be a non-negative integer")),
    }
}

fn json_object<'a>(value: &'a JsonValue, what: &str) -> Result<&'a [(String, JsonValue)], String> {
    match value {
        JsonValue::Object(entries) => Ok(entries),
        _ => Err(format!("{what} must be a JSON object")),
    }
}

fn json_field<'a>(entries: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    entries
        .iter()
        .find(|(existing, _)| existing == key)
        .map(|(_, value)| value)
}

fn required_field<'a>(
    entries: &'a [(String, JsonValue)],
    key: &str,
    what: &str,
) -> Result<&'a JsonValue, String> {
    json_field(entries, key).ok_or_else(|| format!("{what} is missing the {key:?} field"))
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (std only, no regex)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        JsonParser { bytes, pos: 0 }
    }

    fn parse_document(mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        let value = self.parse_value(0)?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return Err(format!(
                "trailing bytes at JSON offset {} of {}",
                self.pos,
                self.bytes.len()
            ));
        }
        Ok(value)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            match byte {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_value(&mut self, depth: u32) -> Result<JsonValue, String> {
        if depth > MAX_JSON_DEPTH {
            return Err(format!(
                "JSON is nested deeper than {MAX_JSON_DEPTH} levels"
            ));
        }
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b't') => {
                self.expect(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.expect(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            Some(b'n') => {
                self.expect(b"null")?;
                Ok(JsonValue::Null)
            }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(format!(
                "unexpected byte 0x{other:02x} at JSON offset {}",
                self.pos
            )),
            None => Err("unexpected end of JSON input".to_string()),
        }
    }

    fn expect(&mut self, literal: &[u8]) -> Result<(), String> {
        if self.bytes.len() < self.pos + literal.len()
            || &self.bytes[self.pos..self.pos + literal.len()] != literal
        {
            return Err(format!(
                "expected {:?} at JSON offset {}",
                String::from_utf8_lossy(literal),
                self.pos
            ));
        }
        self.pos += literal.len();
        Ok(())
    }

    fn parse_object(&mut self, depth: u32) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(format!(
                    "expected an object key at JSON offset {}",
                    self.pos
                ));
            }
            let key = self.parse_string()?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(format!("duplicate JSON key {key:?}"));
            }
            self.skip_whitespace();
            if self.bump() != Some(b':') {
                return Err(format!("expected ':' at JSON offset {}", self.pos));
            }
            self.skip_whitespace();
            let value = self.parse_value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(entries)),
                _ => {
                    return Err(format!(
                        "expected ',' or '}}' at JSON offset {}",
                        self.pos.saturating_sub(1)
                    ))
                }
            }
        }
    }

    fn parse_array(&mut self, depth: u32) -> Result<JsonValue, String> {
        self.pos += 1; // '['
        let mut items: Vec<JsonValue> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(JsonValue::Array(items)),
                _ => {
                    return Err(format!(
                        "expected ',' or ']' at JSON offset {}",
                        self.pos.saturating_sub(1)
                    ))
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.pos += 1; // '"'
        let mut out: Vec<u8> = Vec::new();
        loop {
            let byte = self
                .bump()
                .ok_or_else(|| "unterminated JSON string".to_string())?;
            match byte {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|_| "JSON string is not valid UTF-8".to_string())
                }
                b'\\' => {
                    let escape = self
                        .bump()
                        .ok_or_else(|| "truncated JSON escape".to_string())?;
                    match escape {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0c),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let first = self.parse_hex4()?;
                            let code = match first {
                                0xD800..=0xDBFF => {
                                    if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                                        return Err(
                                            "high surrogate without a low surrogate in JSON string"
                                                .to_string(),
                                        );
                                    }
                                    let second = self.parse_hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&second) {
                                        return Err(
                                            "invalid low surrogate in JSON string".to_string()
                                        );
                                    }
                                    0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                                }
                                0xDC00..=0xDFFF => {
                                    return Err("lone low surrogate in JSON string".to_string())
                                }
                                _ => first,
                            };
                            let character = char::from_u32(code)
                                .ok_or_else(|| "invalid Unicode escape".to_string())?;
                            let mut buffer = [0u8; 4];
                            out.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                        }
                        other => {
                            return Err(format!(
                                "invalid JSON escape \\{}",
                                other as char
                            ))
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(format!(
                        "raw control byte 0x{byte:02x} in JSON string"
                    ))
                }
                byte => out.push(byte),
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .bump()
                .ok_or_else(|| "truncated \\u escape".to_string())?;
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a') + 10,
                b'A'..=b'F' => u32::from(byte - b'A') + 10,
                _ => return Err("invalid hex digit in a \\u escape".to_string()),
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!(
                        "leading zero in JSON number at offset {start}"
                    ));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at offset {start}")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!(
                    "JSON number at offset {start} has no fraction digits"
                ));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!(
                    "JSON number at offset {start} has no exponent digits"
                ));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "JSON number is not valid UTF-8".to_string())?;
        let value: f64 = text
            .parse()
            .map_err(|_| format!("invalid JSON number {text:?}"))?;
        if !value.is_finite() {
            return Err(format!("JSON number {text:?} is not finite"));
        }
        Ok(JsonValue::Number(value))
    }
}

// ---------------------------------------------------------------------------
// Streaming identity
// ---------------------------------------------------------------------------

/// Compute the streaming identity of a GGUF file.
///
/// The 64-bit FNV-1a value is folded, in tensor-table order, over:
///
/// ```text
/// "pmc-fuse-identity/1\0"
/// file_size                    as u64 little-endian
/// for every tensor:
///     name bytes, then a zero byte
///     declared offset          as u64 little-endian
///     computed byte length     as u64 little-endian
///     FNV-1a of the tensor bytes, as u64 little-endian
/// ```
///
/// Tensor payloads are read in 4 MiB chunks; the file is never loaded whole.
/// The result is rendered as `fnv1a64:<16 lowercase hex digits>`.
///
/// This is an integrity fingerprint, **not a cryptographic signature**.
pub fn tensor_stream_identity(path: &Path) -> Result<FileFingerprint, String> {
    let reader = GgufReader::open(path)
        .map_err(|error| format!("open {} for identity: {error}", path.display()))?;
    let mut file = File::open(path)
        .map_err(|error| format!("open {} for identity: {error}", path.display()))?;
    let mut buffer = vec![0u8; CHUNK];
    let mut identity = identity_begin(reader.file_len());
    for info in reader.tensors() {
        let byte_len = tensor_nbytes(info)?;
        let offset = reader
            .data_start()
            .checked_add(info.offset)
            .ok_or_else(|| format!("tensor {:?} offset overflows u64", info.name))?;
        let tensor_hash = hash_range(&mut file, offset, byte_len, &mut buffer)
            .map_err(|error| format!("hash tensor {:?}: {error}", info.name))?;
        identity_add_tensor(&mut identity, info, byte_len, tensor_hash);
    }
    Ok(FileFingerprint {
        size: reader.file_len(),
        identity: identity_hex(identity),
    })
}

fn identity_begin(file_len: u64) -> u64 {
    let mut hash = FNV_OFFSET;
    fnv1a_update(&mut hash, b"pmc-fuse-identity/1\0");
    fnv1a_update(&mut hash, &file_len.to_le_bytes());
    hash
}

fn identity_add_tensor(hash: &mut u64, info: &TensorInfo, byte_len: u64, tensor_hash: u64) {
    fnv1a_update(hash, info.name.as_bytes());
    fnv1a_update(hash, &[0u8]);
    fnv1a_update(hash, &info.offset.to_le_bytes());
    fnv1a_update(hash, &byte_len.to_le_bytes());
    fnv1a_update(hash, &tensor_hash.to_le_bytes());
}

fn identity_hex(hash: u64) -> String {
    format!("fnv1a64:{hash:016x}")
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_range(file: &mut File, offset: u64, len: u64, buffer: &mut [u8]) -> Result<u64, String> {
    let mut hash = FNV_OFFSET;
    if len == 0 {
        return Ok(hash);
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek to byte {offset}: {error}"))?;
    let mut remaining = len;
    while remaining > 0 {
        let take = std::cmp::min(remaining, buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..take])
            .map_err(|error| format!("read {take} bytes at byte {offset}: {error}"))?;
        fnv1a_update(&mut hash, &buffer[..take]);
        remaining -= take as u64;
    }
    Ok(hash)
}

/// SHA-256 of the whole file, streamed in [`CHUNK`] pieces.
fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("open {} for sha256: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read {} for sha256: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize_hex())
}

fn tensor_nbytes(info: &TensorInfo) -> Result<u64, String> {
    let (block_elems, block_bytes) = quant_geometry(info.kind).ok_or_else(|| {
        format!(
            "tensor {:?} uses unsupported type id {}",
            info.name, info.kind
        )
    })?;
    let elements = info
        .elements()
        .ok_or_else(|| format!("tensor {:?} element count overflows usize", info.name))? as u64;
    let block_elems = block_elems as u64;
    if elements % block_elems != 0 {
        return Err(format!(
            "tensor {:?} has {elements} elements, not a multiple of the block size {block_elems}",
            info.name
        ));
    }
    (elements / block_elems)
        .checked_mul(block_bytes as u64)
        .ok_or_else(|| format!("tensor {:?} byte length overflows u64", info.name))
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

/// Fuse `rows.bin` and `meta.json` into `base`, producing `out`.
///
/// The fused file is staged next to `out`, reopened and verified, then renamed
/// into place. On any error the staging file is removed and an existing `out`
/// is left byte-identical.
pub fn fuse(request: &FuseRequest) -> Result<FuseReport, String> {
    ensure_distinct(request)?;
    ensure_parent_dir(&request.out, "--out")?;
    if let Some(manifest) = &request.manifest {
        ensure_parent_dir(manifest, "--manifest")?;
    }

    let delta_bytes = fs::read(&request.rows)
        .map_err(|error| format!("read delta {}: {error}", request.rows.display()))?;
    let delta = parse_delta(&delta_bytes)
        .map_err(|error| format!("invalid delta {}: {error}", request.rows.display()))?;
    let meta_bytes = fs::read(&request.meta)
        .map_err(|error| format!("read meta {}: {error}", request.meta.display()))?;
    let meta = parse_meta(&meta_bytes, delta.n_rows)
        .map_err(|error| format!("invalid meta {}: {error}", request.meta.display()))?;

    let base = GgufReader::open(&request.base)
        .map_err(|error| format!("open base {}: {error}", request.base.display()))?;
    let old_vocab = metadata_array(&base, "tokenizer.ggml.tokens")?.len();
    let token_info = base.tensor("token_embd.weight").ok_or_else(|| {
        format!(
            "base {} has no token_embd.weight tensor",
            request.base.display()
        )
    })?;
    if token_info.dims.len() != 2 {
        return Err(format!(
            "base token_embd.weight has shape {:?}; expected two dimensions",
            token_info.dims
        ));
    }
    let row_len = token_info
        .row_len()
        .ok_or_else(|| "base token_embd.weight row length overflows usize".to_string())?;
    let old_rows = token_info
        .rows()
        .ok_or_else(|| "base token_embd.weight row count overflows usize".to_string())?;
    if delta.dim != row_len {
        return Err(format!(
            "delta dim {0} does not match base token_embd.weight row length {1}",
            delta.dim, row_len
        ));
    }
    if old_vocab != old_rows {
        return Err(format!(
            "base is inconsistent: {old_vocab} tokenizer tokens but {old_rows} embedding rows"
        ));
    }
    ensure_supported_embedding("token_embd.weight", token_info.kind)?;
    if let Some(output) = base.tensor("output.weight") {
        if output.dims.len() != 2 {
            return Err(format!(
                "base output.weight has shape {:?}; expected two dimensions",
                output.dims
            ));
        }
        if output.row_len() != Some(row_len) {
            return Err(format!(
                "base output.weight row length {:?} does not match token_embd.weight {row_len}",
                output.row_len()
            ));
        }
        if output.rows() != Some(old_rows) {
            return Err(format!(
                "base output.weight has {:?} rows, expected {old_rows}",
                output.rows()
            ));
        }
        ensure_supported_embedding("output.weight", output.kind)?;
    }

    let base_fingerprint = tensor_stream_identity(&request.base)?;
    let rows_sha256 = sha256_hex(&delta_bytes);
    let meta_sha256 = sha256_hex(&meta_bytes);
    let base_identity = base_fingerprint.identity.clone();
    let base_sha256 = file_sha256(&request.base)?;
    let facts = FusionFacts {
        base_path: &request.base,
        base_size: base_fingerprint.size,
        base_identity: &base_identity,
        base_sha256: &base_sha256,
        old_vocab,
        delta: &delta,
        meta: &meta,
        rows_sha256: &rows_sha256,
        meta_sha256: &meta_sha256,
    };
    let embedded_manifest = build_manifest(&facts, None);

    let mut writer = GgufWriter::from_source(&base)
        .map_err(|error| format!("prepare writer from {}: {error}", request.base.display()))?;
    writer
        .append_vocab_tokens(&meta.tokens, &meta.scores, &meta.token_types)
        .map_err(|error| format!("append vocabulary: {error}"))?;
    writer
        .append_embedding_rows(&delta.rows)
        .map_err(|error| format!("append embedding rows: {error}"))?;
    for state in &meta.state_tensors {
        let dims = state
            .dims
            .iter()
            .map(|dim| {
                u32::try_from(*dim)
                    .map(GgufValue::U32)
                    .map_err(|_| format!("state tensor {:?} dimension {dim} does not fit u32", state.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        writer
            .set_metadata(
                &format!("{}.dims", state.name),
                GgufValue::array(GgufValue::U32(0).type_id(), dims),
            )
            .map_err(|error| format!("set state dims {:?}: {error}", state.name))?;
        let data: Vec<GgufValue> = state.data.iter().map(|value| GgufValue::F32(*value)).collect();
        writer
            .set_metadata(
                &format!("{}.data", state.name),
                GgufValue::array(GgufValue::F32(0.0).type_id(), data),
            )
            .map_err(|error| format!("set state data {:?}: {error}", state.name))?;
    }
    writer
        .set_metadata("pmc.version", GgufValue::U32(PMC_VERSION))
        .map_err(|error| format!("set pmc.version: {error}"))?;
    writer
        .set_metadata("pmc.manifest", GgufValue::String(embedded_manifest.clone()))
        .map_err(|error| format!("set pmc.manifest: {error}"))?;
    if let Some(dream_id) = &meta.dream_id {
        writer
            .set_metadata("pmc.dream_id", GgufValue::String(dream_id.clone()))
            .map_err(|error| format!("set pmc.dream_id: {error}"))?;
    }

    let mut staged_output = StagedFile::new(staging_path(&request.out, "gguf"));
    writer
        .write_to(&staged_output.path)
        .map_err(|error| format!("write fused GGUF: {error}"))?;
    sync_file(&staged_output.path)?;

    let verified_base = verify_fused(
        &request.base,
        &base,
        &staged_output.path,
        &delta,
        &meta,
        &embedded_manifest,
        old_vocab,
    )?;
    if verified_base != base_fingerprint {
        return Err(format!(
            "base model changed during fusion: identity {} became {}",
            base_fingerprint.identity, verified_base.identity
        ));
    }

    let output_fingerprint = tensor_stream_identity(&staged_output.path)?;
    let output_sha256 = file_sha256(&staged_output.path)?;
    let output_path_text = request.out.display().to_string();
    let mut staged_manifest = None;
    if let Some(target) = &request.manifest {
        let manifest = build_manifest(
            &facts,
            Some((
                &request.out,
                output_fingerprint.size,
                &output_fingerprint.identity,
                &output_sha256,
            )),
        );
        let staged = StagedFile::new(staging_path(target, "manifest"));
        write_text_file(&staged.path, &manifest)?;
        staged_manifest = Some((staged, target.clone()));
    }

    fs::rename(&staged_output.path, &request.out).map_err(|error| {
        format!(
            "publish fused file {}: {error}",
            request.out.display()
        )
    })?;
    staged_output.keep = true;

    if let Some((mut staged, target)) = staged_manifest {
        if let Err(error) = fs::rename(&staged.path, &target) {
            return Err(format!(
                "fused file published at {}, but the manifest could not be published at {}: {error}",
                request.out.display(),
                target.display()
            ));
        }
        staged.keep = true;
    }

    Ok(FuseReport {
        base_path: request.base.display().to_string(),
        base_size: base_fingerprint.size,
        base_identity: base_fingerprint.identity,
        base_sha256,
        old_vocab,
        new_vocab: old_vocab + delta.n_rows,
        token_first: old_vocab,
        token_last: old_vocab + delta.n_rows - 1,
        rows: delta.n_rows,
        dim: delta.dim,
        rows_sha256,
        meta_sha256,
        state_tensors: meta
            .state_tensors
            .iter()
            .map(|state| StateTensorSummary {
                name: state.name.clone(),
                dims: state.dims.clone(),
                elements: state.data.len(),
            })
            .collect(),
        output_path: output_path_text,
        output_size: output_fingerprint.size,
        output_identity: output_fingerprint.identity,
        output_sha256,
        dream_id: meta.dream_id,
        provenance: meta.provenance,
        manifest_path: request.manifest.as_ref().map(|path| path.display().to_string()),
    })
}

fn ensure_supported_embedding(name: &str, kind: u32) -> Result<(), String> {
    match kind {
        GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => Ok(()),
        other => Err(format!(
            "{name} uses {}; PMC fusion supports Q4_K, Q5_K and Q6_K embeddings",
            quant_name(other)
        )),
    }
}

fn metadata_array<'a>(reader: &'a GgufReader, key: &str) -> Result<&'a [GgufValue], String> {
    reader
        .get(key)
        .and_then(GgufValue::as_array)
        .ok_or_else(|| format!("GGUF file has no {key:?} array"))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify_fused(
    base_path: &Path,
    base: &GgufReader,
    fused_path: &Path,
    delta: &Delta,
    meta: &FuseMeta,
    embedded_manifest: &str,
    old_vocab: usize,
) -> Result<FileFingerprint, String> {
    let fused = GgufReader::open(fused_path)
        .map_err(|error| format!("reopen fused file {}: {error}", fused_path.display()))?;
    let new_vocab = old_vocab
        .checked_add(delta.n_rows)
        .ok_or_else(|| "fused vocabulary size overflows usize".to_string())?;

    match fused.get("pmc.version") {
        Some(GgufValue::U32(version)) if *version == PMC_VERSION => {}
        Some(other) => {
            return Err(format!(
                "fused pmc.version is {}, expected a u32",
                other.type_name()
            ))
        }
        None => return Err("fused file is missing pmc.version".to_string()),
    }
    match fused.get("pmc.manifest").and_then(GgufValue::as_str) {
        Some(value) if value == embedded_manifest => {}
        Some(_) => return Err("fused pmc.manifest does not match the manifest that was written".into()),
        None => return Err("fused file is missing the pmc.manifest string".to_string()),
    }
    match (&meta.dream_id, fused.get("pmc.dream_id")) {
        (Some(expected), Some(GgufValue::String(actual))) if actual == expected => {}
        (None, None) => {}
        (Some(expected), None) => {
            return Err(format!(
                "fused file is missing pmc.dream_id {expected:?}"
            ))
        }
        (None, Some(_)) => {
            return Err("fused file has a pmc.dream_id that the delta did not declare".to_string())
        }
        (Some(_), Some(other)) => {
            return Err(format!(
                "fused pmc.dream_id is {}, expected a string",
                other.type_name()
            ))
        }
    }

    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    ] {
        let alen = metadata_array(&fused, key)?.len();
        if alen != new_vocab {
            return Err(format!(
                "fused {key:?} has {alen} entries, expected {new_vocab}"
            ));
        }
    }
    let appended_tokens = metadata_array(&fused, "tokenizer.ggml.tokens")?;
    for (index, expected) in meta.tokens.iter().enumerate() {
        match &appended_tokens[old_vocab + index] {
            GgufValue::String(actual) if actual == expected => {}
            other => {
                return Err(format!(
                    "fused token {} is {:?}, expected {:?}",
                    old_vocab + index,
                    other,
                    expected
                ))
            }
        }
    }
    let appended_scores = metadata_array(&fused, "tokenizer.ggml.scores")?;
    for (index, expected) in meta.scores.iter().enumerate() {
        match &appended_scores[old_vocab + index] {
            GgufValue::F32(actual) if actual.to_bits() == expected.to_bits() => {}
            other => {
                return Err(format!(
                    "fused score {} is {:?}, expected {expected}",
                    old_vocab + index,
                    other
                ))
            }
        }
    }
    let appended_types = metadata_array(&fused, "tokenizer.ggml.token_type")?;
    for (index, expected) in meta.token_types.iter().enumerate() {
        match &appended_types[old_vocab + index] {
            GgufValue::I32(actual) if actual == expected => {}
            other => {
                return Err(format!(
                    "fused token type {} is {:?}, expected {expected}",
                    old_vocab + index,
                    other
                ))
            }
        }
    }

    let fused_token_info = fused
        .tensor("token_embd.weight")
        .ok_or_else(|| "fused file has no token_embd.weight".to_string())?;
    let expected_dims = vec![delta.dim as u64, new_vocab as u64];
    if fused_token_info.dims != expected_dims {
        return Err(format!(
            "fused token_embd.weight has shape {:?}, expected {:?}",
            fused_token_info.dims, expected_dims
        ));
    }
    let base_token_kind = base
        .tensor("token_embd.weight")
        .map(|info| info.kind)
        .ok_or_else(|| "base has no token_embd.weight".to_string())?;
    if fused_token_info.kind != base_token_kind {
        return Err("fused token_embd.weight changed quantization type".to_string());
    }

    match (base.tensor("output.weight"), fused.tensor("output.weight")) {
        (None, None) => {}
        (Some(source), Some(output)) => {
            if output.dims != vec![delta.dim as u64, new_vocab as u64] {
                return Err(format!(
                    "fused output.weight has shape {:?}, expected {:?}",
                    output.dims, expected_dims
                ));
            }
            if output.kind != source.kind {
                return Err("fused output.weight changed quantization type".to_string());
            }
        }
        (None, Some(_)) => {
            return Err("fused file has an output.weight that the base did not have".to_string())
        }
        (Some(_), None) => return Err("fused file lost output.weight".to_string()),
    }

    for state in &meta.state_tensors {
        let dims_key = format!("{}.dims", state.name);
        let data_key = format!("{}.data", state.name);
        let dims = metadata_array(&fused, &dims_key)?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("fused {dims_key:?} contains a non-u32 value"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dims != state.dims {
            return Err(format!(
                "fused state metadata {:?} has shape {:?}, expected {:?}",
                state.name, dims, state.dims
            ));
        }
        let data = metadata_array(&fused, &data_key)?;
        if data.len() != state.data.len() {
            return Err(format!(
                "fused state metadata {:?} returned {} values, expected {}",
                state.name,
                data.len(),
                state.data.len()
            ));
        }
        for (index, (actual, expected)) in data.iter().zip(&state.data).enumerate() {
            let actual = match actual {
                GgufValue::F32(value) => *value,
                other => {
                    return Err(format!(
                        "fused state metadata {data_key:?} element {index} is {}, expected f32",
                        other.type_name()
                    ))
                }
            };
            if actual.to_bits() != expected.to_bits() {
                return Err(format!(
                    "fused state metadata {:?} differs at element {index}: {actual} != {expected}",
                    state.name
                ));
            }
        }
    }

    let expected_tensor_count = base.tensors().len();
    if fused.tensors().len() != expected_tensor_count {
        return Err(format!(
            "fused file has {} tensors, expected {expected_tensor_count}",
            fused.tensors().len()
        ));
    }

    let mut base_file = File::open(base_path)
        .map_err(|error| format!("open base {} for verification: {error}", base_path.display()))?;
    let mut fused_file = File::open(fused_path)
        .map_err(|error| format!("open fused {} for verification: {error}", fused_path.display()))?;
    let mut base_buffer = vec![0u8; CHUNK];
    let mut fused_buffer = vec![0u8; CHUNK];
    let mut identity = identity_begin(base.file_len());

    for info in base.tensors() {
        let fused_info = fused.tensor(&info.name).ok_or_else(|| {
            format!("fused file is missing base tensor {:?}", info.name)
        })?;
        if fused_info.kind != info.kind {
            return Err(format!(
                "base tensor {:?} changed quantization type",
                info.name
            ));
        }
        if fused_info.dims.first() != info.dims.first() {
            return Err(format!(
                "base tensor {:?} changed its row length",
                info.name
            ));
        }
        let extended = info.name == "token_embd.weight" || info.name == "output.weight";
        if info.dims.len() == 2 {
            let expected_rows = if extended {
                info.dims[1]
                    .checked_add(delta.n_rows as u64)
                    .ok_or_else(|| format!("tensor {:?} row count overflows", info.name))?
            } else {
                info.dims[1]
            };
            if fused_info.dims[1] != expected_rows {
                return Err(format!(
                    "base tensor {:?} has {} rows in the fused file, expected {expected_rows}",
                    info.name, fused_info.dims[1]
                ));
            }
        } else if fused_info.dims != info.dims {
            return Err(format!("base tensor {:?} changed shape", info.name));
        }

        let byte_len = tensor_nbytes(info)?;
        let source_offset = base
            .data_start()
            .checked_add(info.offset)
            .ok_or_else(|| format!("tensor {:?} offset overflows u64", info.name))?;
        let fused_offset = fused
            .data_start()
            .checked_add(fused_info.offset)
            .ok_or_else(|| format!("tensor {:?} offset overflows u64", info.name))?;
        let source_hash = hash_range(&mut base_file, source_offset, byte_len, &mut base_buffer)
            .map_err(|error| format!("hash base tensor {:?}: {error}", info.name))?;
        let fused_hash = hash_range(&mut fused_file, fused_offset, byte_len, &mut fused_buffer)
            .map_err(|error| format!("hash fused tensor {:?}: {error}", info.name))?;
        if source_hash != fused_hash {
            return Err(format!(
                "base tensor {:?} bytes changed in the fused file (source 0x{source_hash:016x}, fused 0x{fused_hash:016x})",
                info.name
            ));
        }
        identity_add_tensor(&mut identity, info, byte_len, source_hash);
    }

    Ok(FileFingerprint {
        size: base.file_len(),
        identity: identity_hex(identity),
    })
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

struct FusionFacts<'a> {
    base_path: &'a Path,
    base_size: u64,
    base_identity: &'a str,
    base_sha256: &'a str,
    old_vocab: usize,
    delta: &'a Delta,
    meta: &'a FuseMeta,
    rows_sha256: &'a str,
    meta_sha256: &'a str,
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf")
        .to_string()
}

fn build_manifest(facts: &FusionFacts, output: Option<(&Path, u64, &str, &str)>) -> String {
    let new_vocab = facts.old_vocab + facts.delta.n_rows;
    let mut entries: Vec<(&str, Json)> = vec![
        ("format", Json::str(MANIFEST_FORMAT)),
        (
            "base",
            Json::obj(vec![
                ("file", Json::str(file_name(facts.base_path))),
                ("size", Json::num(facts.base_size as f64)),
                ("identity", Json::str(facts.base_identity)),
                ("sha256", Json::str(facts.base_sha256)),
            ]),
        ),
        (
            "delta",
            Json::obj(vec![
                ("rows", Json::num(facts.delta.n_rows as f64)),
                ("dim", Json::num(facts.delta.dim as f64)),
                ("sha256", Json::str(facts.rows_sha256)),
                ("meta_sha256", Json::str(facts.meta_sha256)),
            ]),
        ),
        (
            "tokens",
            Json::obj(vec![
                ("old_count", Json::num(facts.old_vocab as f64)),
                ("added", Json::num(facts.delta.n_rows as f64)),
                ("new_count", Json::num(new_vocab as f64)),
                ("first_index", Json::num(facts.old_vocab as f64)),
                ("last_index", Json::num(new_vocab.saturating_sub(1) as f64)),
            ]),
        ),
        ("state_tensors", state_meta_json(&facts.meta.state_tensors)),
    ];
    if let Some((path, size, identity, sha256)) = output {
        entries.push((
            "output",
            Json::obj(vec![
                ("file", Json::str(file_name(path))),
                ("size", Json::num(size as f64)),
                ("identity", Json::str(identity)),
                ("sha256", Json::str(sha256)),
            ]),
        ));
    }
    entries.push((
        "dream_id",
        match &facts.meta.dream_id {
            Some(value) => Json::str(value.clone()),
            None => Json::Null,
        },
    ));
    entries.push((
        "provenance",
        Json::Arr(
            facts
                .meta
                .provenance
                .iter()
                .map(|value| Json::str(value.clone()))
                .collect(),
        ),
    ));
    Json::obj(entries).to_string()
}

fn state_meta_json(tensors: &[StateTensor]) -> Json {
    Json::Arr(
        tensors
            .iter()
            .map(|tensor| {
                Json::obj(vec![
                    ("name", Json::str(tensor.name.clone())),
                    ("dims", dims_json(&tensor.dims)),
                    ("elements", Json::num(tensor.data.len() as f64)),
                    ("storage", Json::str("metadata")),
                ])
            })
            .collect(),
    )
}

fn state_summary_json(tensors: &[StateTensorSummary]) -> Json {
    Json::Arr(
        tensors
            .iter()
            .map(|tensor| {
                Json::obj(vec![
                    ("name", Json::str(tensor.name.clone())),
                    ("dims", dims_json(&tensor.dims)),
                    ("elements", Json::num(tensor.elements as f64)),
                    ("storage", Json::str("metadata")),
                ])
            })
            .collect(),
    )
}

fn dims_json(dims: &[u64]) -> Json {
    Json::Arr(dims.iter().map(|dim| Json::num(*dim as f64)).collect())
}

// ---------------------------------------------------------------------------
// Paths and staging
// ---------------------------------------------------------------------------

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

struct StagedFile {
    path: PathBuf,
    keep: bool,
}

impl StagedFile {
    fn new(path: PathBuf) -> Self {
        StagedFile { path, keep: false }
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn staging_path(target: &Path, tag: &str) -> PathBuf {
    let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = target
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| OsString::from("pmc-fuse"));
    name.push(format!(".{tag}.tmp-{}-{counter}", std::process::id()));
    match target.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

fn ensure_distinct(request: &FuseRequest) -> Result<(), String> {
    let mut paths: Vec<(&str, &Path)> = vec![
        ("--base", &request.base),
        ("--rows", &request.rows),
        ("--meta", &request.meta),
        ("--out", &request.out),
    ];
    if let Some(manifest) = &request.manifest {
        paths.push(("--manifest", manifest));
    }
    for left in 0..paths.len() {
        for right in (left + 1)..paths.len() {
            if comparable(paths[left].1) == comparable(paths[right].1) {
                return Err(format!(
                    "{} and {} must be different files (both resolve to {})",
                    paths[left].0,
                    paths[right].0,
                    paths[left].1.display()
                ));
            }
        }
    }
    Ok(())
}

fn comparable(path: &Path) -> PathBuf {
    if let Ok(resolved) = fs::canonicalize(path) {
        return resolved;
    }
    let absolute = normalize(path);
    match (absolute.parent(), absolute.file_name()) {
        (Some(parent), Some(name)) => match fs::canonicalize(parent) {
            Ok(resolved) => resolved.join(name),
            Err(_) => absolute,
        },
        _ => absolute,
    }
}

fn normalize(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(directory) => directory.join(path),
            Err(_) => path.to_path_buf(),
        }
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn ensure_parent_dir(path: &Path, flag: &str) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    match fs::metadata(parent) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(format!(
            "{flag} parent {} is not a directory",
            parent.display()
        )),
        Err(error) => Err(format!(
            "{flag} parent {} is not usable: {error}",
            parent.display()
        )),
    }
}

fn sync_file(path: &Path) -> Result<(), String> {
    // Windows: FlushFileBuffers requires a handle with write access, so a
    // read-only open succeeds everywhere and then fails at sync_all with
    // ERROR_ACCESS_DENIED (os error 5). Open read-write instead.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("open {} for sync: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync {}: {error}", path.display()))
}

fn write_text_file(path: &Path, contents: &str) -> Result<(), String> {
    let mut file = File::create(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    file.write_all(contents.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn delta_bytes(magic: [u8; 4], version: u32, rows: u32, dim: u32, data: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&magic);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&dim.to_le_bytes());
        for value in data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn valid_delta() -> Vec<u8> {
        delta_bytes(
            DELTA_MAGIC,
            DELTA_VERSION,
            2,
            3,
            &[1.0, -2.0, 3.5, 4.25, 5.0, -6.75],
        )
    }

    fn meta_document(tokens: &[&str], scores: &[f64], types: &[i64]) -> String {
        Json::obj(vec![
            (
                "tokens",
                Json::Arr(
                    tokens
                        .iter()
                        .map(|token| Json::str(*token))
                        .collect(),
                ),
            ),
            (
                "scores",
                Json::Arr(scores.iter().map(|score| Json::num(*score)).collect()),
            ),
            (
                "token_types",
                Json::Arr(types.iter().map(|value| Json::num(*value as f64)).collect()),
            ),
        ])
        .to_string()
    }

    #[test]
    fn delta_accepts_a_valid_file() {
        let delta = parse_delta(&valid_delta()).expect("valid delta");
        assert_eq!(delta.n_rows, 2);
        assert_eq!(delta.dim, 3);
        assert_eq!(delta.rows.len(), 2);
        assert_eq!(delta.rows[0], vec![1.0, -2.0, 3.5]);
        assert_eq!(delta.rows[1], vec![4.25, 5.0, -6.75]);
    }

    #[test]
    fn delta_rejects_bad_magic() {
        let bytes = delta_bytes(*b"PMCX", DELTA_VERSION, 1, 1, &[0.0]);
        let error = parse_delta(&bytes).expect_err("bad magic must fail");
        assert!(error.contains("bad magic"), "{error}");
    }

    #[test]
    fn delta_rejects_bad_version() {
        let bytes = delta_bytes(DELTA_MAGIC, 2, 1, 1, &[0.0]);
        let error = parse_delta(&bytes).expect_err("bad version must fail");
        assert!(error.contains("unsupported delta version 2"), "{error}");
    }

    #[test]
    fn delta_rejects_truncation() {
        let mut bytes = valid_delta();
        bytes.pop();
        let error = parse_delta(&bytes).expect_err("truncated delta must fail");
        assert!(error.contains("truncated"), "{error}");

        let error = parse_delta(&bytes[..DELTA_HEADER_BYTES - 1])
            .expect_err("short header must fail");
        assert!(error.contains("truncated"), "{error}");
    }

    #[test]
    fn delta_rejects_extra_bytes() {
        let mut bytes = valid_delta();
        bytes.push(0);
        let error = parse_delta(&bytes).expect_err("extra bytes must fail");
        assert!(error.contains("extra bytes"), "{error}");
    }

    #[test]
    fn delta_rejects_zero_dimensions() {
        let bytes = delta_bytes(DELTA_MAGIC, DELTA_VERSION, 0, 3, &[]);
        let error = parse_delta(&bytes).expect_err("zero rows must fail");
        assert!(error.contains("zero rows"), "{error}");

        let bytes = delta_bytes(DELTA_MAGIC, DELTA_VERSION, 1, 0, &[]);
        let error = parse_delta(&bytes).expect_err("zero dim must fail");
        assert!(error.contains("zero dimensions"), "{error}");
    }

    #[test]
    fn delta_rejects_non_finite_values() {
        let bytes = delta_bytes(DELTA_MAGIC, DELTA_VERSION, 1, 2, &[0.0, f32::NAN]);
        let error = parse_delta(&bytes).expect_err("NaN must fail");
        assert!(error.contains("not a finite f32"), "{error}");
    }

    #[test]
    fn meta_accepts_the_full_schema() {
        let text = Json::obj(vec![
            (
                "tokens",
                Json::Arr(vec![Json::str("<pmc:0>"), Json::str("<pmc:1>")]),
            ),
            (
                "scores",
                Json::Arr(vec![Json::num(0.0), Json::num(-0.5)]),
            ),
            (
                "token_types",
                Json::Arr(vec![Json::num(4.0), Json::num(4.0)]),
            ),
            (
                "state_tensors",
                Json::Obj(vec![(
                    "pmc.state.bank".to_string(),
                    Json::obj(vec![
                        (
                            "dims",
                            Json::Arr(vec![Json::num(2.0), Json::num(2.0)]),
                        ),
                        (
                            "data",
                            Json::Arr(vec![
                                Json::num(1.0),
                                Json::num(-2.0),
                                Json::num(3.5),
                                Json::num(-4.25),
                            ]),
                        ),
                    ]),
                )]),
            ),
            ("dream_id", Json::str("dream-1")),
            (
                "provenance",
                Json::Arr(vec![Json::str("sha256:aaaa"), Json::str("sha256:bbbb")]),
            ),
        ])
        .to_string();

        let meta = parse_meta(text.as_bytes(), 2).expect("valid meta");
        assert_eq!(meta.tokens, vec!["<pmc:0>", "<pmc:1>"]);
        assert_eq!(meta.scores, vec![0.0, -0.5]);
        assert_eq!(meta.token_types, vec![4, 4]);
        assert_eq!(meta.state_tensors.len(), 1);
        assert_eq!(meta.state_tensors[0].name, "pmc.state.bank");
        assert_eq!(meta.state_tensors[0].dims, vec![2, 2]);
        assert_eq!(meta.state_tensors[0].data, vec![1.0, -2.0, 3.5, -4.25]);
        assert_eq!(meta.dream_id.as_deref(), Some("dream-1"));
        assert_eq!(meta.provenance.len(), 2);
    }

    #[test]
    fn meta_rejects_length_mismatches() {
        let text = meta_document(&["a", "b", "c"], &[0.0, 0.0], &[4, 4]);
        let error = parse_meta(text.as_bytes(), 2).expect_err("token mismatch must fail");
        assert!(error.contains("tokens has 3 entries"), "{error}");

        let text = meta_document(&["a", "b"], &[0.0], &[4, 4]);
        let error = parse_meta(text.as_bytes(), 2).expect_err("score mismatch must fail");
        assert!(error.contains("scores has 1 entries"), "{error}");

        let text = meta_document(&["a", "b"], &[0.0, 0.0], &[4]);
        let error = parse_meta(text.as_bytes(), 2).expect_err("type mismatch must fail");
        assert!(error.contains("token_types has 1 entries"), "{error}");

        let text = meta_document(&["a"], &[0.0], &[4]);
        let error = parse_meta(text.as_bytes(), 2).expect_err("n_rows mismatch must fail");
        assert!(error.contains("delta has 2 rows"), "{error}");
    }

    #[test]
    fn meta_rejects_a_bad_state_tensor() {
        let text = Json::obj(vec![
            (
                "tokens",
                Json::Arr(vec![Json::str("<pmc:0>")]),
            ),
            ("scores", Json::Arr(vec![Json::num(0.0)])),
            ("token_types", Json::Arr(vec![Json::num(4.0)])),
            (
                "state_tensors",
                Json::Obj(vec![(
                    "pmc.state.bank".to_string(),
                    Json::obj(vec![
                        ("dims", Json::Arr(vec![Json::num(2.0), Json::num(4.0)])),
                        ("data", Json::Arr(vec![Json::num(1.0), Json::num(2.0)])),
                    ]),
                )]),
            ),
        ])
        .to_string();
        let error = parse_meta(text.as_bytes(), 1).expect_err("shape mismatch must fail");
        assert!(error.contains("declares 8 elements but carries 2"), "{error}");
    }

    #[test]
    fn meta_rejects_a_state_name_outside_the_pmc_namespace() {
        let text = Json::obj(vec![
            ("tokens", Json::Arr(vec![Json::str("<pmc:0>")])),
            ("scores", Json::Arr(vec![Json::num(0.0)])),
            ("token_types", Json::Arr(vec![Json::num(4.0)])),
            (
                "state_tensors",
                Json::Obj(vec![(
                    "bank".to_string(),
                    Json::obj(vec![
                        ("dims", Json::Arr(vec![Json::num(2.0), Json::num(2.0)])),
                        (
                            "data",
                            Json::Arr(vec![
                                Json::num(1.0),
                                Json::num(2.0),
                                Json::num(3.0),
                                Json::num(4.0),
                            ]),
                        ),
                    ]),
                )]),
            ),
        ])
        .to_string();
        let error = parse_meta(text.as_bytes(), 1).expect_err("namespace must be required");
        assert!(error.contains("pmc.state."), "{error}");
    }

    #[test]
    fn meta_rejects_unknown_and_duplicate_fields() {
        let text = r#"{"tokens":["a"],"scores":[0],"token_types":[4],"extra":1}"#;
        let error = parse_meta(text.as_bytes(), 1).expect_err("unknown field must fail");
        assert!(error.contains("unknown meta field"), "{error}");

        let text = r#"{"tokens":["a"],"tokens":["b"],"scores":[0],"token_types":[4]}"#;
        let error = parse_meta(text.as_bytes(), 1).expect_err("duplicate field must fail");
        assert!(error.contains("duplicate JSON key"), "{error}");
    }

    #[test]
    fn meta_rejects_a_non_user_defined_token() {
        let text = meta_document(&["a"], &[0.0], &[1]);
        let error = parse_meta(text.as_bytes(), 1).expect_err("token type 1 must fail");
        assert!(error.contains("type 4"), "{error}");
    }

    #[test]
    fn meta_rejects_malformed_json() {
        let error = parse_meta(b"{\"tokens\": [}", 1).expect_err("bad JSON must fail");
        assert!(error.contains("JSON offset"), "{error}");

        let text = format!("{} trailing", meta_document(&["a"], &[0.0], &[4]));
        let error = parse_meta(text.as_bytes(), 1).expect_err("trailing bytes must fail");
        assert!(error.contains("trailing bytes"), "{error}");
    }

    #[test]
    fn json_strings_handle_escapes_and_surrogate_pairs() {
        let document = JsonParser::new(br#""\u00e9\ud83d\ude00\t\"\\""#)
            .parse_document()
            .expect("escaped string");
        assert_eq!(
            document,
            JsonValue::String("\u{e9}\u{1F600}\t\"\\".to_string())
        );
    }

    #[test]
    fn identity_hex_is_stable() {
        assert_eq!(identity_hex(FNV_OFFSET), "fnv1a64:cbf29ce484222325");
    }
}
