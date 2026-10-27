// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for the GGUF writer and vocabulary extension.
//!
//! The model-dependent tests read `PMC_TEST_GGUF` and return early when it is
//! unset. The llama.cpp test additionally reads `PMC_LLAMA_CLI` and is ignored
//! by default, because it needs a built llama-cli.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pmc_gguf::{
    quant_geometry, GgufReader, GgufValue, GgufWriter, GGML_TYPE_F32, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "pmc-gguf-writer-{}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
            tag
        );
        path.push(unique);
        fs::create_dir_all(&path).expect("create temporary directory");
        TempDir { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn env_model(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(value) => {
            let path = PathBuf::from(value);
            assert!(
                path.is_file(),
                "{name} is set but does not point to a file: {}",
                path.display()
            );
            Some(path)
        }
        None => {
            eprintln!("skipping model-dependent test: {name} is not set");
            None
        }
    }
}

fn tensor_offset(reader: &GgufReader, name: &str) -> u64 {
    let info = reader.tensor(name).expect("tensor present");
    reader.data_start() + info.offset
}

fn supported_tensor_nbytes(reader: &GgufReader, name: &str) -> u64 {
    let info = reader.tensor(name).expect("tensor present");
    let (block_elems, block_bytes) = quant_geometry(info.kind).expect("known tensor type");
    let row_len = reader.row_len(name).expect("row length") as u64;
    let rows = reader.rows(name).expect("rows") as u64;
    assert_eq!(row_len % block_elems as u64, 0);
    (row_len / block_elems as u64) * block_bytes as u64 * rows
}

fn hash_range(path: &Path, offset: u64, len: u64) -> u64 {
    let mut file = File::open(path).expect("open file for hashing");
    file.seek(SeekFrom::Start(offset)).expect("seek for hashing");
    let mut buffer = vec![0u8; 64 * 1024];
    let mut remaining = len;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    while remaining > 0 {
        let take = std::cmp::min(remaining, buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..take]).expect("read for hashing");
        for &byte in &buffer[..take] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        remaining -= take as u64;
    }
    hash
}

fn assert_ranges_equal(
    left_path: &Path,
    left_offset: u64,
    right_path: &Path,
    right_offset: u64,
    len: u64,
) {
    let mut left = File::open(left_path).expect("open left range");
    let mut right = File::open(right_path).expect("open right range");
    left.seek(SeekFrom::Start(left_offset)).expect("seek left");
    right
        .seek(SeekFrom::Start(right_offset))
        .expect("seek right");
    let mut left_buffer = vec![0u8; 64 * 1024];
    let mut right_buffer = vec![0u8; 64 * 1024];
    let mut remaining = len;
    while remaining > 0 {
        let take = std::cmp::min(remaining, left_buffer.len() as u64) as usize;
        left.read_exact(&mut left_buffer[..take])
            .expect("read left range");
        right
            .read_exact(&mut right_buffer[..take])
            .expect("read right range");
        assert_eq!(
            &left_buffer[..take],
            &right_buffer[..take],
            "ranges differ at offset {}",
            len - remaining
        );
        remaining -= take as u64;
    }
}

fn normalized_rms(original: &[f32], decoded: &[f32]) -> f32 {
    let mut error = 0.0f32;
    let mut signal = 0.0f32;
    for (&a, &b) in original.iter().zip(decoded) {
        let delta = a - b;
        error += delta * delta;
        signal += a * a;
    }
    (error / signal.max(f32::MIN_POSITIVE)).sqrt()
}

fn array_len(reader: &GgufReader, key: &str) -> Option<usize> {
    reader
        .get(key)
        .and_then(GgufValue::as_array)
        .map(|items| items.len())
}

fn write_fused_fixture(source: &Path, output: &Path) -> (usize, usize, Vec<Vec<f32>>) {
    let mut reader = GgufReader::open(source).expect("open source model");
    let row_len = reader
        .row_len("token_embd.weight")
        .expect("token_embd.weight row length");
    let old_rows = reader
        .rows("token_embd.weight")
        .expect("token_embd.weight rows");

    let mut rows = Vec::with_capacity(8);
    for row in 0..8 {
        let mut values = vec![0f32; row_len];
        reader
            .read_row_f32("token_embd.weight", row, &mut values)
            .expect("read source row");
        rows.push(values);
    }

    let tokens: Vec<String> = (0..8).map(|i| format!("<pmc_test_{i}>")).collect();
    let scores = vec![0.0f32; 8];
    let token_types = vec![4i32; 8];

    let mut writer = GgufWriter::from_source(&reader).expect("create writer");
    writer
        .append_vocab_tokens(&tokens, &scores, &token_types)
        .expect("append vocabulary");
    writer
        .append_embedding_rows(&rows)
        .expect("append embedding rows");
    writer.write_to(output).expect("write fused GGUF");

    (row_len, old_rows, rows)
}

// ---------------------------------------------------------------------------
// Synthetic GGUF source
// ---------------------------------------------------------------------------

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn synthetic_header(
    token_embd_offset: u64,
    output_offset: u64,
    other_offset: u64,
) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x4655_4747); // "GGUF"
    push_u32(&mut out, 3);
    push_u64(&mut out, 3); // tensors
    push_u64(&mut out, 3); // metadata entries

    push_string(&mut out, "general.alignment");
    push_u32(&mut out, 4); // u32
    push_u32(&mut out, 32);

    // tokenizer.ggml.tokens: array of 4 strings.
    push_string(&mut out, "tokenizer.ggml.tokens");
    push_u32(&mut out, 9); // array
    push_u32(&mut out, 8); // string elements
    push_u64(&mut out, 4);
    for token in ["a", "b", "c", "d"] {
        push_string(&mut out, token);
    }

    // tokenizer.ggml.token_type: array of 4 i32 values.
    push_string(&mut out, "tokenizer.ggml.token_type");
    push_u32(&mut out, 9);
    push_u32(&mut out, 5); // i32 elements
    push_u64(&mut out, 4);
    for token_type in [1i32, 1, 1, 1] {
        push_i32(&mut out, token_type);
    }

    // token_embd.weight: raw dims [row_len, rows].
    push_string(&mut out, "token_embd.weight");
    push_u32(&mut out, 2);
    push_u64(&mut out, 256);
    push_u64(&mut out, 4);
    push_u32(&mut out, GGML_TYPE_Q6_K);
    push_u64(&mut out, token_embd_offset);

    // output.weight: raw dims [row_len, rows], untied in this fixture.
    push_string(&mut out, "output.weight");
    push_u32(&mut out, 2);
    push_u64(&mut out, 256);
    push_u64(&mut out, 4);
    push_u32(&mut out, GGML_TYPE_Q6_K);
    push_u64(&mut out, output_offset);

    // other.weight: F32 [3].
    push_string(&mut out, "other.weight");
    push_u32(&mut out, 1);
    push_u64(&mut out, 3);
    push_u32(&mut out, GGML_TYPE_F32);
    push_u64(&mut out, other_offset);

    out
}

fn synthetic_gguf() -> Vec<u8> {
    let probe = synthetic_header(0, 0, 0);
    let data_start = ((probe.len() + 31) / 32) * 32;
    let q6_single_tensor_bytes = 4 * 210; // four Q6_K rows of 256 elements
    let q6_single_tensor_padded = ((q6_single_tensor_bytes + 31) / 32) * 32;
    let output_offset = q6_single_tensor_padded;
    let other_offset = output_offset + q6_single_tensor_padded;
    let mut out = synthetic_header(0, output_offset as u64, other_offset as u64);
    assert_eq!(out.len(), probe.len());
    out.resize(data_start, 0);
    out.resize(data_start + q6_single_tensor_bytes, 0);
    out.resize(data_start + output_offset, 0);
    out.resize(data_start + output_offset + q6_single_tensor_bytes, 0);
    out.resize(data_start + other_offset, 0);
    for value in [1.5f32, -2.0, 3.0] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

// ---------------------------------------------------------------------------
// Synthetic tests (always run)
// ---------------------------------------------------------------------------

#[test]
fn synthetic_copy_is_lossless() {
    let dir = TempDir::new("copy");
    let source = dir.join("source.gguf");
    let output = dir.join("copy.gguf");
    fs::write(&source, synthetic_gguf()).unwrap();

    let reader = GgufReader::open(&source).unwrap();
    let writer = GgufWriter::from_source(&reader).unwrap();
    writer.write_to(&output).unwrap();

    assert_eq!(fs::read(&source).unwrap(), fs::read(&output).unwrap());
    let reopened = GgufReader::open(&output).unwrap();
    assert_eq!(reopened.tensors().len(), reader.tensors().len());
    assert_eq!(reopened.metadata().len(), reader.metadata().len());
    assert_eq!(
        reopened
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::array_elem_type),
        Some(8)
    );
}

#[test]
fn synthetic_append_tensor_vocab_and_metadata() {
    let dir = TempDir::new("append");
    let source = dir.join("source.gguf");
    let output = dir.join("fused.gguf");
    fs::write(&source, synthetic_gguf()).unwrap();

    let mut reader = GgufReader::open(&source).unwrap();
    let old_rows = reader.rows("token_embd.weight").unwrap();
    let mut source_rows = Vec::new();
    for row in 0..2 {
        let mut values = vec![0f32; 256];
        reader
            .read_row_f32("token_embd.weight", row, &mut values)
            .unwrap();
        source_rows.push(values);
    }

    let state = [1.0f32, -2.0, 3.5, 4.25];
    let tokens = vec!["x".to_string(), "y".to_string()];
    let scores = vec![-1.0f32, -2.0];
    let token_types = vec![4i32, 4];
    let new_rows: Vec<Vec<f32>> = (0..2)
        .map(|row| {
            (0..256)
                .map(|i| ((i as f32 + row as f32) / 256.0) * 2.0 - 1.0)
                .collect()
        })
        .collect();

    let mut writer = GgufWriter::from_source(&reader).unwrap();
    writer
        .append_tensor_f32("pmc.state.test", &[2, 2], &state)
        .unwrap();
    writer
        .set_metadata("pmc.version", GgufValue::U32(7))
        .unwrap();
    writer
        .set_metadata(
            "pmc.manifest",
            GgufValue::String("{\"dream\":1}".to_string()),
        )
        .unwrap();
    writer
        .append_vocab_tokens(&tokens, &scores, &token_types)
        .unwrap();
    writer.append_embedding_rows(&new_rows).unwrap();
    writer.write_to(&output).unwrap();

    let mut reopened = GgufReader::open(&output).unwrap();
    assert_eq!(reopened.tensors().len(), 4);
    assert_eq!(reopened.rows("token_embd.weight"), Some(old_rows + 2));
    assert_eq!(reopened.rows("output.weight"), Some(old_rows + 2));
    assert_eq!(array_len(&reopened, "tokenizer.ggml.tokens"), Some(6));
    assert_eq!(array_len(&reopened, "tokenizer.ggml.scores"), Some(6));
    assert_eq!(array_len(&reopened, "tokenizer.ggml.token_type"), Some(6));
    assert_eq!(
        reopened.get("pmc.version"),
        Some(&GgufValue::U32(7))
    );
    assert_eq!(
        reopened.get("pmc.manifest").and_then(GgufValue::as_str),
        Some("{\"dream\":1}")
    );

    let state_values = reopened
        .read_tensor_f32("pmc.state.test", 4)
        .expect("read appended tensor");
    assert_eq!(state_values.as_slice(), &state[..]);

    // The tensor that followed token_embd.weight must still be byte-identical.
    let source_other = tensor_offset(&reader, "other.weight");
    let output_other = tensor_offset(&reopened, "other.weight");
    assert_ranges_equal(
        &source,
        source_other,
        &output,
        output_other,
        3 * std::mem::size_of::<f32>() as u64,
    );

    for (index, original) in source_rows.iter().enumerate() {
        let mut decoded = vec![0f32; 256];
        reopened
            .read_row_f32("token_embd.weight", index, &mut decoded)
            .unwrap();
        let error = normalized_rms(original, &decoded);
        assert!(error < 0.02, "base row {index} changed: RMS {error}");
    }
    for (index, original) in new_rows.iter().enumerate() {
        let mut decoded = vec![0f32; 256];
        reopened
            .read_row_f32("token_embd.weight", old_rows + index, &mut decoded)
            .unwrap();
        let error = normalized_rms(original, &decoded);
        assert!(error < 0.02, "new row {index} RMS error {error}");

        let mut output_decoded = vec![0f32; 256];
        reopened
            .read_row_f32("output.weight", old_rows + index, &mut output_decoded)
            .unwrap();
        let output_error = normalized_rms(original, &output_decoded);
        assert!(
            output_error < 0.02,
            "output row {index} RMS error {output_error}"
        );
    }
}

// ---------------------------------------------------------------------------
// Model-dependent tests
// ---------------------------------------------------------------------------

#[test]
fn copy_is_lossless() {
    let Some(source) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let dir = TempDir::new("model-copy");
    let output = dir.join("copy.gguf");

    let reader = GgufReader::open(&source).expect("open source");
    let writer = GgufWriter::from_source(&reader).expect("create writer");
    writer.write_to(&output).expect("write copy");

    let reopened = GgufReader::open(&output).expect("open copy");
    assert_eq!(reopened.tensors().len(), reader.tensors().len());
    assert_eq!(reopened.metadata().len(), reader.metadata().len());
    assert_eq!(reader.data_start(), reopened.data_start());
    assert_ranges_equal(&source, 0, &output, 0, reader.data_start());

    let mut combined_hash = 0xcbf2_9ce4_8422_2325u64;
    for (source_info, output_info) in reader.tensors().iter().zip(reopened.tensors()) {
        assert_eq!(source_info.name, output_info.name);
        assert_eq!(source_info.dims, output_info.dims);
        assert_eq!(source_info.kind, output_info.kind);
        let source_len = supported_tensor_nbytes(&reader, &source_info.name);
        let output_len = supported_tensor_nbytes(&reopened, &output_info.name);
        assert_eq!(source_len, output_len);
        assert_ranges_equal(
            &source,
            tensor_offset(&reader, &source_info.name),
            &output,
            tensor_offset(&reopened, &source_info.name),
            source_len,
        );
        combined_hash ^= hash_range(
            &source,
            tensor_offset(&reader, &source_info.name),
            source_len,
        );
        combined_hash = combined_hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    let source_len = fs::metadata(&source).unwrap().len();
    let output_len = fs::metadata(&output).unwrap().len();
    eprintln!(
        "copy_is_lossless: source_len={source_len} output_len={output_len} tensors={} metadata={} combined_hash=0x{combined_hash:016x}",
        reopened.tensors().len(),
        reopened.metadata().len()
    );
    assert_eq!(source_len, output_len);
}

#[test]
fn append_tensor_and_metadata() {
    let Some(source) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let dir = TempDir::new("model-append");
    let output = dir.join("fused.gguf");

    let reader = GgufReader::open(&source).expect("open source");
    let mut writer = GgufWriter::from_source(&reader).expect("create writer");
    let data = [
        1.0f32, -2.5, 3.25, 4.0, 5.5, -6.0, 7.75, 8.125,
    ];
    writer
        .append_tensor_f32("pmc.state.test", &[4, 2], &data)
        .expect("append tensor");
    writer
        .set_metadata("pmc.version", GgufValue::U32(0x0001_0002))
        .expect("set version");
    writer
        .set_metadata(
            "pmc.manifest",
            GgufValue::String("{\"dream\":\"test\",\"n\":1}".to_string()),
        )
        .expect("set manifest");
    writer.write_to(&output).expect("write fused");

    let mut reopened = GgufReader::open(&output).expect("open fused");
    assert_eq!(reopened.tensors().len(), reader.tensors().len() + 1);
    assert_eq!(reopened.metadata().len(), reader.metadata().len() + 2);
    assert_eq!(
        reopened.get("pmc.version"),
        Some(&GgufValue::U32(0x0001_0002))
    );
    assert_eq!(
        reopened.get("pmc.manifest").and_then(GgufValue::as_str),
        Some("{\"dream\":\"test\",\"n\":1}")
    );
    let info = reopened.tensor("pmc.state.test").expect("new tensor");
    assert_eq!(info.dims, vec![4, 2]);
    assert_eq!(info.kind, GGML_TYPE_F32);
    let values = reopened.read_tensor_f32("pmc.state.test", 8).unwrap();
    assert_eq!(values, data);
    eprintln!(
        "append_tensor_and_metadata: source_len={} output_len={}",
        fs::metadata(&source).unwrap().len(),
        fs::metadata(&output).unwrap().len()
    );
}

#[test]
fn extend_vocab() {
    let Some(source) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let dir = TempDir::new("model-vocab");
    let output = dir.join("fused.gguf");
    let (row_len, old_rows, original_rows) = write_fused_fixture(&source, &output);
    let source_reader = GgufReader::open(&source).expect("open source");
    let source_kind = source_reader.tensor("token_embd.weight").unwrap().kind;
    let expected_rows = old_rows + 8;

    let source_size = fs::metadata(&source).unwrap().len();
    let output_size = fs::metadata(&output).unwrap().len();
    let mut reopened = GgufReader::open(&output).expect("open fused");
    assert_eq!(reopened.rows("token_embd.weight"), Some(expected_rows));
    assert_eq!(
        reopened.tensor("token_embd.weight").unwrap().dims,
        vec![row_len as u64, expected_rows as u64]
    );
    assert_eq!(
        reopened.tensor("token_embd.weight").unwrap().kind,
        source_kind
    );
    if source_reader.get("lfm2.vocab_size").is_some() {
        assert_eq!(
            reopened.get("lfm2.vocab_size"),
            Some(&GgufValue::U32(expected_rows as u32))
        );
    }
    assert_eq!(
        array_len(&reopened, "tokenizer.ggml.tokens"),
        Some(expected_rows)
    );
    assert_eq!(
        array_len(&reopened, "tokenizer.ggml.scores"),
        Some(expected_rows)
    );
    assert_eq!(
        array_len(&reopened, "tokenizer.ggml.token_type"),
        Some(expected_rows)
    );

    let tokens = reopened
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
        .expect("tokens array");
    assert_eq!(
        tokens.last(),
        Some(&GgufValue::String("<pmc_test_7>".to_string()))
    );
    let scores = reopened
        .get("tokenizer.ggml.scores")
        .and_then(GgufValue::as_array)
        .expect("scores array");
    assert_eq!(scores.last(), Some(&GgufValue::F32(0.0)));

    let threshold = match source_kind {
        GGML_TYPE_Q6_K => 0.02,
        GGML_TYPE_Q5_K => 0.03,
        GGML_TYPE_Q4_K => 0.06,
        _ => 0.02,
    };
    let mut max_rms = 0.0f32;
    for (index, original) in original_rows.iter().enumerate() {
        let mut decoded = vec![0f32; row_len];
        reopened
            .read_row_f32("token_embd.weight", old_rows + index, &mut decoded)
            .expect("read new row");
        let error = normalized_rms(original, &decoded);
        max_rms = max_rms.max(error);
        assert!(
            error < threshold,
            "new row {index} RMS error {error} >= {threshold}"
        );
    }
    eprintln!(
        "extend_vocab: source_len={source_size} output_len={output_size} delta={} rows={expected_rows} max_rms={max_rms:.6} kind={}",
        output_size as i128 - source_size as i128,
        pmc_gguf::quant_name(source_kind)
    );
}

#[test]
fn base_tensor_bytes_unchanged() {
    let Some(source) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let dir = TempDir::new("model-base");
    let output = dir.join("fused.gguf");
    let (_row_len, _old_rows, _rows) = write_fused_fixture(&source, &output);

    let reader = GgufReader::open(&source).expect("open source");
    let reopened = GgufReader::open(&output).expect("open fused");
    let mut checked = 0usize;
    let mut combined_hash = 0xcbf2_9ce4_8422_2325u64;
    for source_info in reader.tensors() {
        if source_info.name == "token_embd.weight" {
            continue;
        }
        let output_info = reopened
            .tensor(&source_info.name)
            .expect("tensor preserved");
        assert_eq!(source_info.dims, output_info.dims);
        assert_eq!(source_info.kind, output_info.kind);
        let len = supported_tensor_nbytes(&reader, &source_info.name);
        assert_eq!(len, supported_tensor_nbytes(&reopened, &source_info.name));
        assert_ranges_equal(
            &source,
            tensor_offset(&reader, &source_info.name),
            &output,
            tensor_offset(&reopened, &source_info.name),
            len,
        );
        combined_hash ^= hash_range(
            &source,
            tensor_offset(&reader, &source_info.name),
            len,
        );
        combined_hash = combined_hash.wrapping_mul(0x0000_0100_0000_01b3);
        checked += 1;
    }
    assert_eq!(checked, reader.tensors().len() - 1);
    eprintln!(
        "base_tensor_bytes_unchanged: checked={checked} combined_hash=0x{combined_hash:016x}"
    );
}

#[test]
#[ignore = "requires PMC_LLAMA_CLI and PMC_TEST_GGUF"]
fn llama_cpp_loads_fused() {
    let Some(source) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let Some(cli) = std::env::var_os("PMC_LLAMA_CLI").map(PathBuf::from) else {
        eprintln!("skipping: PMC_LLAMA_CLI is not set");
        return;
    };
    assert!(
        cli.is_file(),
        "PMC_LLAMA_CLI does not point to a file: {}",
        cli.display()
    );

    let dir = TempDir::new("llama");
    let output = dir.join("fused.gguf");
    let (_row_len, old_rows, _rows) = write_fused_fixture(&source, &output);
    let expected_vocab = old_rows + 8;
    if let Some(keep) = std::env::var_os("PMC_KEEP_FUSED") {
        fs::copy(&output, &keep).expect("copy fused GGUF for external inspection");
        eprintln!("kept fused GGUF at {}", Path::new(&keep).display());
    }

    let mut library_path = cli.parent().unwrap().as_os_str().to_os_string();
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        library_path.push(":");
        library_path.push(existing);
    }

    // The user-facing command: --log-disable and one generated token. This
    // llama-cli version stays in interactive mode unless -st is supplied, so
    // -st is added to make the process terminate.
    let first = Command::new(&cli)
        .env("LD_LIBRARY_PATH", &library_path)
        .arg("-m")
        .arg(&output)
        .arg("-p")
        .arg("hello")
        .arg("-n")
        .arg("1")
        .arg("-c")
        .arg("64")
        .arg("--log-disable")
        .arg("-st")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run llama-cli with --log-disable");
    assert!(
        first.status.success(),
        "llama-cli --log-disable failed: {}\nstdout:\n{}\nstderr:\n{}",
        first.status,
        String::from_utf8_lossy(&first.stdout),
        String::from_utf8_lossy(&first.stderr)
    );

    // --log-disable pauses llama.cpp's logger and discards INFO messages, so
    // run once more with verbose logging to capture the n_vocab line. -st
    // keeps this to a single turn.
    let second = Command::new(&cli)
        .env("LD_LIBRARY_PATH", &library_path)
        .arg("-m")
        .arg(&output)
        .arg("-p")
        .arg("hello")
        .arg("-n")
        .arg("1")
        .arg("-c")
        .arg("64")
        .arg("-v")
        .arg("-st")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run llama-cli with verbose logging");
    assert!(
        second.status.success(),
        "llama-cli -v failed: {}\nstdout:\n{}\nstderr:\n{}",
        second.status,
        String::from_utf8_lossy(&second.stdout),
        String::from_utf8_lossy(&second.stderr)
    );

    let stderr = String::from_utf8_lossy(&second.stderr);
    let n_vocab = parse_n_vocab(&stderr).expect("n_vocab line in llama-cli stderr");
    assert_eq!(
        n_vocab, expected_vocab as u32,
        "llama.cpp reported the wrong vocabulary size"
    );
    let expected_text = expected_vocab.to_string();
    let matching_line = stderr
        .lines()
        .find(|line| {
            line.contains("n_vocab")
                && line
                    .split_whitespace()
                    .any(|piece| piece == expected_text.as_str())
        })
        .expect("matching n_vocab line");
    eprintln!(
        "llama-cli command: {} -m {} -p hello -n 1 -c 64 -v -st",
        cli.display(),
        output.display()
    );
    eprintln!("llama-cli n_vocab: {}", matching_line.trim());
}

fn parse_n_vocab(text: &str) -> Option<u32> {
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        while let Some(part) = parts.next() {
            if part != "n_vocab" {
                continue;
            }
            let equals = parts.next()?;
            if equals != "=" {
                continue;
            }
            let value = parts.next()?;
            return value.parse().ok();
        }
    }
    None
}
