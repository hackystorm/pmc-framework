// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end tests for `pmc fuse`.
//!
//! The synthetic test always runs: it builds a four-token Q6_K GGUF, fuses two
//! rows plus one state tensor, and checks both the appended data and every
//! preserved base byte.
//!
//! The 350M test reads `PMC_TEST_GGUF` and returns early when it is unset. It
//! also runs `llama-cli` when `PMC_LLAMA_CLI` is set. Only the 350M model is
//! used; the 8B model is never loaded here.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use pmc_cli::fuse::{
    fuse, FuseRequest, DELTA_HEADER_BYTES, DELTA_MAGIC, DELTA_VERSION, MANIFEST_FORMAT,
};
use pmc_core::{sha256_hex, Json};
use pmc_gguf::{
    quant_geometry, GgufReader, GgufValue, GGML_TYPE_F32, GGML_TYPE_Q6_K,
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
        path.push(format!(
            "pmc-cli-fuse-{}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
            tag
        ));
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

fn env_path(name: &str) -> Option<PathBuf> {
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

fn env_optional(name: &str) -> Option<PathBuf> {
    let value = std::env::var_os(name)?;
    let path = PathBuf::from(value);
    assert!(
        path.is_file(),
        "{name} is set but does not point to a file: {}",
        path.display()
    );
    Some(path)
}

fn write_delta(path: &Path, rows: &[Vec<f32>]) {
    let n_rows = rows.len();
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let mut bytes = Vec::with_capacity(DELTA_HEADER_BYTES + n_rows * dim * 4);
    bytes.extend_from_slice(&DELTA_MAGIC);
    bytes.extend_from_slice(&DELTA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(n_rows as u32).to_le_bytes());
    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    for row in rows {
        assert_eq!(row.len(), dim);
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fs::write(path, bytes).expect("write delta");
}

fn state_metadata(reader: &GgufReader, name: &str) -> (Vec<u64>, Vec<f32>) {
    let dims = reader
        .get(&format!("{name}.dims"))
        .and_then(GgufValue::as_array)
        .expect("state dims metadata")
        .iter()
        .map(|value| value.as_u64().expect("u32 state dimension"))
        .collect();
    let data = reader
        .get(&format!("{name}.data"))
        .and_then(GgufValue::as_array)
        .expect("state data metadata")
        .iter()
        .map(|value| match value {
            GgufValue::F32(value) => *value,
            other => panic!("state data is not f32: {other:?}"),
        })
        .collect();
    (dims, data)
}

fn synthetic_meta() -> String {
    Json::obj(vec![
        (
            "tokens",
            Json::Arr(vec![Json::str("<pmc:s0>"), Json::str("<pmc:s1>")]),
        ),
        ("scores", Json::Arr(vec![Json::num(0.0), Json::num(-0.5)])),
        (
            "token_types",
            Json::Arr(vec![Json::num(4.0), Json::num(4.0)]),
        ),
        (
            "state_tensors",
            Json::Obj(vec![(
                "pmc.state.bank".to_string(),
                Json::obj(vec![
                    ("dims", Json::Arr(vec![Json::num(2.0), Json::num(2.0)])),
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
        ("dream_id", Json::str("dream-synthetic")),
        (
            "provenance",
            Json::Arr(vec![Json::str("sha256:0000000000000000")]),
        ),
    ])
    .to_string()
}

fn array_len(reader: &GgufReader, key: &str) -> Option<usize> {
    reader
        .get(key)
        .and_then(GgufValue::as_array)
        .map(|items| items.len())
}

fn tensor_prefix_nbytes(reader: &GgufReader, name: &str) -> u64 {
    let info = reader.tensor(name).expect("tensor present");
    let (block_elems, block_bytes) = quant_geometry(info.kind).expect("known tensor type");
    let elements = info.elements().expect("element count") as u64;
    (elements / block_elems as u64) * block_bytes as u64
}

fn tensor_offset(reader: &GgufReader, name: &str) -> u64 {
    let info = reader.tensor(name).expect("tensor present");
    reader.data_start() + info.offset
}

fn hash_range(path: &Path, offset: u64, len: u64) -> u64 {
    let mut file = File::open(path).expect("open file for hashing");
    file.seek(SeekFrom::Start(offset)).expect("seek for hashing");
    let mut buffer = vec![0u8; 1024 * 1024];
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

fn assert_rows_bit_equal(
    left: &mut GgufReader,
    left_name: &str,
    right: &mut GgufReader,
    right_name: &str,
    rows: &[usize],
    row_len: usize,
) {
    for &row in rows {
        let mut left_values = vec![0f32; row_len];
        let mut right_values = vec![0f32; row_len];
        left.read_row_f32(left_name, row, &mut left_values)
            .expect("read left row");
        right
            .read_row_f32(right_name, row, &mut right_values)
            .expect("read right row");
        for (index, (left, right)) in left_values.iter().zip(&right_values).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{left_name} row {row} element {index} changed"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic GGUF source (four tokens, Q6_K embeddings)
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

fn push_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn synthetic_header(token_embd_offset: u64, output_offset: u64, other_offset: u64) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x4655_4747); // "GGUF"
    push_u32(&mut out, 3);
    push_u64(&mut out, 3); // tensors
    push_u64(&mut out, 4); // metadata entries

    push_string(&mut out, "general.alignment");
    push_u32(&mut out, 4); // u32
    push_u32(&mut out, 32);

    push_string(&mut out, "tokenizer.ggml.tokens");
    push_u32(&mut out, 9); // array
    push_u32(&mut out, 8); // string elements
    push_u64(&mut out, 4);
    for token in ["a", "b", "c", "d"] {
        push_string(&mut out, token);
    }

    push_string(&mut out, "tokenizer.ggml.scores");
    push_u32(&mut out, 9);
    push_u32(&mut out, 6); // f32 elements
    push_u64(&mut out, 4);
    for _ in 0..4 {
        push_f32(&mut out, -1.0);
    }

    push_string(&mut out, "tokenizer.ggml.token_type");
    push_u32(&mut out, 9);
    push_u32(&mut out, 5); // i32 elements
    push_u64(&mut out, 4);
    for _ in 0..4 {
        push_i32(&mut out, 1);
    }

    push_string(&mut out, "token_embd.weight");
    push_u32(&mut out, 2);
    push_u64(&mut out, 256);
    push_u64(&mut out, 4);
    push_u32(&mut out, GGML_TYPE_Q6_K);
    push_u64(&mut out, token_embd_offset);

    push_string(&mut out, "output.weight");
    push_u32(&mut out, 2);
    push_u64(&mut out, 256);
    push_u64(&mut out, 4);
    push_u32(&mut out, GGML_TYPE_Q6_K);
    push_u64(&mut out, output_offset);

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
    out.resize(data_start + q6_single_tensor_padded, 0);
    out.resize(data_start + output_offset + q6_single_tensor_padded, 0);
    out.resize(data_start + other_offset, 0);
    for value in [1.5f32, -2.0, 3.0] {
        push_f32(&mut out, value);
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

fn synthetic_rows() -> Vec<Vec<f32>> {
    (0..2)
        .map(|row| {
            (0..256)
                .map(|index| ((index as f32 + row as f32) / 256.0) * 2.0 - 1.0)
                .collect()
        })
        .collect()
}

fn assert_no_staging_files(dir: &TempDir) {
    for entry in fs::read_dir(&dir.path).expect("list temp dir") {
        let name = entry
            .expect("dir entry")
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(
            !name.contains(".tmp-"),
            "staging file was not cleaned up: {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Always-on tests
// ---------------------------------------------------------------------------

#[test]
fn synthetic_fusion_appends_and_preserves() {
    let dir = TempDir::new("synthetic");
    let base_path = dir.join("base.gguf");
    let rows_path = dir.join("rows.bin");
    let meta_path = dir.join("meta.json");
    let out_path = dir.join("fused.gguf");
    let manifest_path = dir.join("manifest.json");
    fs::write(&base_path, synthetic_gguf()).expect("write base");
    fs::write(&meta_path, synthetic_meta()).expect("write meta");
    write_delta(&rows_path, &synthetic_rows());

    let report = fuse(&FuseRequest {
        base: base_path.clone(),
        rows: rows_path.clone(),
        meta: meta_path.clone(),
        out: out_path.clone(),
        manifest: Some(manifest_path.clone()),
    })
    .expect("fuse synthetic model");

    assert_eq!(report.old_vocab, 4);
    assert_eq!(report.new_vocab, 6);
    assert_eq!(report.token_first, 4);
    assert_eq!(report.token_last, 5);
    assert_eq!(report.rows, 2);
    assert_eq!(report.dim, 256);
    assert_eq!(report.dream_id.as_deref(), Some("dream-synthetic"));
    assert_eq!(report.provenance, vec!["sha256:0000000000000000"]);
    assert_eq!(
        report.rows_sha256,
        sha256_hex(&fs::read(&rows_path).expect("read delta"))
    );
    assert_eq!(
        report.meta_sha256,
        sha256_hex(&fs::read(&meta_path).expect("read meta"))
    );
    assert_eq!(
        report.base_sha256,
        sha256_hex(&fs::read(&base_path).expect("read base"))
    );
    assert_eq!(
        report.output_sha256,
        sha256_hex(&fs::read(&out_path).expect("read fused"))
    );
    assert!(report.base_identity.starts_with("fnv1a64:"));
    assert!(report.output_identity.starts_with("fnv1a64:"));
    assert!(report.output_size > report.base_size);
    assert_eq!(report.manifest_path, Some(manifest_path.display().to_string()));

    let mut base = GgufReader::open(&base_path).expect("open base");
    let mut fused = GgufReader::open(&out_path).expect("open fused");
    assert_eq!(fused.rows("token_embd.weight"), Some(6));
    assert_eq!(fused.rows("output.weight"), Some(6));
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    ] {
        assert_eq!(array_len(&fused, key), Some(6), "{key}");
    }

    assert_eq!(fused.get("pmc.version"), Some(&GgufValue::U32(1)));
    assert_eq!(
        fused.get("pmc.dream_id").and_then(GgufValue::as_str),
        Some("dream-synthetic")
    );
    let embedded = fused
        .get("pmc.manifest")
        .and_then(GgufValue::as_str)
        .expect("embedded manifest");
    assert!(
        embedded.contains(&format!("\"format\":\"{MANIFEST_FORMAT}\"")),
        "{embedded}"
    );
    assert!(embedded.contains("\"first_index\":4"), "{embedded}");
    assert!(embedded.contains("\"file\":\"base.gguf\""), "{embedded}");
    assert!(
        embedded.contains(&report.base_sha256),
        "embedded manifest must carry the base sha256: {embedded}"
    );
    assert!(
        !embedded.contains(dir.path.to_str().expect("temp path")),
        "embedded manifest must not contain a local path: {embedded}"
    );
    assert!(!embedded.contains("\"output\""), "embedded manifest must not describe itself: {embedded}");

    let tokens = fused
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
        .expect("tokens");
    assert_eq!(
        tokens[4],
        GgufValue::String("<pmc:s0>".to_string())
    );
    assert_eq!(
        tokens[5],
        GgufValue::String("<pmc:s1>".to_string())
    );
    let scores = fused
        .get("tokenizer.ggml.scores")
        .and_then(GgufValue::as_array)
        .expect("scores");
    assert_eq!(scores[4], GgufValue::F32(0.0));
    assert_eq!(scores[5], GgufValue::F32(-0.5));
    let types = fused
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_array)
        .expect("token types");
    assert_eq!(types[4], GgufValue::I32(4));
    assert_eq!(types[5], GgufValue::I32(4));

    let (state_dims, state) = state_metadata(&fused, "pmc.state.bank");
    assert_eq!(state_dims, vec![2, 2]);
    assert_eq!(
        state.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        vec![1.0f32, -2.0, 3.5, -4.25]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    // Every base row decodes to exactly the same bits in the fused file.
    assert_rows_bit_equal(&mut base, "token_embd.weight", &mut fused, "token_embd.weight", &[0, 1, 2, 3], 256);
    assert_rows_bit_equal(&mut base, "output.weight", &mut fused, "output.weight", &[0, 1, 2, 3], 256);
    assert_rows_bit_equal(&mut base, "other.weight", &mut fused, "other.weight", &[0], 3);

    // The raw bytes of the untouched tensor are preserved as well.
    assert_eq!(
        hash_range(&base_path, tensor_offset(&base, "other.weight"), tensor_prefix_nbytes(&base, "other.weight")),
        hash_range(&out_path, tensor_offset(&fused, "other.weight"), tensor_prefix_nbytes(&fused, "other.weight"))
    );

    let manifest = fs::read_to_string(&manifest_path).expect("read manifest");
    assert!(manifest.contains(&format!("\"format\":\"{MANIFEST_FORMAT}\"")), "{manifest}");
    assert!(manifest.contains("\"output\":{\"file\":\"fused.gguf\""), "{manifest}");
    assert!(manifest.contains("\"file\":\"base.gguf\""), "{manifest}");
    assert!(
        manifest.contains(&report.base_sha256),
        "manifest must carry the base sha256: {manifest}"
    );
    assert!(
        manifest.contains(&report.output_sha256),
        "manifest must carry the output sha256: {manifest}"
    );
    assert!(
        !manifest.contains(dir.path.to_str().expect("temp path")),
        "manifest must not contain a local path: {manifest}"
    );
    assert!(
        manifest.contains(&report.output_identity),
        "manifest must carry the output identity: {manifest}"
    );
    assert!(manifest.contains("\"dream_id\":\"dream-synthetic\""), "{manifest}");
    assert_no_staging_files(&dir);

    eprintln!(
        "synthetic_fusion_appends_and_preserves: base={} output={} base_identity={} output_identity={} old_vocab={} new_vocab={}",
        report.base_size,
        report.output_size,
        report.base_identity,
        report.output_identity,
        report.old_vocab,
        report.new_vocab
    );
}

#[test]
fn failed_delta_leaves_output_and_manifest_untouched() {
    let dir = TempDir::new("failure");
    let base_path = dir.join("base.gguf");
    let rows_path = dir.join("rows.bin");
    let meta_path = dir.join("meta.json");
    let out_path = dir.join("fused.gguf");
    let manifest_path = dir.join("manifest.json");
    fs::write(&base_path, synthetic_gguf()).expect("write base");
    fs::write(&meta_path, synthetic_meta()).expect("write meta");
    fs::write(&rows_path, b"PMCR\x01\x00\x00\x00").expect("write short delta");
    let out_sentinel = b"previous fused checkpoint".to_vec();
    let manifest_sentinel = b"previous manifest".to_vec();
    fs::write(&out_path, &out_sentinel).expect("write sentinel output");
    fs::write(&manifest_path, &manifest_sentinel).expect("write sentinel manifest");

    let error = fuse(&FuseRequest {
        base: base_path.clone(),
        rows: rows_path.clone(),
        meta: meta_path,
        out: out_path.clone(),
        manifest: Some(manifest_path.clone()),
    })
    .expect_err("truncated delta must be refused");
    assert!(error.contains("truncated"), "{error}");
    assert_eq!(fs::read(&out_path).expect("read output"), out_sentinel);
    assert_eq!(
        fs::read(&manifest_path).expect("read manifest"),
        manifest_sentinel
    );
    assert_no_staging_files(&dir);
}

#[test]
fn output_must_differ_from_base() {
    let dir = TempDir::new("collision");
    let base_path = dir.join("base.gguf");
    let rows_path = dir.join("rows.bin");
    let meta_path = dir.join("meta.json");
    fs::write(&base_path, synthetic_gguf()).expect("write base");
    fs::write(&meta_path, synthetic_meta()).expect("write meta");
    write_delta(&rows_path, &synthetic_rows());
    let before = sha256_hex(&fs::read(&base_path).expect("read base"));

    let error = fuse(&FuseRequest {
        base: base_path.clone(),
        rows: rows_path.clone(),
        meta: meta_path,
        out: base_path.clone(),
        manifest: None,
    })
    .expect_err("output may not overwrite the base");
    assert!(error.contains("must be different files"), "{error}");
    assert_eq!(
        sha256_hex(&fs::read(&base_path).expect("read base")),
        before,
        "base changed after a refused fusion"
    );
    assert_no_staging_files(&dir);
}

// ---------------------------------------------------------------------------
// 350M model test
// ---------------------------------------------------------------------------

#[test]
fn fuse_350m_model() {
    let Some(source) = env_path("PMC_TEST_GGUF") else {
        return;
    };
    let started = Instant::now();
    let dir = TempDir::new("model");
    let rows_path = dir.join("rows.bin");
    let meta_path = dir.join("meta.json");
    let out_path = dir.join("fused.gguf");
    let manifest_path = dir.join("manifest.json");

    let mut base = GgufReader::open(&source).expect("open source model");
    let old_rows = base.rows("token_embd.weight").expect("rows");
    let dim = base.row_len("token_embd.weight").expect("row length");
    assert_eq!(
        old_rows, 65536,
        "PMC_TEST_GGUF must be the LFM2.5-350M model with 65536 embedding rows"
    );

    let mut delta_rows = Vec::with_capacity(8);
    for row in 0..8 {
        let mut values = vec![0f32; dim];
        base.read_row_f32("token_embd.weight", row, &mut values)
            .expect("read source row");
        delta_rows.push(values);
    }
    write_delta(&rows_path, &delta_rows);

    fs::write(&meta_path, model_meta(true)).expect("write meta");

    let report = fuse(&FuseRequest {
        base: source.clone(),
        rows: rows_path.clone(),
        meta: meta_path.clone(),
        out: out_path.clone(),
        manifest: Some(manifest_path.clone()),
    })
    .expect("fuse 350M model");

    assert_eq!(report.old_vocab, 65536);
    assert_eq!(report.new_vocab, 65544);
    assert_eq!(report.rows, 8);
    assert_eq!(report.dim, dim);

    let mut fused = GgufReader::open(&out_path).expect("reopen fused");
    assert_eq!(
        fused.rows("token_embd.weight"),
        Some(65544),
        "fused token_embd.weight must have 65544 rows"
    );
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    ] {
        assert_eq!(array_len(&fused, key), Some(65544), "{key}");
    }
    let types = fused
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_array)
        .expect("token types");
    for index in 65536..65544 {
        assert_eq!(types[index], GgufValue::I32(4), "token {index} type");
    }
    let tokens = fused
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_array)
        .expect("tokens");
    assert_eq!(
        tokens[65543],
        GgufValue::String("<pmc:test7>".to_string())
    );

    let (state_dims, state) = state_metadata(&fused, "pmc.state.bank");
    assert_eq!(state_dims, vec![2, 4]);
    assert_eq!(
        state.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        vec![0.5f32, -1.0, 2.25, 3.0, -0.125, 4.5, -5.0, 0.75]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        fused.get("pmc.dream_id").and_then(GgufValue::as_str),
        Some("wp3-350m-test")
    );

    // Independent byte check of every base tensor. token_embd.weight and
    // output.weight get rows appended, so only their base prefix is compared.
    let source_len = fs::metadata(&source).expect("source metadata").len();
    let mut checked = 0usize;
    for info in base.tensors() {
        let fused_info = fused.tensor(&info.name).expect("tensor preserved");
        assert_eq!(info.kind, fused_info.kind, "{}", info.name);
        let prefix = tensor_prefix_nbytes(&base, &info.name);
        assert_eq!(
            hash_range(&source, tensor_offset(&base, &info.name), prefix),
            hash_range(&out_path, tensor_offset(&fused, &info.name), prefix),
            "base bytes of {:?} changed",
            info.name
        );
        if info.name == "token_embd.weight" || info.name == "output.weight" {
            assert_eq!(
                fused_info.dims[1],
                info.dims[1] + 8,
                "{} appended rows",
                info.name
            );
        } else {
            assert_eq!(info.dims, fused_info.dims, "{} dims", info.name);
        }
        checked += 1;
    }
    assert_eq!(checked, base.tensors().len());
    assert_eq!(
        fused.tensors().len(),
        base.tensors().len(),
        "state metadata must not add tensors"
    );

    assert_rows_bit_equal(
        &mut base,
        "token_embd.weight",
        &mut fused,
        "token_embd.weight",
        &[0, 1, 7, 1234, 32768, 65534, 65535],
        dim,
    );

    let manifest = fs::read_to_string(&manifest_path).expect("read manifest");
    assert!(manifest.contains(&format!("\"format\":\"{MANIFEST_FORMAT}\"")), "{manifest}");
    assert!(manifest.contains(&report.output_identity), "{manifest}");
    assert!(manifest.contains("\"new_count\":65544"), "{manifest}");
    assert_no_staging_files(&dir);

    // A refused delta must leave the already published checkpoint and sidecar
    // byte-identical. The delta parser rejects this before any model I/O.
    let published = sha256_hex(&fs::read(&out_path).expect("read published fused"));
    let published_manifest = sha256_hex(&fs::read(&manifest_path).expect("read published manifest"));
    let broken_rows = dir.join("broken_rows.bin");
    fs::write(&broken_rows, b"PMCR\x01\x00\x00\x00").expect("write broken delta");
    let refusal = fuse(&FuseRequest {
        base: source.clone(),
        rows: broken_rows,
        meta: meta_path.clone(),
        out: out_path.clone(),
        manifest: Some(manifest_path.clone()),
    })
    .expect_err("truncated delta must be refused on the real artifact");
    assert!(refusal.contains("truncated"), "{refusal}");
    assert_eq!(
        sha256_hex(&fs::read(&out_path).expect("read fused after refusal")),
        published,
        "published fused file changed after a refused delta"
    );
    assert_eq!(
        sha256_hex(&fs::read(&manifest_path).expect("read manifest after refusal")),
        published_manifest,
        "published manifest changed after a refused delta"
    );
    assert_no_staging_files(&dir);

    let elapsed = started.elapsed();
    eprintln!(
        "fuse_350m_model: source_len={source_len} output_len={} base_identity={} output_identity={} rows={} new_vocab={} tensors_checked={checked} elapsed_ms={}",
        report.output_size,
        report.base_identity,
        report.output_identity,
        report.rows,
        report.new_vocab,
        elapsed.as_millis()
    );

    if let Some(cli) = env_optional("PMC_LLAMA_CLI") {
        let mut library_path = cli.parent().expect("llama-cli parent").as_os_str().to_os_string();
        if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
            library_path.push(":");
            library_path.push(existing);
        }
        let run = Command::new(&cli)
            .env("LD_LIBRARY_PATH", &library_path)
            .arg("-m")
            .arg(&out_path)
            .arg("-p")
            .arg("hello")
            .arg("-n")
            .arg("1")
            .arg("-c")
            .arg("64")
            .arg("-v")
            .arg("-st")
            .stdin(Stdio::null())
            .output()
            .expect("run llama-cli");
        assert!(
            run.status.success(),
            "llama-cli failed: {}\nstdout:\n{}\nstderr:\n{}",
            run.status,
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        let stderr = String::from_utf8_lossy(&run.stderr);
        let n_vocab = parse_n_vocab(&stderr).expect("n_vocab line in llama-cli stderr");
        assert_eq!(n_vocab, 65544, "llama.cpp reported the wrong vocabulary size");
        let matching_line = stderr
            .lines()
            .find(|line| {
                line.contains("n_vocab")
                    && line.split_whitespace().any(|piece| piece == "65544")
            })
            .expect("matching n_vocab line");
        eprintln!(
            "llama-cli command: {} -m {} -p hello -n 1 -c 64 -v -st",
            cli.display(),
            out_path.display()
        );
        eprintln!("llama-cli n_vocab: {}", matching_line.trim());
    } else {
        eprintln!("skipping llama-cli: PMC_LLAMA_CLI is not set");
    }
}

fn model_meta(with_state: bool) -> String {
    let mut entries: Vec<(&str, Json)> = vec![
        (
            "tokens",
            Json::Arr(
                (0..8)
                    .map(|index| Json::str(format!("<pmc:test{index}>")))
                    .collect(),
            ),
        ),
        (
            "scores",
            Json::Arr((0..8).map(|_| Json::num(0.0)).collect()),
        ),
        (
            "token_types",
            Json::Arr((0..8).map(|_| Json::num(4.0)).collect()),
        ),
    ];
    if with_state {
        entries.push((
            "state_tensors",
            Json::Obj(vec![(
                "pmc.state.bank".to_string(),
                Json::obj(vec![
                    // 2 x 4 = 8 elements, the shape required by the test plan.
                    ("dims", Json::Arr(vec![Json::num(2.0), Json::num(4.0)])),
                    (
                        "data",
                        Json::Arr(vec![
                            Json::num(0.5),
                            Json::num(-1.0),
                            Json::num(2.25),
                            Json::num(3.0),
                            Json::num(-0.125),
                            Json::num(4.5),
                            Json::num(-5.0),
                            Json::num(0.75),
                        ]),
                    ),
                ]),
            )]),
        ));
    }
    entries.push(("dream_id", Json::str("wp3-350m-test")));
    entries.push((
        "provenance",
        Json::Arr(vec![Json::str("sha256:0123456789abcdef")]),
    ));
    Json::obj(entries).to_string()
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
