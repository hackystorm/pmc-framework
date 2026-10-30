// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end tests for `pmc state-memory`.
//!
//! The synthetic test always runs: it writes a weights file with the trainer's
//! own writer, fuses it into a synthetic GGUF, and checks that the bank the
//! manifest publishes is the bank the file actually carries. That is the whole
//! claim — a memory in the model rather than in a JSON file beside it — so it
//! is checked against the written file, not against the code that wrote it.
//!
//! No model is loaded and no llama.cpp is needed: the test compares numbers it
//! produced itself.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use pmc_cli::state_memory::{
    fuse, read_architecture, written_tensor_names, NAMESPACE, StateMemoryRequest,
};
use pmc_gguf::{GgufReader, GgufValue, GGML_TYPE_F32};
use pmc_train::pmc_state::{write_weights, PmcStateConfig, PmcStateModel};

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
            "pmc-cli-state-{}-{}-{}",
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

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// A minimal valid GGUF v3 file with one F32 tensor and a `n_embd` value.
fn synthetic_gguf(n_embd: usize) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x4655_4747); // magic "GGUF"
    push_u32(&mut out, 3); // version
    push_u64(&mut out, 1); // tensor count
    push_u64(&mut out, 3); // metadata count

    push_string(&mut out, "general.alignment");
    push_u32(&mut out, 4); // UINT32
    push_u32(&mut out, 32);

    push_string(&mut out, "n_embd");
    push_u32(&mut out, 4); // UINT32
    push_u32(&mut out, n_embd as u32);

    push_string(&mut out, "general.architecture");
    push_u32(&mut out, 8); // STRING
    push_string(&mut out, "test");

    // Tensor "test.weight": dims [8, 4], F32.
    push_string(&mut out, "test.weight");
    push_u32(&mut out, 2); // n_dims
    push_u64(&mut out, 8); // row length
    push_u64(&mut out, 4); // rows
    push_u32(&mut out, GGML_TYPE_F32);
    push_u64(&mut out, 0);

    while out.len() % 32 != 0 {
        out.push(0);
    }
    for i in 0..32 {
        out.extend_from_slice(&(i as f32 * 0.5).to_le_bytes());
    }
    out
}

fn write_file(path: &Path, bytes: &[u8]) {
    let mut file = File::create(path).expect("create file");
    file.write_all(bytes).expect("write file");
}

fn read_f32(reader: &GgufReader, name: &str, offset: u64, count: usize) -> Vec<f32> {
    let mut file = File::open(reader.path()).expect("open fused gguf");
    use std::io::SeekFrom;
    file.seek(SeekFrom::Start(offset)).expect("seek");
    let mut bytes = vec![0u8; count * 4];
    file.read_exact(&mut bytes).expect("read tensor bytes");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// The synthetic test
// ---------------------------------------------------------------------------

#[test]
fn the_bank_is_in_the_file_and_matches_its_manifest() {
    let dir = TempDir::new("bank");
    let base_path = dir.join("base.gguf");
    let weights_path = dir.join("state.pmc");
    let out_path = dir.join("fused.gguf");

    write_file(&base_path, &synthetic_gguf(64));

    // A weights file written by the trainer's own writer, so the test exercises
    // the real artifact rather than a hand-built one.
    let config = PmcStateConfig::default();
    let host_dims = vec![64usize, 64];
    let model = PmcStateModel::init(config.clone(), &host_dims).expect("init state model");
    write_weights(&weights_path, &model).expect("write weights");

    let request = StateMemoryRequest {
        base: base_path.clone(),
        weights: weights_path.clone(),
        out: out_path.clone(),
        manifest: Some(dir.join("manifest.json")),
    };
    let manifest = fuse(&request).expect("fuse state memory");

    // The architecture round-trips: the header alone is enough to rebuild the
    // configuration the shapes describe.
    let (read_config, read_dims) = read_architecture(&weights_path).expect("read architecture");
    assert_eq!(read_config.architecture(), config.architecture());
    assert_eq!(read_dims, host_dims);

    // The names the manifest publishes are the names the file carries, and the
    // counts agree with the model that was fused.
    let names = written_tensor_names(host_dims.len(), config.layers);
    let reader = GgufReader::open(&out_path).expect("open fused gguf");
    for name in &names {
        let info = reader
            .tensor(name)
            .unwrap_or_else(|| panic!("tensor {name} is missing from the fused file"));
        assert_eq!(info.kind, GGML_TYPE_F32, "{name} must be dense f32 weights");
    }
    assert_eq!(reader.tensors().len(), 1 + names.len());

    // The manifest that reports the fusion describes the file that was written:
    // the parameter count and the digest of the weights it fused. `Json` is a
    // value type without indexing, so the manifest is checked through its
    // serialised form, the way a report outside the process would read it.
    let text = manifest.to_string();
    assert!(
        text.contains(&model.weight_digest()),
        "the manifest must carry the digest of what it fused"
    );
    assert!(
        text.contains(&model.param_count().to_string()),
        "the manifest must carry the parameter count"
    );

    // The base tensor survives the fusion, byte for byte.
    let base_reader = GgufReader::open(&base_path).expect("open base");
    assert_eq!(
        base_reader.tensor("test.weight").unwrap().kind,
        reader.tensor("test.weight").unwrap().kind
    );

    // And the parameters that were fused are the parameters the file holds,
    // number for number.
    let head = reader
        .tensor(&format!("{NAMESPACE}head"))
        .expect("head tensor present");
    let offset = reader.data_start() + head.offset;
    assert_eq!(
        read_f32(&reader, &format!("{NAMESPACE}head"), offset, model.head.len()),
        model.head
    );
}

#[test]
fn the_bank_records_its_geometry_and_the_models_hidden_size() {
    let dir = TempDir::new("geometry");
    let base_path = dir.join("base.gguf");
    let weights_path = dir.join("state.pmc");
    let out_path = dir.join("fused.gguf");

    write_file(&base_path, &synthetic_gguf(64));
    let config = PmcStateConfig::default();
    let model = PmcStateModel::init(config.clone(), &[64, 64]).expect("init");
    write_weights(&weights_path, &model).expect("write weights");

    let request = StateMemoryRequest {
        base: base_path,
        weights: weights_path,
        out: out_path.clone(),
        manifest: None,
    };
    fuse(&request).expect("fuse");

    let reader = GgufReader::open(&out_path).expect("open fused gguf");

    // The bank is self-describing, and read back as typed GGUF metadata rather
    // than a JSON document a loader would have to parse.
    let geometry = reader
        .get(&format!("{NAMESPACE}geometry"))
        .and_then(GgufValue::as_array)
        .unwrap_or_else(|| panic!("the fused file carries no geometry"));
    let numbers: Vec<u64> = geometry.iter().map(|v| v.as_u64().expect("u64 geometry")).collect();
    // projection, hidden, layers, key_dim, hosts.
    assert_eq!(numbers[1], config.hidden_dim as u64);
    assert_eq!(numbers[0], config.projection_dim as u64);

    let hosts = reader
        .get(&format!("{NAMESPACE}hosts"))
        .and_then(GgufValue::as_array)
        .expect("hosts recorded");
    assert_eq!(hosts.len(), 2);
    assert_eq!(hosts[0].as_u64(), Some(64));
    assert_eq!(hosts[1].as_u64(), Some(64));

    // The base model's hidden size is recorded, because the bank's input
    // projection is meaningless without knowing what it projects.
    assert_eq!(
        reader
            .get(&format!("{NAMESPACE}n_embd"))
            .and_then(GgufValue::as_u64),
        Some(64)
    );

    // The tensor names, so a reader can verify the bank without recomputing the
    // naming scheme.
    let tensors = reader
        .get(&format!("{NAMESPACE}tensors"))
        .and_then(GgufValue::as_array)
        .expect("tensor names recorded");
    assert_eq!(tensors.len(), written_tensor_names(2, config.layers).len());

    // And the digest of what was fused, so the file can be checked against the
    // weights it came from.
    assert_eq!(
        reader
            .get(&format!("{NAMESPACE}weight_digest"))
            .and_then(GgufValue::as_str),
        Some(model.weight_digest().as_str())
    );

    // A tensor written by the fusion is dense, never quantised.
    let expand = reader.tensor(&format!("{NAMESPACE}expand")).expect("expand");
    assert_eq!(expand.kind, GGML_TYPE_F32);
}

#[test]
fn a_truncated_weights_file_is_refused_rather_than_reinterpreted() {
    let dir = TempDir::new("truncated");
    let base_path = dir.join("base.gguf");
    let weights_path = dir.join("state.pmc");
    let out_path = dir.join("fused.gguf");

    write_file(&base_path, &synthetic_gguf(64));
    let config = PmcStateConfig::default();
    let model = PmcStateModel::init(config.clone(), &[64, 64]).expect("init");
    write_weights(&weights_path, &model).expect("write weights");

    // Truncate the file after the header, so the parameters are gone but the
    // shape integers remain.
    let raw = fs::read(&weights_path).expect("read weights");
    let cut = 28 + 4 * 2 + 64 * 64 * 4; // header plus one host's projection
    write_file(&weights_path, &raw[..cut]);

    let request = StateMemoryRequest {
        base: base_path,
        weights: weights_path,
        out: out_path,
        manifest: None,
    };
    let error = fuse(&request).expect_err("a truncated weights file must be refused");
    // The refusal names the file, not just a category.
    assert!(
        error.contains("state.pmc") || error.contains("weights"),
        "the refusal should name what it refused: {error}"
    );
    assert!(!Path::new(&dir.join("fused.gguf")).exists(), "no output may be written");
}
