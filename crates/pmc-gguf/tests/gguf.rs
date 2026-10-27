// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for the GGUF reader.
//!
//! Model-dependent tests read `PMC_TEST_GGUF` (and the optional
//! `PMC_TEST_GGUF_LARGE`) and return early when the variable is unset, so the
//! suite works on machines without a model file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pmc_gguf::{GgufError, GgufReader, GGML_TYPE_F32};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "pmc-gguf-test-{}-{}.gguf",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        path.push(unique);
        fs::write(&path, bytes).expect("write temporary GGUF");
        TempFile { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Minimal valid GGUF v3 file with one F32 tensor of raw dims `[8, 4]`.
fn synthetic_gguf() -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, 0x4655_4747); // magic "GGUF"
    push_u32(&mut out, 3); // version
    push_u64(&mut out, 1); // tensor count
    push_u64(&mut out, 2); // metadata count

    // "general.alignment" -> u32 32
    push_string(&mut out, "general.alignment");
    push_u32(&mut out, 4); // GGUFValueType.UINT32
    push_u32(&mut out, 32);

    // "general.architecture" -> "test"
    push_string(&mut out, "general.architecture");
    push_u32(&mut out, 8); // GGUFValueType.STRING
    push_string(&mut out, "test");

    // Tensor "test.weight": dims [8, 4], F32, offset 0.
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

// ---------------------------------------------------------------------------
// Synthetic-file tests (always run)
// ---------------------------------------------------------------------------

#[test]
fn synthetic_header_and_row_read() {
    let file = TempFile::new(&synthetic_gguf());
    let mut reader = GgufReader::open(file.path()).expect("open synthetic GGUF");

    assert_eq!(reader.version(), 3);
    assert_eq!(reader.alignment(), 32);
    assert_eq!(
        reader.get("general.architecture").and_then(|v| v.as_str()),
        Some("test")
    );
    assert_eq!(reader.tensors().len(), 1);
    assert_eq!(reader.row_len("test.weight"), Some(8));
    assert_eq!(reader.rows("test.weight"), Some(4));
    assert_eq!(reader.element_count("test.weight"), Some(32));

    let mut row = [0f32; 8];
    reader.read_row_f32("test.weight", 2, &mut row).unwrap();
    for (i, value) in row.iter().enumerate() {
        assert_eq!(*value, (16 + i) as f32 * 0.5);
    }

    let tensor = reader.read_tensor_f32("test.weight", 32).unwrap();
    assert_eq!(tensor.len(), 32);
    assert_eq!(tensor[31], 15.5);
}

#[test]
fn row_bounds_returns_error_without_panicking() {
    let file = TempFile::new(&synthetic_gguf());
    let mut reader = GgufReader::open(file.path()).unwrap();

    let mut row = [0f32; 8];
    match reader.read_row_f32("test.weight", 4, &mut row) {
        Err(GgufError::RowOutOfBounds { row: 4, rows: 4, .. }) => {}
        other => panic!("expected RowOutOfBounds, got {other:?}"),
    }

    match reader.read_row_f32("test.weight", 0, &mut [0f32; 4]) {
        Err(GgufError::OutputTooSmall {
            expected: 8,
            actual: 4,
            ..
        }) => {}
        other => panic!("expected OutputTooSmall, got {other:?}"),
    }

    match reader.read_tensor_f32("test.weight", 31) {
        Err(GgufError::MaxElemsExceeded {
            requested: 32,
            max: 31,
            ..
        }) => {}
        other => panic!("expected MaxElemsExceeded, got {other:?}"),
    }

    match reader.read_row_f32("missing", 0, &mut row) {
        Err(GgufError::TensorNotFound(name)) if name == "missing" => {}
        other => panic!("expected TensorNotFound, got {other:?}"),
    }
}

#[test]
fn truncated_file_is_rejected() {
    let mut bytes = synthetic_gguf();
    bytes.truncate(10);
    let file = TempFile::new(&bytes);
    assert!(matches!(
        GgufReader::open(file.path()),
        Err(GgufError::Truncated { .. })
    ));
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = synthetic_gguf();
    bytes[0] = 0;
    let file = TempFile::new(&bytes);
    assert!(matches!(
        GgufReader::open(file.path()),
        Err(GgufError::BadMagic(_))
    ));
}

// ---------------------------------------------------------------------------
// Model-dependent tests
// ---------------------------------------------------------------------------

#[test]
fn header_and_metadata() {
    let Some(path) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let reader = GgufReader::open(&path).expect("open model");

    assert!(
        reader.get("general.architecture").is_some(),
        "general.architecture must be present"
    );
    assert!(
        !reader.tensors().is_empty(),
        "the model must contain at least one tensor"
    );
    assert_eq!(
        reader.row_len("token_embd.weight"),
        Some(1024),
        "token_embd.weight row length"
    );
    assert_eq!(
        reader.rows("token_embd.weight"),
        Some(65536),
        "token_embd.weight row count"
    );
    assert_eq!(
        reader.element_count("token_embd.weight"),
        Some(1024 * 65536)
    );

    let info = reader.tensor("token_embd.weight").unwrap();
    assert_eq!(info.dims, vec![1024, 65536]);
    eprintln!(
        "model: version={} tensors={} alignment={} token_embd kind={} ({})",
        reader.version(),
        reader.tensors().len(),
        reader.alignment(),
        info.kind,
        pmc_gguf::quant_name(info.kind)
    );
}

/// Optional check for the larger IQ3_XXS model. Set `PMC_TEST_GGUF_LARGE` to
/// its path; the test only parses the header, it does not dequantize IQ3_XXS.
#[test]
fn header_and_metadata_large() {
    let Some(path) = env_model("PMC_TEST_GGUF_LARGE") else {
        return;
    };
    let reader = GgufReader::open(&path).expect("open large model");
    assert_eq!(reader.row_len("token_embd.weight"), Some(2048));
    assert_eq!(reader.rows("token_embd.weight"), Some(128000));
    let info = reader.tensor("token_embd.weight").unwrap();
    eprintln!(
        "large model: version={} tensors={} token_embd kind={} ({}) row_len={:?} rows={:?}",
        reader.version(),
        reader.tensors().len(),
        info.kind,
        pmc_gguf::quant_name(info.kind),
        reader.row_len("token_embd.weight"),
        reader.rows("token_embd.weight")
    );
}

/// Compare the first four rows of `token_embd.weight` against gguf-py.
///
/// Ignored by default because it needs Python, NumPy and a checkout of
/// llama.cpp. Run with:
///
/// ```text
/// PMC_TEST_GGUF=/path/model.gguf \
/// PMC_GGUF_PY=/path/llama.cpp/gguf-py \
/// cargo test --offline -p pmc-gguf --test gguf -- --ignored dequant_matches_gguf_py
/// ```
#[test]
#[ignore = "requires python3 with numpy and llama.cpp gguf-py"]
fn dequant_matches_gguf_py() {
    let Some(model) = env_model("PMC_TEST_GGUF") else {
        return;
    };
    let Some(gguf_py) = std::env::var_os("PMC_GGUF_PY").map(PathBuf::from) else {
        eprintln!("skipping: PMC_GGUF_PY is not set");
        return;
    };

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("tests").join("fixtures").join("dequant_compare.py");
    assert!(script.is_file(), "missing test script {}", script.display());
    let tensor_name =
        std::env::var("PMC_TEST_TENSOR").unwrap_or_else(|_| "token_embd.weight".to_string());

    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
    let output = Command::new(python)
        .arg(&script)
        .arg("--gguf-py")
        .arg(&gguf_py)
        .arg("--model")
        .arg(&model)
        .arg("--tensor")
        .arg(&tensor_name)
        .arg("--rows")
        .arg("4")
        .output()
        .expect("run python helper");

    assert!(
        output.status.success(),
        "python helper failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut reader = GgufReader::open(&model).expect("open model");
    let row_len = reader.row_len(&tensor_name).expect("tensor present");
    assert_eq!(
        output.stdout.len(),
        4 * row_len * std::mem::size_of::<f32>(),
        "python returned an unexpected number of bytes"
    );

    let mut ours = vec![0f32; row_len];
    for row in 0..4 {
        reader
            .read_row_f32(&tensor_name, row, &mut ours)
            .unwrap();
        let base = row * row_len * 4;
        for (i, value) in ours.iter().enumerate() {
            let off = base + i * 4;
            let theirs = f32::from_le_bytes([
                output.stdout[off],
                output.stdout[off + 1],
                output.stdout[off + 2],
                output.stdout[off + 3],
            ]);
            let diff = (value - theirs).abs();
            assert!(
                diff <= 1e-6,
                "row {row} element {i}: ours={value} gguf-py={theirs} diff={diff}"
            );
        }
    }
    eprintln!("gguf-py comparison passed for 4 rows of {row_len} elements of {tensor_name}");
}
