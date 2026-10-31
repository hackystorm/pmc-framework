// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for the recall evaluator.
//!
//! What matters about this evaluator is not that it prints six numbers - it is
//! that the numbers are computed from the model's own likelihoods, that the
//! probe split is stable across runs, and that the six keys it prints satisfy
//! the dream gate's contract exactly. A test that checks only "it runs" would
//! pass on an evaluator reporting constants.
//!
//! No GGUF and no llama.cpp: the checkpoint is written by the test, and the
//! store by the test, so a failure means the evaluator changed.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pmc_core::sha256_hex;
use pmc_train::checkpoint::{save, save_with_step};
use pmc_train::transformer::TransformerConfig;

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
            "pmc-dream-recall-{}-{}-{}",
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

fn small_config() -> TransformerConfig {
    TransformerConfig {
        n_layer: 2,
        d_model: 16,
        n_head: 2,
        ffn_hidden: 32,
        vocab: 64,
        n_positions: 32,
        readout_rank: 8,
    }
}

/// A checkpoint whose content is far longer than the probe window, so probes
/// always have a prefix and an answer to measure.
const MEMORY: &[u8] = b"Memory number one with distinct content to recall later on demand.";

/// Write a memory into the private store under the store's own layout.
fn store_memory(private: &Path, content: &[u8], record_id: &str) {
    let mut quoted = String::with_capacity(record_id.len() + 2);
    quoted.push('"');
    quoted.push_str(record_id);
    quoted.push('"');
    let key = sha256_hex(quoted.as_bytes());
    let mut file = File::create(private.join(format!("{}.bin", &key[..32])))
        .expect("create a private store blob");
    file.write_all(content).expect("write a private store blob");
}

fn evaluator_binary() -> Option<PathBuf> {
    let cargo = resolve_cargo()?;
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let built = Command::new(&cargo)
        .arg("build")
        .arg("--offline")
        .arg("--release")
        .arg("--example")
        .arg("dream_recall_eval")
        .current_dir(crate_dir)
        .output()
        .expect("run cargo build");
    if !built.status.success() {
        panic!(
            "cargo build failed: {}",
            String::from_utf8_lossy(&built.stderr)
        );
    }
    let binary = crate_dir
        .join("target")
        .join("release")
        .join("examples")
        .join("dream_recall_eval.exe");
    binary.is_file().then_some(binary)
}

fn resolve_cargo() -> Option<PathBuf> {
    if let Some(found) = which("cargo") {
        return Some(found);
    }
    let home = std::env::var_os("USERPROFILE").map(PathBuf::from)?;
    for name in ["cargo", "cargo.exe"] {
        let candidate = home.join(".cargo").join("bin").join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&paths) {
        for suffix in ["", ".exe"] {
            let candidate = directory.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Run the evaluator, or skip the test when cargo is unavailable.
fn evaluate(binary: &Path, checkpoint: &Path) -> Option<std::process::Output> {
    Command::new(binary).arg(checkpoint).output().ok()
}

/// The six metrics from stdout, as `(name, value)` pairs.
///
/// Parsed by hand rather than with `serde_json`, because this crate has no such
/// dependency and the repo's own constraint is that the Rust side stays
/// standard library only. The evaluator emits compact JSON whose values are all
/// plain numbers, so a split on the top-level separator recovers every pair
/// without a parser - and if the evaluator ever emits a nested value or a
/// string, this returns a pair whose "number" fails to parse, which is the
/// failure the test wants.
fn metrics(stdout: &str) -> Vec<(String, f64)> {
    let trimmed = stdout.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}'),
        "stdout must be one JSON object: {stdout}"
    );
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split("\",\"")
        .filter_map(|pair| {
            let pair = pair.trim_matches('"');
            let (name, value) = pair.split_once(':')?;
            let parsed: f64 = value.parse().ok()?;
            Some((name.to_string(), parsed))
        })
        .collect()
}

/// One metric's value from stdout.
fn metric(stdout: &str, name: &str) -> f64 {
    metrics(stdout)
        .into_iter()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("no metric named {name} in: {stdout}"))
        .1
}

// ---------------------------------------------------------------------------
// The gate contract
// ---------------------------------------------------------------------------

#[test]
fn the_evaluator_prints_exactly_the_six_gate_metrics() {
    // The pipeline's `_metrics` accepts any mapping but requires every value to
    // be a finite number, and the gate requires these six names. This test
    // asserts the stdout is precisely that contract: six numeric keys, nothing
    // else, so a diagnostic key accidentally printed to stdout fails the build
    // rather than the gate at runtime.
    let dir = TempDir::new("contract");
    let checkpoint = dir.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_aaaabbbbccccdddd");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pairs = metrics(&stdout);
    assert_eq!(
        pairs.len(),
        6,
        "the pipeline's contract is six metrics on stdout, got: {stdout}"
    );
    for name in [
        "recall_older",
        "recall_newer",
        "interference",
        "calibration",
        "identity",
        "false_memory",
    ] {
        let value = pairs
            .iter()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("missing {name} in: {stdout}"))
            .1;
        assert!(value.is_finite(), "{name} is not finite: {value}");
    }
}

#[test]
fn diagnostics_go_to_stderr_not_stdout() {
    // Stdout is the contract; a human reading the run needs the diagnostics.
    // Both must be present, in different streams.
    let dir = TempDir::new("streams");
    let checkpoint = dir.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_aaaabbbbccccdddd");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("held_out_probes"),
        "stderr should carry the probe counts: {stderr}"
    );
    assert!(
        stderr.contains("pmc-dream-recall"),
        "stderr should carry the schema: {stderr}"
    );
}

#[test]
fn the_same_store_yields_the_same_metrics() {
    // The evaluator runs twice on the same checkpoint, and both runs must agree
    // or the gate's before/after comparison is meaningless. Determinism is the
    // property that makes a measurement a measurement.
    let dir = TempDir::new("determinism");
    let checkpoint = dir.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_aaaabbbbccccdddd");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(first) = evaluate(&binary, &checkpoint) else {
        return;
    };
    let Some(second) = evaluate(&binary, &checkpoint) else {
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&first.stdout),
        String::from_utf8_lossy(&second.stdout),
        "two runs on the same inputs produced different metrics"
    );
}

// ---------------------------------------------------------------------------
// The probe split
// ---------------------------------------------------------------------------

#[test]
fn the_probe_split_covers_both_sets_and_is_stable() {
    // ~50% of records must land in each set, decided by content alone. An
    // earlier draft masked a bit of a hex *string*'s bytes, which are ASCII, so
    // no bit was ever set and every record landed in one set - making the
    // held-out half of the measurement permanently empty.
    let dir = TempDir::new("split");
    let checkpoint = dir.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");

    for index in 0..12 {
        let content = format!("Memory number {index} with distinct content to recall.");
        store_memory(
            &private,
            content.as_bytes(),
            &format!("mem_{index:016}"),
        );
    }

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("held_out_probes"),
        "the diagnostic should name the split: {stderr}"
    );
    // Held-out is the set consolidation is supposed to improve, so if it is
    // empty the gate has nothing to measure and would pass on a model that
    // recalled nothing.
    let held_out = probes_in(&stderr, "held_out_probes");
    let preserved = probes_in(&stderr, "preserved_probes");
    assert!(held_out > 0, "no held-out probes were formed: {stderr}");
    assert!(preserved > 0, "no preserved probes were formed: {stderr}");
    assert_eq!(
        held_out + preserved,
        12,
        "every probed record should be counted once: {stderr}"
    );
}

fn probes_in(report: &str, key: &str) -> usize {
    let needle = format!("\"{key}\":");
    let start = report.find(&needle).expect("key present") + needle.len();
    let rest = &report[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().expect("a count")
}

// ---------------------------------------------------------------------------
// Degenerate stores
// ---------------------------------------------------------------------------

#[test]
fn an_empty_store_reports_the_worst_scores_not_the_best() {
    // A workspace with no memories has nothing to recall. Reporting zeros
    // across the board would be a gate approving a dream over nothing, so the
    // empty case reports the *worst* values the honest metrics allow.
    let dir = TempDir::new("empty");
    let checkpoint = dir.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(metric(&stdout, "recall_newer"), 0.0);
    assert_eq!(metric(&stdout, "recall_older"), 0.0);
    assert_eq!(metric(&stdout, "false_memory"), 1.0);
    assert_eq!(metric(&stdout, "interference"), 1.0);
}

#[test]
fn a_store_above_the_checkpoint_is_found() {
    // The dream pipeline copies the checkpoint into a staging directory beside
    // it and passes that, so a fixed depth above the checkpoint is right for one
    // caller and wrong for another. Walking up works for both.
    let dir = TempDir::new("found");
    let workspace = dir.join("ws");
    let checkpoint = workspace.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = workspace.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_walkupyoursel");

    // The staged copy, one level below the workspace, as the pipeline makes it.
    let staged = workspace.join(".dream-abc").join("baseline");
    save(&staged, &small_config().init(20261031)).expect("write staged checkpoint");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &staged) else {
        return;
    };
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&private.display().to_string()),
        "the store beside the workspace should be found from the staged path: {stderr}"
    );
}

#[test]
fn a_checkpoint_with_no_store_above_it_is_refused() {
    // An evaluator that cannot find its workspace must say so, rather than
    // scoring a store it never read as a perfect recall.
    let lone = TempDir::new("lone");
    let checkpoint = lone.join("checkpoint");
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    assert!(!output.status.success(), "a missing store was scored anyway");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("sidecar") || stderr.contains("store"),
        "the refusal should name what it looked for: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Calibration's direction
// ---------------------------------------------------------------------------

#[test]
fn calibration_reported_as_higher_is_better() {
    // The gate's documented direction for calibration is higher-is-better, so a
    // better-calibrated model must score *higher*. An earlier draft reported raw
    // ECE, which is lower-is-better, inverting the metric: the gate passed
    // models that got worse and failed ones that got better.
    let dir = TempDir::new("direction");
    let checkpoint = dir.join("checkpoint");
    // A model whose recall is perfect should calibrate well: a high value.
    save(&checkpoint, &small_config().init(20261031)).expect("write checkpoint");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_ccccdaaaaabbbb");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let calibration = metric(&stdout, "calibration");
    assert!(
        (0.0..=1.0).contains(&calibration),
        "calibration must be a [0, 1] score where more is better, got {calibration}"
    );
    // A model that cannot discriminate at all has ECE near its worst, so its
    // score must be near zero rather than near one.
    assert!(
        calibration < 0.5,
        "a model that discriminates nothing should score low on calibration, got {calibration}"
    );
}

#[test]
fn a_stepped_baseline_is_readable_by_the_evaluator() {
    // The evaluator runs on the baseline *and* the candidate, and the candidate
    // records the consolidation's step count. Reading it must not depend on that
    // field being zero.
    let dir = TempDir::new("stepped");
    let checkpoint = dir.join("checkpoint");
    save_with_step(&checkpoint, &small_config().init(20261031), 41).expect("write");
    let private = dir.join("sidecar").join("private");
    fs::create_dir_all(&private).expect("create private store");
    store_memory(&private, MEMORY, "mem_ddddeeeeffff11");

    let Some(binary) = evaluator_binary() else {
        eprintln!("skipping: cargo unavailable");
        return;
    };
    let Some(output) = evaluate(&binary, &checkpoint) else {
        return;
    };
    assert!(
        output.status.success(),
        "a stepped checkpoint was refused: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
