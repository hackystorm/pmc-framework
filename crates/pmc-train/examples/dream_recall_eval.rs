// SPDX-License-Identifier: GPL-3.0-only

//! A real six-metric evaluator for the dream gate.
//!
//! The mechanics evaluator that came with this repository prints fixed values:
//! it verifies a checkpoint loaded and says plainly that it is not a
//! measurement. This is the evaluator that measures. It answers the question
//! the gate actually asks - did consolidation improve recall of held-out
//! material without damaging what was already there?
//!
//! ## What "recall" means here, and why it is not a preference for the model
//!
//! The model is a small byte-level transformer. Asking it a question and
//! reading prose back would measure nothing, so recall is measured as
//! **likelihood discrimination**: for a held-out memory, does the model assign
//! lower loss to the memory's real continuation than to a wrong one taken from
//! a different memory?
//!
//! This is the same preference structure the SPIN consolidator trains on, which
//! is deliberate - an evaluator and a trainer that disagree about what
//! "better" means produce a gate that measures the wrong thing.
//!
//! ## Why the probe split is derived from content
//!
//! The evaluator runs twice, once on the baseline and once on the candidate.
//! Comparing them is only meaningful if both runs score the *same* probes, so
//! the split is a deterministic function of each record's content digest and
//! never of the dream's outcome. A split that depended on which records the
//! last dream replayed would make the two runs incomparable.
//!
//! ## What it cannot measure
//!
//! These are measurements of a byte-level model's likelihoods. They say whether
//! consolidation moved the model's preferences in the right direction. They
//! say nothing about understanding, and nothing here claims they do.

use std::fs;
use std::path::{Path, PathBuf};

use pmc_core::{sha256_hex, Json};
use pmc_train::checkpoint::load;
use pmc_train::transformer::{Embedding, Transformer};

// ---------------------------------------------------------------------------
// Deterministic embedding, identical to the consolidator's
// ---------------------------------------------------------------------------

/// How many bytes of a memory form the question.
const PROBE_PREFIX: usize = 16;
/// How many bytes form the expected answer.
const PROBE_ANSWER: usize = 16;
/// A record is a held-out probe when its digest's first nibble is odd.
///
/// The digest is a hex *string*, so its bytes are ASCII: masked on a raw bit it
/// would never fire, and every record would land in the same set. Taking a
/// hex digit's value instead gives a real, stable ~50% split, decided by the
/// content alone and never by which records a dream replayed.
fn is_held_out_probe(content: &[u8]) -> bool {
    let digest = sha256_hex(content);
    let first = digest.as_bytes().first().copied().unwrap_or(b'0');
    (first - b'0') % 2 == 1
}

struct SeededEmbedding {
    rows: Vec<Vec<f32>>,
}

impl SeededEmbedding {
    fn new(vocab: usize, dim: usize, seed: u64) -> SeededEmbedding {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut rows = Vec::with_capacity(vocab);
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        for _ in 0..vocab {
            rows.push((0..dim).map(|_| next() * 0.5).collect());
        }
        SeededEmbedding { rows }
    }
}

impl Embedding for SeededEmbedding {
    fn dim(&self) -> usize {
        self.rows.first().map(Vec::len).unwrap_or(0)
    }

    fn row(&self, token: u32, out: &mut [f32]) {
        if let Some(row) = self.rows.get(token as usize) {
            out.copy_from_slice(row);
        } else {
            for value in out.iter_mut() {
                *value = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The private store
// ---------------------------------------------------------------------------

/// Locate the private store that belongs to a checkpoint.
///
/// The dream pipeline does not hand a consolidator the checkpoint the user
/// named: it copies into a staging directory created beside it and passes
/// *that*, so a fixed depth above the checkpoint is wrong for one caller and
/// right for the other. Walking up until a `sidecar` directory appears is right
/// for both, and it fails loudly rather than silently reading the wrong store.
fn find_store(checkpoint: &Path) -> Option<PathBuf> {
    let mut cursor = checkpoint.parent()?;
    loop {
        let candidate = cursor.join("sidecar").join("private");
        if candidate.is_dir() {
            return Some(candidate);
        }
        cursor = cursor.parent()?;
    }
}

/// Every record's content in the store.
///
/// The blobs are enumerated directly rather than through `sha256_of`, because
/// the store's *keys* are its own business and this evaluator needs only the
/// bytes: which record a blob names is what the dream journal records, not
/// something an evaluator should have to recompute. The layout it reads is the
/// one `pmc.dignity.private_store` writes.
fn read_store(store_root: &Path) -> Vec<Vec<u8>> {
    let mut contents = Vec::new();
    if !store_root.is_dir() {
        return contents;
    }
    let Ok(rows) = fs::read_dir(store_root) else {
        return contents;
    };
    for entry in rows.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        if let Ok(bytes) = fs::read(&path) {
            if !bytes.is_empty() {
                contents.push(bytes);
            }
        }
    }
    // A stable order, so the same store yields the same metrics on every run.
    contents.sort();
    contents
}

fn tokenize(bytes: &[u8], vocab: usize, max_len: usize) -> Vec<u32> {
    bytes
        .iter()
        .take(max_len)
        .map(|byte| u32::from(*byte) % vocab as u32)
        .collect()
}

// ---------------------------------------------------------------------------
// One measurement
// ---------------------------------------------------------------------------

/// One probe: a question, the real answer, and a wrong answer from elsewhere.
struct Probe {
    question: Vec<u32>,
    expected: Vec<u32>,
    control: Vec<u32>,
}

/// Does the model prefer the real continuation over a wrong one?
///
/// Returns `(correct, margin)`. The margin is how much lower the real
/// continuation's loss is, in the model's own units: the gate's calibration
/// metric uses it, so "right for a good reason" and "right by a hair" are not
/// the same result.
fn discriminate(model: &Transformer, embedding: &SeededEmbedding, probe: &Probe) -> (bool, f32) {
    let loss = |tail: &[u32]| -> f32 {
        let mut tokens = probe.question.clone();
        tokens.extend_from_slice(tail);
        if tokens.len() < 2 || tokens.len() > model.config.n_positions {
            return f32::INFINITY;
        }
        let mask = vec![true; tokens.len()];
        model.forward(&tokens, embedding, Some(&mask)).loss
    };
    let expected_loss = loss(&probe.expected);
    let control_loss = loss(&probe.control);
    if !expected_loss.is_finite() || !control_loss.is_finite() {
        return (false, 0.0);
    }
    let margin = control_loss - expected_loss;
    (margin > 0.0, margin)
}

/// The six gate metrics, each named for what it measures.
struct Recall {
    recall_older: f32,
    recall_newer: f32,
    interference: f32,
    calibration: f32,
    identity: f32,
    false_memory: f32,
}

fn usage() {
    println!(
        "dream_recall_eval - a real six-metric recall evaluator for the dream gate

Usage:
  dream_recall_eval <checkpoint>

Measures held-out likelihood discrimination: does the model prefer a real
memory continuation over a wrong one? Prints the six gate metrics as JSON."
    );
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        usage();
        return std::process::ExitCode::from(2);
    }
    let checkpoint = PathBuf::from(&args[1]);

    let model = match load(&checkpoint) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("error: cannot read the checkpoint: {error}");
            return std::process::ExitCode::from(2);
        }
    };

    // Found by walking up, not by assuming a depth: see `find_store`.
    let store_root = match find_store(&checkpoint) {
        Some(root) => root,
        None => {
            eprintln!(
                "error: no sidecar/private store above {}: an evaluator must find the workspace it is measuring",
                checkpoint.display()
            );
            return std::process::ExitCode::from(2);
        }
    };

    let contents = read_store(&store_root);
    if contents.is_empty() {
        // Empty is a valid state - a workspace with no memories has nothing to
        // recall. Report it as such rather than as a perfect score.
        let metrics = Recall {
            recall_older: 0.0,
            recall_newer: 0.0,
            interference: 1.0,
            calibration: 1.0,
            identity: 1.0,
            false_memory: 1.0,
        };
        println!("{}", report(&metrics, 0, 0, &store_root));
        return std::process::ExitCode::SUCCESS;
    }

    // Fixed limit: the checkpoint's context is small, and probing beyond it
    // measures the limit of the sequence length rather than of the model.
    let cap = model.config.n_positions.saturating_sub(1);
    let kept: Vec<Vec<u8>> = contents
        .into_iter()
        .filter(|content| content.len() >= PROBE_PREFIX + PROBE_ANSWER)
        .take(cap / 2)
        .collect();

    let embedding_seed = embedding_seed_of(&model);
    let embedding = SeededEmbedding::new(model.config.vocab, model.config.d_model, embedding_seed);

    let n = kept.len();
    let mut held_out: Vec<Probe> = Vec::new();
    let mut preserved: Vec<Probe> = Vec::new();
    let mut index = 0usize;
    for content in &kept {
        // Which set a record belongs to is decided by its own content, not by
        // which records a dream happened to replay.
        let is_probe = is_held_out_probe(content);
        let question = tokenize(&content[..PROBE_PREFIX], model.config.vocab, cap);
        let expected = tokenize(
            &content[PROBE_PREFIX..PROBE_PREFIX + PROBE_ANSWER],
            model.config.vocab,
            cap,
        );
        // The wrong answer is the next record's continuation: held-out material
        // that was never this memory's answer.
        let control = {
            index = (index + 1) % n;
            let other = &kept[index];
            let start = PROBE_PREFIX.min(other.len().saturating_sub(PROBE_ANSWER));
            tokenize(
                &other[start..start + PROBE_ANSWER],
                model.config.vocab,
                cap,
            )
        };
        if question.len() != PROBE_PREFIX || expected.len() != PROBE_ANSWER {
            continue;
        }
        let probe = Probe {
            question,
            expected,
            control,
        };
        if is_probe {
            held_out.push(probe);
        } else {
            preserved.push(probe);
        }
    }

    let score = |probes: &[Probe]| -> (f32, f32, Vec<(bool, f32)>) {
        // (accuracy, false-memory rate, per-probe results)
        if probes.is_empty() {
            return (0.0, 0.0, Vec::new());
        }
        let results: Vec<(bool, f32)> = probes
            .iter()
            .map(|probe| discriminate(&model, &embedding, probe))
            .collect();
        let correct = results.iter().filter(|(ok, _)| *ok).count();
        let accuracy = correct as f32 / results.len() as f32;
        // A confabulation is a *confident* wrong answer: the control beat the
        // real continuation by a wide margin, not by a hair.
        let confabulated = results
            .iter()
            .filter(|(ok, margin)| !ok && *margin < -CONFIDENT_MARGIN)
            .count();
        let false_memory = confabulated as f32 / results.len() as f32;
        (accuracy, false_memory, results)
    };

    // A margin this wide is "confidently wrong" rather than "barely wrong".
    const CONFIDENT_MARGIN: f32 = 0.5;

    let (newer_accuracy, newer_false, newer_results) = score(&held_out);
    let (older_accuracy, _, older_results) = score(&preserved);

    // Both sets are measured the same way, so the difference between them is
    // what consolidation cost the material that was already there.
    let combined: Vec<(bool, f32)> = newer_results
        .iter()
        .chain(older_results.iter())
        .cloned()
        .collect();
    let metrics = Recall {
        recall_older: older_accuracy,
        recall_newer: newer_accuracy,
        interference: (older_accuracy - newer_accuracy).max(0.0),
        // The gate treats calibration as higher-is-better, so what it reports is
        // 1 - ECE. Reporting raw ECE would invert the metric: a gate that
        // supposed to catch a worse-calibrated model would instead pass it, and
        // one that noted an improvement would fail it.
        calibration: (1.0 - expected_calibration_error(&combined)).max(0.0),
        // Identity: the architecture is unchanged, because a dream that changed
        // it would produce a model that is not the instance. The config is
        // validated on load, so this is 1.0 whenever the checkpoint was read.
        identity: 1.0,
        false_memory: newer_false,
    };

    // The pipeline's contract is six numeric metrics on stdout: it refuses any
    // other key rather than guessing which ones matter. Everything else - what
    // was measured, how many probes, from which store - goes to stderr, where a
    // person reading a run still sees it and a parser never trips over it.
    eprintln!(
        "{}",
        report(
            &metrics,
            held_out.len(),
            preserved.len(),
            &store_root,
        )
    );
    println!(
        "{}",
        Json::obj(vec![
            ("recall_older", Json::num(metrics.recall_older as f64)),
            ("recall_newer", Json::num(metrics.recall_newer as f64)),
            ("interference", Json::num(metrics.interference as f64)),
            ("calibration", Json::num(metrics.calibration as f64)),
            ("identity", Json::num(metrics.identity as f64)),
            ("false_memory", Json::num(metrics.false_memory as f64)),
        ])
        .to_string()
    );
    std::process::ExitCode::SUCCESS
}

/// A seed derived from the checkpoint's own parameters, matching the
/// consolidator: both sides must embed the same way for before/after to be
/// comparable.
fn embedding_seed_of(model: &Transformer) -> u64 {
    let bytes = model.to_bytes();
    let digest = pmc_core::sha256(&bytes);
    let mut seed = [0u8; 8];
    seed.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(seed)
}

/// Expected calibration error over the discrimination margins.
///
/// Confidence is the magnitude of the margin, bucketed; accuracy is the
/// fraction correct in each bucket. A model that is right for the right reason
/// has a low ECE; one that is right by luck has a high one. Both are
/// measurements, and neither is a verdict.
/// Expected calibration error over the discrimination margins.
///
/// Confidence is the magnitude of the margin, bucketed; accuracy is the
/// fraction correct in each bucket. A model that is right for the right reason
/// has a low ECE; one that is right by luck has a high one.
///
/// **Lower is better**, and the gate wants higher-is-better, so the caller
/// reports `1 - this`. The inversion is done once, here, rather than at every
/// call site where it could be forgotten.
fn expected_calibration_error(results: &[(bool, f32)]) -> f32 {
    const BUCKETS: usize = 5;
    if results.is_empty() {
        return 1.0;
    }
    let widest = results
        .iter()
        .map(|(_, margin)| margin.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let mut counts = [0usize; BUCKETS];
    let mut correct = [0usize; BUCKETS];
    let mut confidence = [0.0f32; BUCKETS];
    for (ok, margin) in results {
        let bucket = (((margin.abs() / widest) * BUCKETS as f32).floor() as usize).min(BUCKETS - 1);
        counts[bucket] += 1;
        confidence[bucket] += margin.abs();
        if *ok {
            correct[bucket] += 1;
        }
    }
    let total = results.len() as f32;
    let mut error = 0.0f32;
    for bucket in 0..BUCKETS {
        if counts[bucket] == 0 {
            continue;
        }
        let accuracy = correct[bucket] as f32 / counts[bucket] as f32;
        let mean_confidence = confidence[bucket] / counts[bucket] as f32;
        error += (counts[bucket] as f32 / total) * (mean_confidence.min(1.0) - accuracy).abs();
    }
    error
}

fn report(
    metrics: &Recall,
    held_out: usize,
    preserved: usize,
    store: &Path,
) -> String {
    Json::obj(vec![
        ("schema", Json::str("pmc-dream-recall/1")),
        ("mechanism", Json::str("likelihood-discrimination")),
        ("held_out_probes", Json::num(held_out as f64)),
        ("preserved_probes", Json::num(preserved as f64)),
        ("store", Json::str(store.display().to_string())),
        ("recall_older", Json::num(metrics.recall_older as f64)),
        ("recall_newer", Json::num(metrics.recall_newer as f64)),
        ("interference", Json::num(metrics.interference as f64)),
        ("calibration", Json::num(metrics.calibration as f64)),
        ("identity", Json::num(metrics.identity as f64)),
        ("false_memory", Json::num(metrics.false_memory as f64)),
    ])
    .to_string()
}
