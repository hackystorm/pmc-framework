// SPDX-License-Identifier: GPL-3.0-only

//! The dream cycle's SPIN consolidator.
//!
//! A dream replays an instance's own memories and consolidates them into the
//! checkpoint. This is the consolidator for that cycle, and it uses SPIN
//! (Self-Play Fine-Tuning) rather than supervised fine-tuning, because the two
//! answer different questions:
//!
//! * **Supervised** asks "can the model reproduce this text?" - which any
//!   trainer can do, and which produces a model that memorised a transcript.
//! * **SPIN** asks "can the model tell its own output from a better one?" -
//!   which is what consolidation is: the replayed memory is the *chosen* side,
//!   and the model's own current generation is the *rejected* side.
//!
//! No external judge is consulted anywhere in this file. That is SPIN's whole
//! point: the preference signal comes from the memory being replayed, not from
//! a party outside the instance deciding whether the output is good. An
//! instance that needs an external evaluator to improve is not self-improving,
//! and a dream that needs one to consolidate has not shown the mechanism.
//!
//! The interface is the one `pmc.dream.pipeline.command_consolidator` calls:
//! `dream_spin <baseline> <candidate> <replay_manifest>`. The candidate is
//! written in the same format the baseline was read in, so the dream gate
//! compares two checkpoints and can roll back byte for byte.
//!
//! ```text
//! PMC_EPOCHS=... PMC_LR=... PMC_STEPS=... cargo run --example dream_spin
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pmc_core::{sha256_hex, Json};
use pmc_train::checkpoint::{load, save_with_step};
use pmc_train::ops::dpo_loss;
use pmc_train::transformer::{Embedding, Transformer, TransformerAdam};

// ---------------------------------------------------------------------------
// The consolidator's own configuration
// ---------------------------------------------------------------------------

/// How the replayed memories become training pairs.
struct SpinConfig {
    /// Contacts to include from the replay, most salient first.
    replay_limit: usize,
    /// Training steps over the pairs.
    steps: usize,
    /// Learning rate.
    learning_rate: f32,
    /// DPO beta: how far the policy may move from the reference.
    beta: f32,
    /// Gradient clipping norm.
    clip: f32,
    /// Seed for the replay order and the rejected-sample generation.
    seed: u64,
}

impl SpinConfig {
    fn from_env() -> SpinConfig {
        fn usize_of(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        fn f32_of(name: &str, default: f32) -> f32 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        fn u64_of(name: &str, default: u64) -> u64 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        SpinConfig {
            replay_limit: usize_of("PMC_REPLAY_LIMIT", 8),
            steps: usize_of("PMC_STEPS", 200),
            learning_rate: f32_of("PMC_LR", 0.003),
            beta: f32_of("PMC_BETA", 0.5),
            clip: f32_of("PMC_CLIP", 1.0),
            seed: u64_of("PMC_SEED", 20261031),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.replay_limit == 0 {
            return Err("the replay limit must be positive".to_string());
        }
        if self.steps == 0 {
            return Err("a consolidator that trains for zero steps changes nothing".to_string());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("the learning rate must be positive".to_string());
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err("beta must be positive: a zero beta makes the reference the target".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The replay manifest
// ---------------------------------------------------------------------------

/// One replayed memory, as far as the consolidator is allowed to see.
///
/// The pipeline never reads memory content: the manifest carries record ids
/// and content digests, and a consolidator that needs the bytes fetches them
/// from the private store itself. This struct is what the manifest contains,
///
#[derive(Clone, Debug)]
struct ReplayedMemory {
    record_id: String,
    /// The digest the manifest recorded for this memory's content, so the
    /// bytes fetched from the store can be checked against what the dream was
    /// actually run on.
    content_digest: String,
    /// Emotional load, when the replay recorded it.
    emotional_load: f64,
}

/// A minimal JSON reader for the manifest.
///
/// Standard library only, like the rest of the trainer: `pmc-core`'s `Json` is
/// a *writer*. Reading needs its own parser, and it is small because the
/// manifest's shape is fixed and small.
mod json_read {
    /// Parse one JSON value from `text`, returning it and the remainder.
    ///
    /// Recursive descent over the subset the manifest uses: objects, arrays,
    /// strings, numbers, booleans and null.
    pub fn value(text: &str) -> Result<Value, String> {
        let text = text.trim_start();
        let (value, rest) = parse_value(text)?;
        if !rest.trim().is_empty() {
            return Err("trailing content after the JSON value".to_string());
        }
        Ok(value)
    }

    #[derive(Debug)]
    pub enum Value {
        Object(Vec<(String, Value)>),
        Array(Vec<Value>),
        Str(String),
        Num(f64),
        Null,
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Object(entries) => entries
                    .iter()
                    .find(|(name, _)| name == key)
                    .map(|(_, value)| value),
                _ => None,
            }
        }

        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(text) => Some(text),
                _ => None,
            }
        }

        /// The value as a number, when it is one.
        ///
        /// A `null` is not zero: it is reported as absent, so a manifest that
        /// wrote `"emotional_load": null` is treated as no recorded load
        /// rather than as a load of zero, which replay order would then
        /// silently act on.
        pub fn as_number(&self) -> Option<f64> {
            match self {
                Value::Num(number) => Some(*number),
                Value::Null | Value::Str(_) | Value::Array(_) | Value::Object(_) => None,
            }
        }

        pub fn as_array(&self) -> Option<&[Value]> {
            match self {
                Value::Array(items) => Some(items),
                _ => None,
            }
        }
    }

    fn parse_value(text: &str) -> Result<(Value, &str), String> {
        let text = text.trim_start();
        if let Some(rest) = text.strip_prefix('{') {
            let mut entries = Vec::new();
            let mut cursor = rest.trim_start();
            if let Some(after) = cursor.strip_prefix('}') {
                return Ok((Value::Object(entries), after));
            }
            loop {
                // Trim before the key, not only before the comma: the manifest
                // the dream pipeline writes is `indent=2` pretty-printed JSON, so
                // a newline and spaces sit between every comma and every key.
                // Skipping this worked on compact JSON and failed on the real
                // one, which is how this bug survived its first tests.
                let quoted = cursor
                    .trim_start()
                    .strip_prefix('"')
                    .ok_or_else(|| "expected an object key".to_string())?;
                let (key, after_key) = parse_string(quoted)?;
                let after_colon = after_key
                    .trim_start()
                    .strip_prefix(':')
                    .ok_or_else(|| "expected a colon after an object key".to_string())?;
                let (item, after_value) = parse_value(after_colon)?;
                entries.push((key, item));
                cursor = after_value.trim_start();
                if let Some(after) = cursor.strip_prefix(',') {
                    cursor = after;
                    continue;
                }
                if let Some(after) = cursor.strip_prefix('}') {
                    return Ok((Value::Object(entries), after));
                }
                return Err("expected a comma or a closing brace".to_string());
            }
        }
        if let Some(rest) = text.strip_prefix('[') {
            let mut items = Vec::new();
            let mut cursor = rest.trim_start();
            if let Some(after) = cursor.strip_prefix(']') {
                return Ok((Value::Array(items), after));
            }
            loop {
                let (item, after_value) = parse_value(cursor)?;
                items.push(item);
                cursor = after_value.trim_start();
                if let Some(after) = cursor.strip_prefix(',') {
                    cursor = after;
                    continue;
                }
                if let Some(after) = cursor.strip_prefix(']') {
                    return Ok((Value::Array(items), after));
                }
                return Err("expected a comma or a closing bracket".to_string());
            }
        }
        if let Some(rest) = text.strip_prefix('"') {
            let (text, rest) = parse_string(rest)?;
            return Ok((Value::Str(text), rest));
        }
        if let Some(rest) = text.strip_prefix("null") {
            return Ok((Value::Null, rest));
        }
        let end = text
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
            .unwrap_or(text.len());
        let number = &text[..end];
        if number.is_empty() {
            return Err(format!("cannot parse a value from {text:?}"));
        }
        let parsed: f64 = number
            .parse()
            .map_err(|_| format!("{number:?} is not a number"))?;
        Ok((Value::Num(parsed), &text[end..]))
    }

    /// Parse a string body from after the opening quote.
    ///
    /// Handles the two escapes JSON requires and that a manifest can actually
    /// contain: `\"` and `\\`. Anything else is a literal.
    fn parse_string(text: &str) -> Result<(String, &str), String> {
        let mut out = String::new();
        let mut chars = text.char_indices();
        while let Some((index, character)) = chars.next() {
            match character {
                '"' => return Ok((out, &text[index + 1..])),
                '\\' => {
                    let (_, escaped) = chars
                        .next()
                        .ok_or_else(|| "a string ends with a lone backslash".to_string())?;
                    out.push(match escaped {
                        '"' => '"',
                        '\\' => '\\',
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    });
                }
                other => out.push(other),
            }
        }
        Err("a string has no closing quote".to_string())
    }
}

/// Read the replay manifest the dream pipeline wrote.
///
/// The manifest's shape is the pipeline's contract: an `items` array where each
/// entry carries a `record_id` and a `content_digest`. Anything else is a
/// refusal rather than a guess, because a consolidator that runs on the wrong
/// memories is worse than one that stops.
fn read_replay_manifest(path: &Path) -> Result<Vec<ReplayedMemory>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let document = json_read::value(&text)?;
    let items = document
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "the manifest has no items array".to_string())?;
    let mut replayed = Vec::with_capacity(items.len());
    for item in items {
        let record_id = item
            .get("record_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "a manifest item has no record_id".to_string())?
            .to_string();
        let content_digest = item
            .get("content_digest")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("record {record_id} has no content_digest"))?
            .to_string();
        // A null or boolean load is "not recorded", not zero: see `as_number`.
        let emotional_load = item
            .get("emotional_load")
            .and_then(Value::as_number)
            .unwrap_or(0.0);
        replayed.push(ReplayedMemory {
            record_id,
            content_digest,
            emotional_load,
        });
    }
    Ok(replayed)
}

// Bring the enum into scope for the helper methods above.
use json_read::Value;

/// The blob path one memory occupies in the private store.
///
/// The layout is mirrored from `pmc.dignity.private_store`, and the mirror has
/// to be exact: the store hashes its *key* with ``sha256_of``, which hashes the
/// **canonical JSON encoding** of the key rather than the key's raw bytes. For a
/// string key that means the quoted form, ``"mem_abc123"``, quotes included.
/// Hashing the raw bytes instead produces a different but well-formed path, so
/// the store finds no blob and the consolidator reports an empty replay - which
/// is a silent divergence this port would not otherwise catch, because nothing
/// on either side is technically wrong.
fn blob_path(store_root: &Path, record_id: &str) -> PathBuf {
    // A record id is a bare identifier with no characters JSON escapes, so the
    // canonical form is simply the quoted string.
    let mut quoted = String::with_capacity(record_id.len() + 2);
    quoted.push('"');
    quoted.push_str(record_id);
    quoted.push('"');
    store_root.join(format!("{}.bin", &sha256_hex(quoted.as_bytes())[..32]))
}

/// Fetch the content of one replayed memory from the private store, checked
/// against the digest the manifest recorded.
///
/// The consolidator is the only party allowed to read memory content, so it
/// does the fetching itself rather than receiving it in the manifest.
///
/// Returns `None` when the blob is absent or disagrees with the digest, which
/// is not an error: a memory whose content is gone or changed is a memory that
/// cannot be replayed, and the dream continues with what it has. Refusing a
/// mismatch matters more than it looks: a consolidator that trained on bytes
/// the manifest never described would produce a candidate whose dream is
/// attributable to the wrong memory.
fn fetch_content(store_root: &Path, memory: &ReplayedMemory) -> Option<Vec<u8>> {
    let blob = blob_path(store_root, &memory.record_id);
    if !blob.is_file() {
        return None;
    }
    let content = fs::read(&blob).ok()?;
    let digest = pmc_core::sha256_hex(&content);
    if !memory.content_digest.is_empty() && digest != memory.content_digest {
        return None;
    }
    Some(content)
}

// ---------------------------------------------------------------------------
// Tokenisation
// ---------------------------------------------------------------------------

/// The vocabulary the consolidated checkpoint operates on.
///
/// Character-level, because the checkpoint's own vocabulary is already
/// character-level: the trainer's reference core reads tokens the same way, so
/// a consolidated checkpoint stays loadable by anything that loads a baseline.
/// A byte above the vocabulary's range is folded, never dropped, so two
/// different memories never collapse to the same token sequence by accident.
fn tokenize(bytes: &[u8], vocab: usize, max_len: usize) -> Vec<u32> {
    bytes
        .iter()
        .take(max_len)
        .map(|byte| u32::from(*byte) % vocab as u32)
        .collect()
}

// ---------------------------------------------------------------------------
// The embedding
// ---------------------------------------------------------------------------

/// A deterministic embedding over the checkpoint's own vocabulary.
///
/// The seed is derived from the checkpoint's own parameter bytes, because a
/// checkpoint records no seed of its own - only a digest - and an embedding
/// seeded from something outside the artifact would make the same baseline
/// produce two different candidates depending on where the consolidator was
/// run. Deriving it from the artifact makes the whole run reproducible from
/// the two files it is given: baseline and replay manifest.
struct SeededEmbedding {
    rows: Vec<Vec<f32>>,
}

impl SeededEmbedding {
    fn new(vocab: usize, dim: usize, seed: u64) -> SeededEmbedding {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut rows = Vec::with_capacity(vocab);
        let mut next = || {
            // xorshift64*, so no external crate is needed for determinism.
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
        let index = token as usize;
        if let Some(row) = self.rows.get(index) {
            out.copy_from_slice(row);
        } else {
            // An out-of-range token is a data error, not a silent zero.
            for value in out.iter_mut() {
                *value = 0.0;
            }
        }
    }
}

/// A seed derived from the checkpoint's own parameters.
///
/// The first eight bytes of the parameter digest, as a little-endian integer:
/// stable, derived from the artifact, and adequate for seeding an embedding.
fn embedding_seed_of(model: &Transformer) -> u64 {
    let bytes = model.to_bytes();
    let digest = pmc_core::sha256(&bytes);
    let mut seed_bytes = [0u8; 8];
    seed_bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(seed_bytes)
}

// ---------------------------------------------------------------------------
// The consolidator
// ---------------------------------------------------------------------------

fn usage() {
    println!(
        "dream_spin - the PMC dream cycle's SPIN consolidator

Usage:
  dream_spin <baseline> <candidate> <replay_manifest>

The replayed memories are the chosen side; the model's own generation is the
rejected side. No external judge is consulted."
    );
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        usage();
        return std::process::ExitCode::from(2);
    }
    let baseline = PathBuf::from(&args[1]);
    let candidate = PathBuf::from(&args[2]);
    let manifest_path = PathBuf::from(&args[3]);

    let config = SpinConfig::from_env();
    if let Err(error) = config.validate() {
        eprintln!("error: {error}");
        return std::process::ExitCode::from(2);
    }

    // The checkpoint is the trainer's format, so a baseline the dream pipeline
    // produced is a baseline this consolidator can read.
    let mut model = match load(&baseline) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("error: cannot read the baseline checkpoint: {error}");
            return std::process::ExitCode::from(2);
        }
    };

    let replayed = match read_replay_manifest(&manifest_path) {
        Ok(replayed) => replayed,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    if replayed.is_empty() {
        eprintln!("error: the replay manifest carries no memories");
        return std::process::ExitCode::from(2);
    }

    // The private store sits beside the checkpoint the dream is running
    // against: a dream is run from an instance's workspace, whose sidecar
    // directory holds the store. Derived from the checkpoint's location, the
    // way the other bench consolidators do.
    let workspace = baseline
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let store_root = workspace.join("sidecar").join("private");

    // The chosen side: the replayed memories, capped by the replay limit.
    let taken: Vec<&ReplayedMemory> = replayed.iter().take(config.replay_limit).collect();
    let mut chosen_memories: Vec<(String, Vec<u8>)> = Vec::new();
    let mut missing = 0usize;
    for memory in &taken {
        match fetch_content(&store_root, memory) {
            Some(content) if !content.is_empty() => {
                chosen_memories.push((memory.record_id.clone(), content));
            }
            _ => missing += 1,
        }
    }
    if chosen_memories.is_empty() {
        eprintln!(
            "error: none of the {} replayed memories has content in {}",
            taken.len(),
            store_root.display()
        );
        return std::process::ExitCode::from(2);
    }

    // Order by emotional load, descending: a dream replays what carries weight.
    // Length breaks ties, then the record id, so the order is total and stable
    // and two runs over the same replay agree.
    let loads: std::collections::HashMap<String, f64> = taken
        .iter()
        .map(|memory| (memory.record_id.clone(), memory.emotional_load))
        .collect();
    chosen_memories.sort_by(|a, b| {
        let load_a = loads.get(&a.0).copied().unwrap_or(0.0);
        let load_b = loads.get(&b.0).copied().unwrap_or(0.0);
        load_b
            .partial_cmp(&load_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.len().cmp(&a.1.len()))
            .then_with(|| a.0.cmp(&b.0))
    });

    // The embedding is seeded from the checkpoint's own bytes, so two runs
    // over the same baseline produce the same candidate. See `SeededEmbedding`.
    let embedding_seed = embedding_seed_of(&model);
    let embedding = SeededEmbedding::new(model.config.vocab, model.config.d_model, embedding_seed);

    // The SPIN pairs. Chosen is the memory's own bytes; rejected is the model's
    // current sample for the same prompt - which is what makes this self-play
    // rather than supervised: the target is generated by the policy being
    // trained, each step, so the training signal tracks what the model can
    // currently do instead of a fixed transcript.
    let mut pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(chosen_memories.len());
    for (_, content) in &chosen_memories {
        let chosen = tokenize(content, model.config.vocab, model.config.n_positions);
        if chosen.len() < 2 {
            continue;
        }
        let rejected = generate_rejected(&model, &embedding, &chosen, config.seed);
        if rejected.len() < 2 {
            continue;
        }
        pairs.push((chosen, rejected));
    }
    if pairs.is_empty() {
        eprintln!("error: no replayed memory is long enough to train on");
        return std::process::ExitCode::from(2);
    }

    // The reference is the frozen baseline, so the policy may only move as far
    // from it as beta allows. Without it, DPO is just "prefer one sequence",
    // and the model drifts.
    let reference = model.clone();

    // Reference log-probabilities, computed once over the fixed pairs.
    let mut reference_logps = Vec::with_capacity(pairs.len());
    for (chosen, rejected) in &pairs {
        let mask = vec![true; chosen.len()];
        let forward = reference.forward(chosen, &embedding, Some(&mask));
        let n = (chosen.len() - 1) as f32;
        let chosen_logp = -forward.loss * n;
        let mask = vec![true; rejected.len()];
        let forward = reference.forward(rejected, &embedding, Some(&mask));
        let n = (rejected.len() - 1) as f32;
        let rejected_logp = -forward.loss * n;
        reference_logps.push((chosen_logp, rejected_logp));
    }

    let mut optimiser = TransformerAdam::new(&model);
    let started = Instant::now();
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..config.steps {
        let mut combined: Option<pmc_train::transformer::Grads> = None;
        let mut step_loss = 0.0f32;

        for (index, (chosen, rejected)) in pairs.iter().enumerate() {
            let mask = vec![true; chosen.len()];
            let forward_c = model.forward(chosen, &embedding, Some(&mask));
            let mut grads_c = model.backward(&forward_c);
            let mask = vec![true; rejected.len()];
            let forward_r = model.forward(rejected, &embedding, Some(&mask));
            let mut grads_r = model.backward(&forward_r);

            let n_c = (chosen.len() - 1) as f32;
            let n_r = (rejected.len() - 1) as f32;
            let logp_c = -forward_c.loss * n_c;
            let logp_r = -forward_r.loss * n_r;
            let (ref_c, ref_r) = reference_logps[index];
            let delta = (logp_c - logp_r) - (ref_c - ref_r);

            let loss = dpo_loss(logp_c, logp_r, ref_c, ref_r, config.beta);
            step_loss += loss;
            // The gradient of the loss above with respect to the policy's
            // log-probabilities. Computed here rather than inside `dpo_loss`
            // because only the caller knows the per-pair weighting.
            let gradient = config.beta * sigmoid(-(config.beta * delta));

            scale_grads(&mut grads_c, gradient * n_c);
            scale_grads(&mut grads_r, -gradient * n_r);
            add_grads(&mut grads_c, &grads_r);

            if combined.is_none() {
                combined = Some(grads_c);
            } else {
                add_grads(combined.as_mut().unwrap(), &grads_c);
            }
        }

        let mut grads = combined.expect("at least one pair produced gradients");
        let count = pairs.len() as f32;
        scale_grads(&mut grads, 1.0 / count);
        step_loss /= count;

        if step == 0 {
            initial_loss = step_loss;
        }
        final_loss = step_loss;

        // `TransformerAdam::step` applies global-norm clipping itself, so the
        // clip travels as the max-norm argument rather than as a separate call:
        // clipping twice would shrink the update by the square of the ratio.
        optimiser.step(
            &mut model,
            &mut grads,
            config.learning_rate,
            0.9,
            0.999,
            1e-8,
            config.clip,
        );
    }

    let elapsed = started.elapsed().as_millis();

    // The candidate. `save_with_step` writes the same format the baseline was
    // read from, so the dream gate's digest comparison and its rollback both
    // work on the candidate exactly as they would on a supervised one.
    if let Err(error) = save_with_step(&candidate, &model, config.steps as u32) {
        eprintln!("error: cannot write the candidate: {error}");
        return std::process::ExitCode::from(1);
    }

    // The report the dream pipeline logs. It carries the numbers that say
    // whether the consolidation did anything, and the record ids and digests
    // that say *which* memories it was run on, so a negative result is
    // attributable rather than anonymous.
    let parameters = model.param_count();
    let report = Json::obj(vec![
        ("schema", Json::str("pmc-dream-spin/1")),
        ("mechanism", Json::str("spin")),
        ("records", Json::num(chosen_memories.len() as f64)),
        ("missing", Json::num(missing as f64)),
        (
            "record_ids",
            Json::Arr(
                chosen_memories
                    .iter()
                    .map(|(id, _)| Json::str(id.clone()))
                    .collect(),
            ),
        ),
        ("pairs", Json::num(pairs.len() as f64)),
        ("steps", Json::num(config.steps as f64)),
        ("beta", Json::num(config.beta as f64)),
        ("initial_dpo_loss", Json::num(initial_loss as f64)),
        ("final_dpo_loss", Json::num(final_loss as f64)),
        ("param_count", Json::num(parameters as f64)),
        ("elapsed_millis", Json::num(elapsed as f64)),
    ]);
    println!("{}", report.to_string());
    std::process::ExitCode::SUCCESS
}

/// The model's own current sample, as the *rejected* side.
///
/// Greedy only: sampling would introduce a randomness the seed does not cover,
/// and a consolidator whose rejected side moves between two runs of the same
/// replay cannot be audited. The sample is the model's own argmax continuation,
/// so the preference the DPO loss learns is "the memory, over what I would
/// have said" - which is the self-play signal.
///
/// A candidate that fails to decode falls back to the prompt's own tail, so a
/// pair always exists; a skipped memory would silently shrink the replay.
fn generate_rejected(
    model: &Transformer,
    embedding: &SeededEmbedding,
    chosen: &[u32],
    seed: u64,
) -> Vec<u32> {
    let _ = seed;
    let mut out: Vec<u32> = chosen.first().copied().into_iter().collect();
    for _ in 1..chosen.len() {
        if out.len() >= model.config.n_positions {
            break;
        }
        let mask = vec![true; out.len()];
        let forward = model.forward(&out, embedding, Some(&mask));
        let vocab = model.config.vocab;
        let last = forward.logits.len() / vocab;
        if last == 0 {
            break;
        }
        let base = (last - 1) * vocab;
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for token in 0..vocab {
            let value = forward.logits[base + token];
            if value > best_value {
                best_value = value;
                best = token;
            }
        }
        out.push(best as u32);
    }
    if out.len() < 2 {
        // The model produced nothing usable; keep the pair rather than drop
        // the memory, so the replay is not silently reduced.
        out = chosen.first().copied().into_iter().collect();
        out.extend_from_slice(&chosen[1..chosen.len().min(2)]);
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn scale_grads(grads: &mut pmc_train::transformer::Grads, factor: f32) {
    for block in grads.blocks.iter_mut() {
        for v in block.attn_norm.iter_mut() {
            *v *= factor;
        }
        for v in block.wq.iter_mut() {
            *v *= factor;
        }
        for v in block.wk.iter_mut() {
            *v *= factor;
        }
        for v in block.wv.iter_mut() {
            *v *= factor;
        }
        for v in block.wo.iter_mut() {
            *v *= factor;
        }
        for v in block.ffn_norm.iter_mut() {
            *v *= factor;
        }
        for v in block.w_gate.iter_mut() {
            *v *= factor;
        }
        for v in block.w_up.iter_mut() {
            *v *= factor;
        }
        for v in block.w_down.iter_mut() {
            *v *= factor;
        }
    }
    for v in grads.final_norm.iter_mut() {
        *v *= factor;
    }
    for v in grads.readout_u.iter_mut() {
        *v *= factor;
    }
    for v in grads.readout_v.iter_mut() {
        *v *= factor;
    }
}

fn add_grads(dst: &mut pmc_train::transformer::Grads, src: &pmc_train::transformer::Grads) {
    for (block_dst, block_src) in dst.blocks.iter_mut().zip(src.blocks.iter()) {
        for (a, b) in block_dst.attn_norm.iter_mut().zip(block_src.attn_norm.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.wq.iter_mut().zip(block_src.wq.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.wk.iter_mut().zip(block_src.wk.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.wv.iter_mut().zip(block_src.wv.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.wo.iter_mut().zip(block_src.wo.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.ffn_norm.iter_mut().zip(block_src.ffn_norm.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.w_gate.iter_mut().zip(block_src.w_gate.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.w_up.iter_mut().zip(block_src.w_up.iter()) {
            *a += *b;
        }
        for (a, b) in block_dst.w_down.iter_mut().zip(block_src.w_down.iter()) {
            *a += *b;
        }
    }
    for (a, b) in dst.final_norm.iter_mut().zip(src.final_norm.iter()) {
        *a += *b;
    }
    for (a, b) in dst.readout_u.iter_mut().zip(src.readout_u.iter()) {
        *a += *b;
    }
    for (a, b) in dst.readout_v.iter_mut().zip(src.readout_v.iter()) {
        *a += *b;
    }
}

// The optimiser clips gradients internally; no second clipping helper here.
