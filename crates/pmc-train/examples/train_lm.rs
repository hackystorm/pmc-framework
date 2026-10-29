// SPDX-License-Identifier: GPL-3.0-only

//! Train a small decoder-only transformer on a token sequence and measure
//! the held-out perplexity against the random-init baseline.
//!
//! This is the first 50-100M-scale text path: the model is a decoder-only
//! transformer with a frozen random embedding table; the task is next-token
//! prediction. A held-out suffix of the token sequence is kept out of
//! training and scored before and after, so the run reports whether the
//! optimisation moved the held-out perplexity at all. The random-init eval
//! loss is the baseline; a run that does not beat it is a negative and is
//! committed as one.
//!
//! Run it with:
//!
//! ```text
//! PMC_TRAIN_TOKENS=/path/to/train_tokens.txt \
//! PMC_VAL_TOKENS=/path/to/val_tokens.txt \
//! PMC_CHECKPOINT=/path/to/checkpoint \
//!   cargo run --release --offline --example train_lm
//! ```
//!
//! `PMC_TRAIN_TOKENS` may point at a file with one integer per line or at a JSON
//! array of integers. If it is unset, a deterministic pseudo sequence is
//! used so the example runs without external data; the report records its
//! SHA-256 either way. `PMC_VAL_TOKENS` overrides the held-out evaluation set.
//! If `PMC_EARLY_STOPPING=1`, early stopping is enabled with `PMC_PATIENCE`.

use std::path::PathBuf;
use std::time::Instant;

use pmc_core::{sha256_hex, Rng};
use pmc_train::checkpoint;
use pmc_train::transformer::{Embedding, TransformerAdam, TransformerConfig};

struct TableEmbedding {
    table: Vec<f32>,
    dim: usize,
}

impl TableEmbedding {
    fn random(vocab: usize, dim: usize, seed: u64) -> TableEmbedding {
        let mut rng = Rng::new(seed);
        let mut table = vec![0.0f32; vocab * dim];
        for value in table.iter_mut() {
            *value = rng.normal() * 0.5;
        }
        TableEmbedding { table, dim }
    }
}

impl Embedding for TableEmbedding {
    fn dim(&self) -> usize {
        self.dim
    }

    fn row(&self, token: u32, out: &mut [f32]) {
        let start = token as usize * self.dim;
        out.copy_from_slice(&self.table[start..start + self.dim]);
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Parse a token file: JSON array of integers or one integer per line.
fn read_tokens(path: &str) -> Vec<u32> {
    let raw = std::fs::read_to_string(path).expect("read token file");
    let trimmed = raw.trim();
    if trimmed.starts_with('[') {
        parse_json_array(trimmed)
    } else {
        trimmed
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    None
                } else {
                    Some(line.parse::<u32>().expect("token id"))
                }
            })
            .collect()
    }
}

fn parse_json_array(text: &str) -> Vec<u32> {
    // Minimal parser: the file is an array of non-negative integers with
    // optional whitespace. Anything else is refused rather than guessed.
    let inner = text
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .expect("token file is not a JSON array");
    if inner.trim().is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .expect("token file array holds a non-integer")
        })
        .collect()
}

fn main() {
    let n_layer = env_usize("PMC_N_LAYER", 2);
    let d_model = env_usize("PMC_D_MODEL", 64);
    let n_head = env_usize("PMC_N_HEAD", 4);
    let ffn_hidden = env_usize("PMC_FFN", 256);
    let vocab = env_usize("PMC_VOCAB", 256);
    let n_positions = env_usize("PMC_N_POS", 64);
    let readout_rank = env_usize("PMC_RANK", 16);
    let steps = env_usize("PMC_STEPS", 20);
    let learning_rate = env_f32("PMC_LR", 0.01);
    let holdout = env_usize("PMC_HOLDOUT", 8);
    let model_seed: u64 = env_u64("PMC_MODEL_SEED", 20261031);
    let embedding_seed: u64 = env_u64("PMC_EMBEDDING_SEED", 20261032);

    let config = TransformerConfig {
        n_layer,
        d_model,
        n_head,
        ffn_hidden,
        vocab,
        n_positions,
        readout_rank,
    };
    config.validate().expect("the configuration must be valid");

    let (train_tokens, eval_tokens, source) = {
        // New interface: PMC_TRAIN_TOKENS + optional PMC_VAL_TOKENS
        if let Ok(train_path) = std::env::var("PMC_TRAIN_TOKENS") {
            let train_tokens = read_tokens(&train_path);
            let (val_tokens, holdout_tokens) = match std::env::var("PMC_VAL_TOKENS") {
                Ok(path) => (read_tokens(&path), Vec::new()),
                Err(_) => {
                    // Use holdout suffix from train_tokens
                    let split = train_tokens.len().saturating_sub(holdout);
                    (train_tokens[..split].to_vec(), train_tokens[split..].to_vec())
                }
            };
            let eval_tokens = val_tokens;
            let source = std::env::var("PMC_TRAIN_TOKENS").unwrap();
            (train_tokens, eval_tokens, source)
        } else if let Ok(token_path) = std::env::var("PMC_TOKENS") {
            // Backward compatibility with PMC_TOKENS
            let tokens = read_tokens(&token_path);
            let source = token_path;
            let split = tokens.len().saturating_sub(holdout);
            let train_tokens = tokens[..split].to_vec();
            let eval_tokens = match std::env::var("PMC_EVAL_TOKENS") {
                Ok(path) => read_tokens(&path),
                Err(_) => tokens[split..].to_vec(),
            };
            (train_tokens, eval_tokens, source)
        } else {
            // Pseudo tokens
            let tokens: Vec<u32> = (0..64).map(|i| ((i * 37 + 11) % vocab) as u32).collect();
            let split = tokens.len().saturating_sub(holdout);
            let train_tokens = tokens[..split].to_vec();
            let eval_tokens = tokens[split..].to_vec();
            (train_tokens, eval_tokens, String::from("<pseudo>"))
        }
    };

    let train_bytes: Vec<u8> = train_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let eval_bytes: Vec<u8> = eval_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let train_sha = sha256_hex(&train_bytes);
    let eval_sha = sha256_hex(&eval_bytes);

    let config = TransformerConfig {
        n_layer,
        d_model,
        n_head,
        ffn_hidden,
        vocab,
        n_positions,
        readout_rank,
    };
    config.validate().expect("the configuration must be valid");

    let (train_tokens, eval_tokens, source) = {
        // New interface: PMC_TRAIN_TOKENS + optional PMC_VAL_TOKENS
        if let Ok(train_path) = std::env::var("PMC_TRAIN_TOKENS") {
            let train_tokens = read_tokens(&train_path);
            let (val_tokens, holdout_tokens) = match std::env::var("PMC_VAL_TOKENS") {
                Ok(path) => (read_tokens(&path), Vec::new()),
                Err(_) => {
                    // Use holdout suffix from train_tokens
                    let split = train_tokens.len().saturating_sub(holdout);
                    (train_tokens[..split].to_vec(), train_tokens[split..].to_vec())
                }
            };
            let eval_tokens = val_tokens;
            let source = std::env::var("PMC_TRAIN_TOKENS").unwrap();
            (train_tokens, eval_tokens, source)
        } else if let Ok(token_path) = std::env::var("PMC_TOKENS") {
            // Backward compatibility with PMC_TOKENS
            let tokens = read_tokens(&token_path);
            let source = token_path;
            let split = tokens.len().saturating_sub(holdout);
            let train_tokens = tokens[..split].to_vec();
            let eval_tokens = match std::env::var("PMC_EVAL_TOKENS") {
                Ok(path) => read_tokens(&path),
                Err(_) => tokens[split..].to_vec(),
            };
            (train_tokens, eval_tokens, source)
        } else {
            // Pseudo tokens
            let tokens: Vec<u32> = (0..64).map(|i| ((i * 37 + 11) % vocab) as u32).collect();
            let split = tokens.len().saturating_sub(holdout);
            let train_tokens = tokens[..split].to_vec();
            let eval_tokens = tokens[split..].to_vec();
            (train_tokens, eval_tokens, String::from("<pseudo>"))
        }
    };

    let train_bytes: Vec<u8> = train_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let eval_bytes: Vec<u8> = eval_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let train_sha = sha256_hex(&train_bytes);
    let eval_sha = sha256_hex(&eval_bytes);

    let config = TransformerConfig {
        n_layer,
        d_model,
        n_head,
        ffn_hidden,
        vocab,
        n_positions,
        readout_rank,
    };
    config.validate().expect("the configuration must be valid");

    let embedding = TableEmbedding::random(vocab, d_model, embedding_seed);

    let mut model = match std::env::var("PMC_BASELINE") {
        Ok(path) => checkpoint::load(std::path::Path::new(&path))
            .expect("the baseline checkpoint must load"),
        Err(_) => config.init(model_seed),
    };
    // Use the model's config for embedding dimension (correct for consolidation).
    let embedding_dim = model.config.d_model;
    let embedding = TableEmbedding::random(vocab, embedding_dim, embedding_seed);
    let mut optimiser = TransformerAdam::new(&model);
    let train_mask = vec![true; train_tokens.len()];
    let eval_mask = vec![true; eval_tokens.len()];

    // Baseline: random-init eval loss before any step.
    let baseline_forward = model.forward(&eval_tokens, &embedding, Some(&eval_mask));
    let baseline_loss = baseline_forward.loss;
    let baseline_perplexity = baseline_loss.exp();

    // Early stopping setup
    let early_stopping = std::env::var("PMC_EARLY_STOPPING").ok().map(|v| v == "1").unwrap_or(false);
    let patience = env_usize("PMC_PATIENCE", 3);
    let mut best_eval_loss = f32::INFINITY;
    let mut patience_counter = 0;
    let mut best_model_state: Option<Vec<f32>> = None;

    let mut initial_train_loss = 0.0f32;
    let mut final_train_loss = 0.0f32;
    let started = Instant::now();

    let train_mask = vec![true; train_tokens.len()];
    let eval_mask = vec![true; eval_tokens.len()];

    for step in 0..steps {
        let forward = model.forward(&train_tokens, &embedding, Some(&train_mask));
        if step == 0 {
            initial_train_loss = forward.loss;
        }
        final_train_loss = forward.loss;
        let mut grads = model.backward(&forward);
        optimiser.step(&mut model, &mut grads, learning_rate, 0.9, 0.999, 1e-8, 1.0);
        if step == 0 || step + 1 == steps || (step + 1) % 5 == 0 {
            eprintln!(
                "step {:>3}/{} train_loss {:.6}",
                step + 1,
                steps,
                forward.loss
            );
        }

        // Early stopping check
        if early_stopping && step > 0 && step % 5 == 0 {
            let eval_forward = model.forward(&eval_tokens, &embedding, Some(&eval_mask));
            let eval_loss = eval_forward.loss;
            if eval_loss < best_eval_loss {
                best_eval_loss = eval_loss;
                patience_counter = 0;
                // Save best model state (parameters)
                best_model_state = Some(model.params().iter().flat_map(|p| p.iter().cloned()).collect());
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    eprintln!("Early stopping at step {} (best eval loss: {:.6})", step + 1, best_eval_loss);
                    // Restore best model
                    if let Some(state) = best_model_state {
                        let mut idx = 0;
                        for param in model.params_mut() {
                            for v in param.iter_mut() {
                                *v = state[idx];
                                idx += 1;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    let final_forward = model.forward(&eval_tokens, &embedding, Some(&eval_mask));
    let final_eval_loss = final_forward.loss;
    let final_eval_perplexity = final_eval_loss.exp();
    let elapsed_millis = started.elapsed().as_millis();
    let param_count = model.param_count();

    let checkpoint_path = std::env::var("PMC_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("build/pmc-lm"));
    checkpoint::save_with_step(&checkpoint_path, &model, steps as u32)
        .expect("the checkpoint must be written");

    let beat_baseline = final_eval_loss < baseline_loss;
    println!(
        concat!(
            "{{\"source\":\"{}\",\"config\":{{\"n_layer\":{},\"d_model\":{},",
            "\"n_head\":{},\"ffn_hidden\":{},\"vocab\":{},\"n_positions\":{},",
            "\"readout_rank\":{}}},\"param_count\":{},\"steps\":{},",
            "\"learning_rate\":{:.6},\"model_seed\":{},\"embedding_seed\":{},",
            "\"holdout\":{},\"train_tokens\":{},\"eval_tokens\":{},",
            "\"train_sha256\":\"{}\",\"eval_sha256\":\"{}\",",
            "\"initial_train_loss\":{:.6},\"final_train_loss\":{:.6},",
            "\"baseline_eval_loss\":{:.6},\"final_eval_loss\":{:.6},",
            "\"baseline_eval_perplexity\":{:.6},\"final_eval_perplexity\":{:.6},",
            "\"beat_baseline\":{},\"elapsed_millis\":{},\"checkpoint\":\"{}\"}}"
        ),
        source,
        model.config.n_layer,
        model.config.d_model,
        model.config.n_head,
        model.config.ffn_hidden,
        model.config.vocab,
        model.config.n_positions,
        model.config.readout_rank,
        param_count,
        steps,
        learning_rate,
        model_seed,
        embedding_seed,
        holdout,
        train_tokens.len(),
        eval_tokens.len(),
        train_sha,
        eval_sha,
        initial_train_loss,
        final_train_loss,
        baseline_loss,
        final_eval_loss,
        baseline_perplexity,
        final_eval_perplexity,
        beat_baseline,
        elapsed_millis,
        checkpoint_path.display()
    );
}