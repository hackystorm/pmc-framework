// SPDX-License-Identifier: GPL-3.0-only

//! Train the PMC state encoder on a views file and report the gate numbers.
//!
//! The views file is the `PMCV` written by `bench/pmc_state_views.py`: the
//! raw hidden states of both substrates for the training facts and the
//! training questions, and nothing else. The encoder is the v4 mechanism:
//! a trainable input projection per host, a shared residual body, an L2 key.
//!
//! The cross-validation is the pre-registered gate. It trains on all but one
//! fold, then reports the retrieval over every label, counting only the
//! held-out ones. `host_b_question` is the number the gate reads: the query
//! written by the second substrate, answered from the keys the first
//! substrate wrote.
//!
//! Run it with:
//!
//! ```text
//! PMC_VIEWS=views.pmc PMC_REPORT=report.json \
//!   cargo run --release --offline --example train_pmc_state
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use pmc_core::Json;
use pmc_train::pmc_state::{
    self, read_views, train, FoldReport, PmcStateConfig, RetrievalReport,
};

/// Override a configuration value from the environment, keeping the default
/// when the variable is absent or unparsable-but-empty.
fn from_env(name: &str, config: &mut PmcStateConfig, apply: fn(f64, &mut PmcStateConfig)) {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => {
            if let Ok(number) = value.trim().parse::<f64>() {
                apply(number, config);
            }
        }
        _ => {}
    }
}

fn apply_seed(value: f64, config: &mut PmcStateConfig) {
    config.seed = value as u64;
}
fn apply_epochs(value: f64, config: &mut PmcStateConfig) {
    config.epochs = value.max(1.0) as usize;
}
fn apply_lr(value: f64, config: &mut PmcStateConfig) {
    config.learning_rate = value as f32;
}
fn apply_decay(value: f64, config: &mut PmcStateConfig) {
    config.weight_decay = value as f32;
}
fn apply_temperature(value: f64, config: &mut PmcStateConfig) {
    config.temperature = value as f32;
}
fn apply_projection(value: f64, config: &mut PmcStateConfig) {
    config.projection_dim = value.max(1.0) as usize;
}
fn apply_hidden(value: f64, config: &mut PmcStateConfig) {
    config.hidden_dim = value.max(1.0) as usize;
}
fn apply_layers(value: f64, config: &mut PmcStateConfig) {
    config.layers = value.max(1.0) as usize;
}
fn apply_key(value: f64, config: &mut PmcStateConfig) {
    config.key_dim = value.max(1.0) as usize;
}
fn apply_clip(value: f64, config: &mut PmcStateConfig) {
    config.max_grad_norm = value as f32;
}

fn counts_json(counts: &RetrievalReport) -> Json {
    Json::obj(vec![
        ("labels", Json::num(counts.labels as f64)),
        ("host_a_fact", Json::num(counts.host_a_fact as f64)),
        ("host_a_question", Json::num(counts.host_a_question as f64)),
        ("host_b_fact", Json::num(counts.host_b_fact as f64)),
        ("host_b_question", Json::num(counts.host_b_question as f64)),
    ])
}

fn fold_json(fold: &FoldReport) -> Json {
    Json::obj(vec![
        ("fold", Json::num(fold.fold as f64)),
        (
            "held_out",
            Json::Arr(fold.held_out.iter().map(|l| Json::num(*l as f64)).collect()),
        ),
        ("counts", counts_json(&fold.counts)),
        ("final_loss", Json::num(fold.final_loss as f64)),
    ])
}

fn main() {
    let started = std::time::Instant::now();
    let views_path = match env::var("PMC_VIEWS") {
        Ok(value) => PathBuf::from(value),
        Err(_) => {
            eprintln!("PMC_VIEWS is required: the path to the views file");
            std::process::exit(2);
        }
    };
    let entries = match read_views(Path::new(&views_path)) {
        Ok(entries) => entries,
        Err(error) => {
            eprintln!("cannot read {}: {error}", views_path.display());
            std::process::exit(2);
        }
    };

    let mut config = PmcStateConfig::default();
    from_env("PMC_SEED", &mut config, apply_seed);
    from_env("PMC_EPOCHS", &mut config, apply_epochs);
    from_env("PMC_LR", &mut config, apply_lr);
    from_env("PMC_DECAY", &mut config, apply_decay);
    from_env("PMC_TEMPERATURE", &mut config, apply_temperature);
    from_env("PMC_PROJECTION", &mut config, apply_projection);
    from_env("PMC_HIDDEN", &mut config, apply_hidden);
    from_env("PMC_LAYERS", &mut config, apply_layers);
    from_env("PMC_KEY", &mut config, apply_key);
    from_env("PMC_CLIP", &mut config, apply_clip);
    if let Err(error) = config.validate() {
        eprintln!("invalid configuration: {error}");
        std::process::exit(2);
    }

    // The fold count is a run setting, not an architecture setting: it is how
    // the pre-registered gate is run, and it is recorded in the report.
    let mut fold_count = 6usize;
    if let Ok(value) = env::var("PMC_FOLDS") {
        if let Ok(number) = value.trim().parse::<usize>() {
            fold_count = number.max(2);
        }
    }

    let labels = {
        let mut all: Vec<usize> = entries.iter().map(|entry| entry.label).collect();
        all.sort();
        all.dedup();
        all
    };
    eprintln!(
        "views {} labels {} config {}",
        entries.len(),
        labels.len(),
        config.architecture()
    );

    let (model, outcome) = match train(&config, &entries, &labels, fold_count) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("training failed: {error}");
            std::process::exit(1);
        }
    };
    eprintln!("host_b_question is the cross-host, cross-view gate number");
    // The report is written even when the cross-validation is skipped, and
    // the folds it does contain are the ones the baseline must reuse.
    let folds: Vec<FoldReport> = outcome.folds.clone();
    let retrieval = outcome.retrieval;
    let elapsed = started.elapsed().as_millis();

    if let Ok(path) = env::var("PMC_WEIGHTS") {
        if let Err(error) = pmc_state::write_weights(Path::new(&path), &model) {
            eprintln!("cannot write {}: {error}", path);
        }
    }

    let report = Json::obj(vec![
        ("schema", Json::str("pmc-state-train/1")),
        ("views", Json::str(views_path.display().to_string())),
        (
            "view_count",
            Json::num(entries.len() as f64),
        ),
        ("label_count", Json::num(labels.len() as f64)),
        ("config", Json::obj(vec![
            ("architecture", Json::str(config.architecture())),
            ("seed", Json::num(config.seed as f64)),
            ("epochs", Json::num(config.epochs as f64)),
            ("learning_rate", Json::num(config.learning_rate as f64)),
            ("weight_decay", Json::num(config.weight_decay as f64)),
            ("temperature", Json::num(config.temperature as f64)),
            ("projection_dim", Json::num(config.projection_dim as f64)),
            ("hidden_dim", Json::num(config.hidden_dim as f64)),
            ("layers", Json::num(config.layers as f64)),
            ("key_dim", Json::num(config.key_dim as f64)),
            ("max_grad_norm", Json::num(config.max_grad_norm as f64)),
            ("cv_folds", Json::num(fold_count as f64)),
        ])),
        ("host_dims", Json::Arr(model.host_dims.iter().map(|d| Json::num(*d as f64)).collect())),
        ("param_count", Json::num(outcome.param_count as f64)),
        ("weight_digest", Json::str(outcome.weight_digest.clone())),
        ("initial_loss", Json::num(outcome.initial_loss as f64)),
        ("final_loss", Json::num(outcome.final_loss as f64)),
        ("elapsed_millis", Json::num(elapsed as f64)),
        ("train_retrieval", counts_json(&retrieval)),
        ("folds", Json::Arr(folds.iter().map(fold_json).collect())),
    ]);

    let printed = report.to_string();
    println!("{printed}");
    if let Ok(path) = env::var("PMC_REPORT") {
        if let Some(parent) = Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(error) = fs::write(&path, format!("{printed}\n")) {
            eprintln!("cannot write {path}: {error}");
        }
    }
}
