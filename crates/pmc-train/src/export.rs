// SPDX-License-Identifier: GPL-3.0-only

//! Artifact export.
//!
//! The trainer writes a checkpoint directory with a model description, a
//! binary weight blob, and a manifest naming the digest of every file. The
//! manifest is what makes a run reproducible: the weight digest is derived
//! from the seed and the data, so two runs that agree are provably the same
//! run, not merely similar ones.

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use pmc_core::{sha256_hex, Json, ARTIFACT_FORMAT_VERSION};

use crate::model::Mlp;
use crate::TrainConfig;

fn file_entry(name: &str, bytes: &[u8]) -> Json {
    Json::obj(vec![
        ("name", Json::str(name)),
        ("bytes", Json::num(bytes.len() as f64)),
        ("sha256", Json::str(sha256_hex(bytes))),
    ])
}

pub fn export_checkpoint(
    directory: &Path,
    model: &Mlp,
    config: &TrainConfig,
    metrics: &Json,
) -> std::io::Result<()> {
    fs::create_dir_all(directory)?;

    let weights = model.to_bytes();
    let weight_digest = sha256_hex(&weights);

    let model_json = Json::obj(vec![
        ("format_version", Json::num(ARTIFACT_FORMAT_VERSION as f64)),
        (
            "architecture",
            Json::obj(vec![
                ("kind", Json::str("mlp")),
                ("d_in", Json::num(model.d_in as f64)),
                ("d_hidden", Json::num(model.d_hidden as f64)),
                ("d_out", Json::num(model.d_out as f64)),
            ]),
        ),
        ("seed", Json::num(config.seed as f64)),
        ("steps", Json::num(config.steps as f64)),
        ("learning_rate", Json::num(config.learning_rate as f64)),
        ("keys", Json::num(config.keys as f64)),
        ("weight_digest", Json::str(weight_digest.clone())),
        ("metrics", metrics.clone()),
    ])
    .to_string();

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let manifest = Json::obj(vec![
        ("format_version", Json::num(ARTIFACT_FORMAT_VERSION as f64)),
        ("created_epoch", Json::num(created as f64)),
        ("seed", Json::num(config.seed as f64)),
        ("weight_digest", Json::str(weight_digest)),
        (
            "files",
            Json::Arr(vec![
                file_entry("model.json", model_json.as_bytes()),
                file_entry("weights.bin", &weights),
            ]),
        ),
    ])
    .to_string();

    fs::write(directory.join("model.json"), model_json.as_bytes())?;
    fs::write(directory.join("weights.bin"), &weights)?;
    fs::write(directory.join("manifest.json"), manifest.as_bytes())?;
    Ok(())
}
