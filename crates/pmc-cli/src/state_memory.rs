// SPDX-License-Identifier: GPL-3.0-only

//! Fuse a trained PMC state encoder into a GGUF as dense neural weights.
//!
//! The `PMCW` weights file written by
//! `crates/pmc-train/examples/train_pmc_state.rs` is a *portable checkpoint*,
//! not the model. This module turns it into the thing a loader actually reads:
//! the encoder's parameters, appended to a base GGUF as `pmc.state.encoder.*`
//! F32 tensors alongside the model it belongs to.
//!
//! That is the whole point of the bank. Memory that lives in a JSON sidecar is
//! a file a loader ignores, a backup can drop, and a person can edit without
//! leaving a trace. Memory that lives in the model file travels with the model,
//! is covered by the model's digest, and needs no second program to be present
//! at inference time. Nothing here parses, tokens, or *reads* the state: it
//! copies numbers.
//!
//! The tensor names are namespaced under `pmc.state.` like every other state
//! bank, and the geometry is recorded as metadata rather than inferred, so a
//! loader that finds a bank it does not understand can report it instead of
//! guessing.

use std::path::{Path, PathBuf};

use pmc_core::Json;
use pmc_gguf::writer::GgufWriter;
use pmc_train::pmc_state::{read_weights, PmcStateConfig, PmcStateModel};

/// The namespace every state bank this command writes lives under.
pub const NAMESPACE: &str = "pmc.state.encoder.";

/// A fused state-memory request.
pub struct StateMemoryRequest {
    /// The base GGUF the encoder is fused into.
    pub base: PathBuf,
    /// The trained `PMCW` weights file.
    pub weights: PathBuf,
    /// The output GGUF.
    pub out: PathBuf,
    /// An optional external manifest path.
    pub manifest: Option<PathBuf>,
}

/// Read the architecture header of a `PMCW` file.
///
/// The header is the five integers the file begins with, in this order:
/// projection, hidden, layers, key dimensions, then the host count and each
/// host dimension. They are enough to rebuild the configuration the shapes
/// describe, and they are checked against the file's own digest on read, so a
/// truncated or edited file is refused rather than silently reinterpreted.
pub fn read_architecture(path: &Path) -> Result<(PmcStateConfig, Vec<usize>), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 28 {
        return Err(format!("{} is too short to be a weights file", path.display()));
    }
    let u32_at = |offset: usize| -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    let projection = u32_at(8) as usize;
    let hidden = u32_at(12) as usize;
    let layers = u32_at(16) as usize;
    let key_dim = u32_at(20) as usize;
    let hosts = u32_at(24) as usize;

    let mut config = PmcStateConfig::default();
    config.projection_dim = projection;
    config.hidden_dim = hidden;
    config.layers = layers;
    config.key_dim = key_dim;

    let mut host_dims = Vec::with_capacity(hosts);
    for host in 0..hosts {
        host_dims.push(u32_at(28 + host * 4) as usize);
    }
    Ok((config, host_dims))
}

/// Fuse the encoder's parameters into a GGUF as dense state tensors.
pub fn fuse(request: &StateMemoryRequest) -> Result<Json, String> {
    let (config, host_dims) = read_architecture(&request.weights)?;
    let model: PmcStateModel =
        read_weights(&request.weights, &config).map_err(|error| error.to_string())?;

    let reader = pmc_gguf::GgufReader::open(&request.base).map_err(|error| error.to_string())?;
    let mut writer = GgufWriter::from_source(&reader).map_err(|error| error.to_string())?;

    // The hidden size of the base model, when the file declares one. A missing
    // or non-numeric value is not fatal: the bank's geometry is recorded
    // separately, and a model whose loader has no `n_embd` has no hidden state
    // to project anyway.
    let base_n_embd = reader
        .get("n_embd")
        .and_then(|value| match value {
            pmc_gguf::GgufValue::U32(value) => Some(*value as u64),
            pmc_gguf::GgufValue::U64(value) => Some(*value),
            pmc_gguf::GgufValue::I32(value) if *value >= 0 => Some(*value as u64),
            pmc_gguf::GgufValue::I64(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        })
        .unwrap_or(0);

    append_encoder(&mut writer, &config, &host_dims, &model).map_err(|error| error.to_string())?;

    // The geometry the fused bank describes, so a reader never has to infer it.
    // Recorded as typed GGUF metadata rather than one JSON blob, because a GGUF
    // value is a typed scalar or array: a loader that understands one key can
    // read it without parsing anything, and a loader that does not can report
    // the key it does not know.
    let names = written_tensor_names(host_dims.len(), config.layers);
    writer
        .set_metadata(
            &format!("{NAMESPACE}geometry"),
            pmc_gguf::GgufValue::array(
                pmc_gguf::GgufValue::U64(0).type_id(),
                [
                    config.projection_dim as u64,
                    config.hidden_dim as u64,
                    config.layers as u64,
                    config.key_dim as u64,
                    host_dims.len() as u64,
                ]
                .iter()
                .copied()
                .map(pmc_gguf::GgufValue::U64)
                .collect(),
            ),
        )
        .map_err(|error| error.to_string())?;
    writer
        .set_metadata(
            &format!("{NAMESPACE}hosts"),
            pmc_gguf::GgufValue::array(
                pmc_gguf::GgufValue::U64(0).type_id(),
                host_dims.iter().map(|d| pmc_gguf::GgufValue::U64(*d as u64)).collect(),
            ),
        )
        .map_err(|error| error.to_string())?;
    writer
        .set_metadata(
            &format!("{NAMESPACE}n_embd"),
            pmc_gguf::GgufValue::U64(base_n_embd),
        )
        .map_err(|error| error.to_string())?;
    writer
        .set_metadata(
            &format!("{NAMESPACE}architecture"),
            pmc_gguf::GgufValue::String(config.architecture()),
        )
        .map_err(|error| error.to_string())?;
    // The tensor names, published so a reader can verify the bank against the
    // naming scheme without recomputing it.
    writer
        .set_metadata(
            &format!("{NAMESPACE}tensors"),
            pmc_gguf::GgufValue::array(
                pmc_gguf::GgufValue::String(String::new()).type_id(),
                names
                    .iter()
                    .map(|name| pmc_gguf::GgufValue::String(name.clone()))
                    .collect(),
            ),
        )
        .map_err(|error| error.to_string())?;
    // The digest of the trained parameters that were fused, so a fused model
    // can be checked against the weights file it came from.
    writer
        .set_metadata(
            &format!("{NAMESPACE}weight_digest"),
            pmc_gguf::GgufValue::String(model.weight_digest()),
        )
        .map_err(|error| error.to_string())?;

    writer
        .write_to(&request.out)
        .map_err(|error| error.to_string())?;

    // A manifest records what was fused and from where, by basename: the file
    // is portable, so an absolute path in it would be a leak. The tensor names
    // come from the same helper the metadata above was written with, so the two
    // cannot drift apart.
    let manifest_path = request.manifest.clone().unwrap_or_else(|| {
        let mut path = request.out.clone();
        path.set_extension("pmc-state.json");
        path
    });
    let manifest = Json::obj(vec![
        ("schema", Json::str("pmc-state-memory/1")),
        ("base", Json::str(base_name(&request.base))),
        ("weights", Json::str(base_name(&request.weights))),
        ("out", Json::str(base_name(&request.out))),
        ("architecture", Json::str(config.architecture())),
        ("param_count", Json::num(model.param_count() as f64)),
        ("weight_digest", Json::str(model.weight_digest())),
        ("n_embd", Json::num(base_n_embd as f64)),
        ("tensors", Json::Arr(names.iter().map(Json::str).collect())),
    ]);
    std::fs::write(&manifest_path, manifest.to_string())
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;

    Ok(manifest)
}

/// Append every parameter group of the encoder as one F32 tensor.
///
/// The order is the same as [`written_tensor_names`], which is what the manifest
/// publishes. Writing the gate and the up branch as one `[2h, h]` tensor rather
/// than two per layer is deliberate: they are one SwiGLU, and a loader that
/// reads one reads the pair.
fn append_encoder(
    writer: &mut GgufWriter,
    config: &PmcStateConfig,
    host_dims: &[usize],
    model: &PmcStateModel,
) -> Result<(), String> {
    let p = config.projection_dim as u64;
    let h = config.hidden_dim as u64;
    let k = config.key_dim as u64;

    for (host, dim) in host_dims.iter().enumerate() {
        writer.append_tensor_f32(
            &format!("{NAMESPACE}input.{host}"),
            &[p, *dim as u64],
            &model.input[host],
        ).map_err(|e| e.to_string())?;
        writer.append_tensor_f32(
            &format!("{NAMESPACE}input_bias.{host}"),
            &[p],
            &model.input_bias[host],
        ).map_err(|e| e.to_string())?;
    }

    writer.append_tensor_f32(&format!("{NAMESPACE}expand"), &[h, p], &model.expand).map_err(|e| e.to_string())?;
    writer.append_tensor_f32(&format!("{NAMESPACE}expand_bias"), &[h], &model.expand_bias).map_err(|e| e.to_string())?;

    for layer in 0..config.layers {
        let gate = &model.gate[layer];
        let up = &model.up[layer];
        // One [layers, 2*hidden, hidden] tensor rather than two per layer: the
        // gate and the up branch are one SwiGLU, and a loader that reads one
        // reads the pair.
        let mut merged = Vec::with_capacity(2 * gate.len());
        merged.extend_from_slice(gate);
        merged.extend_from_slice(up);
        writer.append_tensor_f32(
            &format!("{NAMESPACE}layer.{layer}"),
            &[2 * h, h],
            &merged,
        ).map_err(|e| e.to_string())?;
        let mut bias = Vec::with_capacity(model.gate_bias[layer].len() + model.up_bias[layer].len());
        bias.extend_from_slice(&model.gate_bias[layer]);
        bias.extend_from_slice(&model.up_bias[layer]);
        writer.append_tensor_f32(
            &format!("{NAMESPACE}layer_bias.{layer}"),
            &[2 * h],
            &bias,
        ).map_err(|e| e.to_string())?;
    }

    writer.append_tensor_f32(&format!("{NAMESPACE}head"), &[k, h], &model.head).map_err(|e| e.to_string())?;
    writer.append_tensor_f32(&format!("{NAMESPACE}head_bias"), &[k], &model.head_bias).map_err(|e| e.to_string())?;

    Ok(())
}

fn base_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

/// Every tensor name the bank writes, in the order they are appended.
///
/// The manifest lists them so a reader can check the bank against the naming
/// scheme without recomputing it, and so a test can assert the two agree.
pub fn written_tensor_names(hosts: usize, layers: usize) -> Vec<String> {
    let mut names = Vec::new();
    for host in 0..hosts {
        names.push(format!("{NAMESPACE}input.{host}"));
        names.push(format!("{NAMESPACE}input_bias.{host}"));
    }
    names.push(format!("{NAMESPACE}expand"));
    names.push(format!("{NAMESPACE}expand_bias"));
    for layer in 0..layers {
        names.push(format!("{NAMESPACE}layer.{layer}"));
        names.push(format!("{NAMESPACE}layer_bias.{layer}"));
    }
    names.push(format!("{NAMESPACE}head"));
    names.push(format!("{NAMESPACE}head_bias"));
    names
}
