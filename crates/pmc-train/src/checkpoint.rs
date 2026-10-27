// SPDX-License-Identifier: GPL-3.0-only

//! Binary transformer checkpoints and a JSON manifest.
//!
//! A checkpoint is a directory containing:
//!
//! * `checkpoint.bin`: the magic bytes `PMCC`, a little-endian `u32` version,
//!   the seven configuration fields as little-endian `u32`, then every
//!   parameter as little-endian `f32` in [`Transformer::params`] order.
//! * `manifest.json`: the configuration, parameter count, training step and
//!   SHA-256 of the raw parameter bytes.
//!
//! Loading verifies the magic, the version, the exact file length, and the
//! digest recorded in the manifest. A truncated or corrupted checkpoint is an
//! error, never a silently different model.

use std::fs;
use std::io;
use std::path::Path;

use pmc_core::{sha256_hex, Json};

use crate::transformer::{Transformer, TransformerConfig};

const MAGIC: [u8; 4] = *b"PMCC";
const VERSION: u32 = 1;
const PARAMS_FILE: &str = "checkpoint.bin";
const MANIFEST_FILE: &str = "manifest.json";
const HEADER_LEN: usize = 4 + 4 + 7 * 4;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    if offset + 4 > bytes.len() {
        return Err(invalid("checkpoint header is truncated"));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn config_json(config: &TransformerConfig) -> Json {
    Json::obj(vec![
        ("n_layer", Json::num(config.n_layer as f64)),
        ("d_model", Json::num(config.d_model as f64)),
        ("n_head", Json::num(config.n_head as f64)),
        ("ffn_hidden", Json::num(config.ffn_hidden as f64)),
        ("vocab", Json::num(config.vocab as f64)),
        ("n_positions", Json::num(config.n_positions as f64)),
        ("readout_rank", Json::num(config.readout_rank as f64)),
    ])
}

fn encode_config(config: &TransformerConfig, out: &mut Vec<u8>) -> io::Result<()> {
    let fields = [
        config.n_layer,
        config.d_model,
        config.n_head,
        config.ffn_hidden,
        config.vocab,
        config.n_positions,
        config.readout_rank,
    ];
    for field in fields {
        let value = u32::try_from(field)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "configuration value exceeds u32"))?;
        push_u32(out, value);
    }
    Ok(())
}

fn decode_config(bytes: &[u8]) -> io::Result<TransformerConfig> {
    let mut offset = 8;
    let mut values = [0u32; 7];
    for value in values.iter_mut() {
        *value = read_u32(bytes, offset)?;
        offset += 4;
    }
    let config = TransformerConfig {
        n_layer: values[0] as usize,
        d_model: values[1] as usize,
        n_head: values[2] as usize,
        ffn_hidden: values[3] as usize,
        vocab: values[4] as usize,
        n_positions: values[5] as usize,
        readout_rank: values[6] as usize,
    };
    config
        .validate()
        .map_err(|message| invalid(format!("invalid checkpoint configuration: {}", message)))?;
    Ok(config)
}

fn extract_manifest_digest(manifest: &str) -> Option<String> {
    // The writer emits compact JSON, so the key and value are adjacent and
    // no JSON parser is needed to recover the one field that matters here.
    let key = "\"param_sha256\":\"";
    let start = manifest.find(key)? + key.len();
    let end = manifest[start..].find('"')? + start;
    Some(manifest[start..end].to_string())
}

/// Save a checkpoint into `path`, which is treated as a directory.
///
/// The manifest records step zero. Use [`save_with_step`] to record a real
/// training step.
pub fn save(path: &Path, model: &Transformer) -> io::Result<()> {
    save_with_step(path, model, 0)
}

/// Save a checkpoint and record the training step in the manifest.
pub fn save_with_step(path: &Path, model: &Transformer, step: u32) -> io::Result<()> {
    fs::create_dir_all(path)?;

    let mut bytes = Vec::with_capacity(HEADER_LEN + model.param_count() * 4);
    bytes.extend_from_slice(&MAGIC);
    push_u32(&mut bytes, VERSION);
    encode_config(&model.config, &mut bytes)?;
    debug_assert_eq!(bytes.len(), HEADER_LEN);

    for param in model.params() {
        for value in param {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    let parameter_bytes = &bytes[HEADER_LEN..];
    let digest = sha256_hex(parameter_bytes);

    fs::write(path.join(PARAMS_FILE), &bytes)?;

    let manifest = Json::obj(vec![
        ("format", Json::str("pmc-checkpoint")),
        ("version", Json::num(VERSION as f64)),
        ("config", config_json(&model.config)),
        ("param_count", Json::num(model.param_count() as f64)),
        ("step", Json::num(step as f64)),
        ("param_sha256", Json::str(digest)),
        ("file", Json::str(PARAMS_FILE)),
    ])
    .to_string();
    fs::write(path.join(MANIFEST_FILE), manifest.as_bytes())?;
    Ok(())
}

/// Load a checkpoint directory written by [`save`] or [`save_with_step`].
pub fn load(path: &Path) -> io::Result<Transformer> {
    let bytes = fs::read(path.join(PARAMS_FILE))?;
    if bytes.len() < HEADER_LEN {
        return Err(invalid("checkpoint is shorter than its header"));
    }
    if bytes[0..4] != MAGIC {
        return Err(invalid("checkpoint magic is not PMCC"));
    }
    let version = read_u32(&bytes, 4)?;
    if version != VERSION {
        return Err(invalid(format!(
            "unsupported checkpoint version {} (expected {})",
            version, VERSION
        )));
    }

    let config = decode_config(&bytes)?;
    let expected = config.param_count();
    let expected_len = HEADER_LEN
        .checked_add(expected.checked_mul(4).ok_or_else(|| invalid("parameter count overflow"))?)
        .ok_or_else(|| invalid("parameter count overflow"))?;
    if bytes.len() != expected_len {
        return Err(invalid(format!(
            "checkpoint has {} bytes, expected {} for {} parameters",
            bytes.len(),
            expected_len,
            expected
        )));
    }

    let parameter_bytes = &bytes[HEADER_LEN..];
    let digest = sha256_hex(parameter_bytes);
    let manifest = fs::read_to_string(path.join(MANIFEST_FILE))?;
    let recorded = extract_manifest_digest(&manifest)
        .ok_or_else(|| invalid("manifest does not contain a parameter digest"))?;
    if recorded != digest {
        return Err(invalid(format!(
            "parameter digest mismatch: manifest {}, file {}",
            recorded, digest
        )));
    }

    let mut model = Transformer::zeros(config);
    let mut offset = HEADER_LEN;
    for param in model.params_mut() {
        for value in param.iter_mut() {
            *value = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
        }
    }
    debug_assert_eq!(offset, bytes.len());
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_directory() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("pmc-checkpoint-{}-{}", std::process::id(), nanos))
    }

    fn small_config() -> TransformerConfig {
        TransformerConfig {
            n_layer: 2,
            d_model: 8,
            n_head: 2,
            ffn_hidden: 16,
            vocab: 32,
            n_positions: 64,
            readout_rank: 4,
        }
    }

    #[test]
    fn checkpoint_roundtrip() {
        let directory = unique_directory();
        let model = small_config().init(1234);

        save(&directory, &model).expect("save should succeed");
        let manifest = fs::read_to_string(directory.join(MANIFEST_FILE)).expect("manifest");
        assert!(manifest.contains("\"param_count\":"));
        assert!(manifest.contains("\"step\":0"));
        assert!(manifest.contains("\"param_sha256\":\""));

        let loaded = load(&directory).expect("load should succeed");
        assert_eq!(loaded.config, model.config);
        let original = model.params();
        let restored = loaded.params();
        assert_eq!(original.len(), restored.len());
        for (index, (left, right)) in original.iter().zip(restored.iter()).enumerate() {
            assert_eq!(left.len(), right.len(), "parameter {} length", index);
            for (a, b) in left.iter().zip(right.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "parameter {} is not bit-identical", index);
            }
        }

        save_with_step(&directory, &model, 42).expect("save with step should succeed");
        let manifest = fs::read_to_string(directory.join(MANIFEST_FILE)).expect("manifest");
        assert!(manifest.contains("\"step\":42"));

        // A single flipped parameter byte must be rejected by the digest.
        let params_path = directory.join(PARAMS_FILE);
        let mut bytes = fs::read(&params_path).expect("checkpoint bytes");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&params_path, &bytes).expect("corrupt checkpoint");
        assert!(load(&directory).is_err(), "corruption must be detected");

        fs::remove_dir_all(&directory).expect("cleanup");
    }
}
