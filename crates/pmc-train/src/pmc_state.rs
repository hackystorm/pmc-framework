// SPDX-License-Identifier: GPL-3.0-only

//! The PMC state encoder: a trainable core that reads a frozen model's
//! hidden state and writes a memory key.
//!
//! The PMC-mini bench (v1, v2, v3) measured a hard limit of the frozen
//! mechanism: a *linear* interface per host over a *fixed* random projection
//! cannot align two substrates. Same-host retrieval is perfect while the
//! cross-host retrieval stays near chance for a third of the training cues,
//! even in sample. This module implements the other mechanism, pre-registered
//! before any of it was measured:
//!
//! ```text
//! y = W_host x + b_host            x is the L2-normalised hidden state
//! u = W_expand y + b_expand        P wide in, H wide out, shared
//! h = u
//! h = h + SwiGLU(Wg h + bg, Wu h + bu)    (per layer, shared by every host)
//! z = W_out h + b_out
//! key = z / ||z||
//! ```
//!
//! The input projection is trainable and per host, because the substrates
//! differ; the expansion, the body and the head are shared, because the
//! memory is one. The loss is InfoNCE over the views of one fact, so a key
//! written from one substrate is trained to sit next to the query the other
//! substrate produces for the same fact.
//!
//! No external dependencies: the crate stays std-only, like the rest of the
//! trainer. The loss and the gradient are a port of the bench oracle in
//! `bench/substrate_swap.py`, so a failing port is a test failure rather than
//! a silent divergence.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use pmc_core::{sha256, sha256_hex, Rng};

use crate::ops::{adam_step, grad_clip, swiglu, swiglu_backward};

/// Magic bytes of the views file.
pub const VIEWS_MAGIC: [u8; 4] = *b"PMCV";
/// Version of the views file.
pub const VIEWS_VERSION: u32 = 1;
/// Magic bytes of the weights file.
pub const WEIGHTS_MAGIC: [u8; 4] = *b"PMCW";
/// Version of the weights file.
pub const WEIGHTS_VERSION: u32 = 1;

/// A view of a fact statement, written by a substrate.
pub const KIND_FACT: u8 = 0;
/// A view of a question about a fact, written by a substrate.
pub const KIND_QUESTION: u8 = 1;

/// The host whose fact view writes the memory keys.
pub const KEY_HOST: usize = 0;
/// The host whose fact view is the cross-host counter-part.
pub const QUERY_HOST: usize = 1;

const DIGEST_BYTES: usize = 32;
/// magic, version, projection, hidden, layers, key, host count.
const HEADER: usize = 28;
/// A ceiling on a host dimension, so a corrupted header cannot ask for a
/// multi-gigabyte allocation.
const MAX_HOST_DIM: usize = 1 << 20;

/// Architecture and training description of the state encoder.
///
/// The host dimensions live in the data, not here: one encoder serves any
/// number of substrates, each with its own input projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PmcStateConfig {
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub temperature: f32,
    /// Width of the per-host projection output.
    pub projection_dim: usize,
    /// Width of the shared residual body.
    pub hidden_dim: usize,
    pub layers: usize,
    /// Width of the memory key.
    pub key_dim: usize,
    pub max_grad_norm: f32,
}

impl Default for PmcStateConfig {
    fn default() -> Self {
        PmcStateConfig {
            seed: 20261031,
            epochs: 300,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            temperature: 0.1,
            projection_dim: 128,
            hidden_dim: 256,
            layers: 2,
            key_dim: 64,
            max_grad_norm: 1.0,
        }
    }
}

impl PmcStateConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("the state encoder needs at least one epoch".to_string());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("the learning rate must be positive and finite".to_string());
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err("the weight decay must be finite and non-negative".to_string());
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err("the InfoNCE temperature must be positive and finite".to_string());
        }
        if self.projection_dim == 0 {
            return Err("the projection dimension must be positive".to_string());
        }
        if self.hidden_dim == 0 {
            return Err("the hidden dimension must be positive".to_string());
        }
        if self.layers == 0 {
            return Err("the body needs at least one layer".to_string());
        }
        if self.key_dim == 0 {
            return Err("the key dimension must be positive".to_string());
        }
        if !self.max_grad_norm.is_finite() || self.max_grad_norm < 0.0 {
            return Err("the gradient clip must be finite and non-negative".to_string());
        }
        Ok(())
    }

    /// Trainable parameters implied by the configuration and the host dims.
    pub fn param_count(&self, host_dims: &[usize]) -> usize {
        let p = self.projection_dim;
        let h = self.hidden_dim;
        let k = self.key_dim;
        let input: usize = host_dims.iter().map(|d| p * d + p).sum();
        let expand = h * p + h;
        let body = self.layers * (2 * h * h + 2 * h);
        let head = k * h + k;
        input + expand + body + head
    }

    /// The architecture string the weights file records, so a mismatched
    /// configuration is refused instead of silently mis-shaped.
    pub fn architecture(&self) -> String {
        format!(
            "p{}h{}l{}k{}",
            self.projection_dim, self.hidden_dim, self.layers, self.key_dim
        )
    }
}

/// One hidden-state vector, labelled by the fact it belongs to and the view
/// kind it was extracted from.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewEntry {
    pub host: usize,
    pub label: usize,
    pub kind: u8,
    pub data: Vec<f32>,
}

impl ViewEntry {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.host as u32).to_le_bytes());
        out.extend_from_slice(&(self.label as u32).to_le_bytes());
        out.push(self.kind);
        out.push(0);
        out.push(0);
        out.push(0);
        out.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Read a `PMCV` views file.
pub fn read_views(path: &Path) -> Result<Vec<ViewEntry>, String> {
    let raw = read_file(path)?;
    if raw.len() < 12 {
        return Err(format!("{} is too short to be a views file", path.display()));
    }
    if raw[0..4] != VIEWS_MAGIC[..] {
        return Err(format!("{} does not carry the views magic", path.display()));
    }
    let version = read_u32(&raw, 4);
    if version != VIEWS_VERSION {
        return Err(format!("unsupported views version {version}"));
    }
    let count = read_u32(&raw, 8) as usize;
    let mut offset = 12;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if offset + 16 > raw.len() {
            return Err("the views file is truncated in a header".to_string());
        }
        let host = read_u32(&raw, offset) as usize;
        let label = read_u32(&raw, offset + 4) as usize;
        let kind = raw[offset + 8];
        let dim = read_u32(&raw, offset + 12) as usize;
        offset += 16;
        let bytes = dim * 4;
        if offset + bytes > raw.len() {
            return Err("the views file is truncated in a vector".to_string());
        }
        let mut data = Vec::with_capacity(dim);
        for index in 0..dim {
            let start = offset + index * 4;
            data.push(f32::from_le_bytes([
                raw[start],
                raw[start + 1],
                raw[start + 2],
                raw[start + 3],
            ]));
        }
        offset += bytes;
        if host > 1 {
            return Err(format!(
                "the views file names host {host}, only two protocols are defined"
            ));
        }
        if kind != KIND_FACT && kind != KIND_QUESTION {
            return Err(format!("unknown view kind {kind}"));
        }
        entries.push(ViewEntry {
            host,
            label,
            kind,
            data,
        });
    }
    if offset != raw.len() {
        return Err("the views file has trailing bytes".to_string());
    }
    Ok(entries)
}

/// Write a `PMCV` views file.
pub fn write_views(path: &Path, entries: &[ViewEntry]) -> Result<(), String> {
    let mut raw = Vec::new();
    raw.extend_from_slice(&VIEWS_MAGIC[..]);
    raw.extend_from_slice(&VIEWS_VERSION.to_le_bytes());
    raw.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        entry.encode(&mut raw);
    }
    write_file(path, &raw)
}

/// The dimension of every host present in the views, in host order.
pub fn host_dimensions(entries: &[ViewEntry]) -> Result<Vec<usize>, String> {
    let mut found: Vec<Option<usize>> = Vec::new();
    for entry in entries {
        while found.len() <= entry.host {
            found.push(None);
        }
        match &mut found[entry.host] {
            None => found[entry.host] = Some(entry.data.len()),
            Some(existing) => {
                if *existing != entry.data.len() {
                    return Err(format!(
                        "host {} mixes {existing} and {} dimensional views",
                        entry.host,
                        entry.data.len()
                    ));
                }
            }
        }
    }
    if found.len() < 2 {
        return Err(format!(
            "the views file names {} host, the protocol needs two",
            found.len()
        ));
    }
    let mut out = Vec::with_capacity(found.len());
    for (host, dim) in found.iter().enumerate() {
        match dim {
            Some(dim) => {
                if *dim == 0 {
                    return Err(format!("host {host} has an empty view"));
                }
                out.push(*dim);
            }
            None => return Err(format!("the views file has no view for host {host}")),
        }
    }
    Ok(out)
}

/// Where each parameter group starts in a flattened parameter list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SliceLayout {
    input: usize,
    input_bias: usize,
    expand: usize,
    expand_bias: usize,
    gate: usize,
    gate_bias: usize,
    up: usize,
    up_bias: usize,
    head: usize,
    head_bias: usize,
}

impl SliceLayout {
    fn new(hosts: usize, layers: usize) -> SliceLayout {
        let input = 0;
        let input_bias = hosts;
        let expand = 2 * hosts;
        let expand_bias = expand + 1;
        let gate = expand_bias + 1;
        let gate_bias = gate + layers;
        let up = gate_bias + layers;
        let up_bias = up + layers;
        let head = up_bias + layers;
        let head_bias = head + 1;
        SliceLayout {
            input,
            input_bias,
            expand,
            expand_bias,
            gate,
            gate_bias,
            up,
            up_bias,
            head,
            head_bias,
        }
    }

    fn slices(&self) -> usize {
        self.head_bias + 1
    }

    /// The indices of the slices that carry weight decay: every matrix, no
    /// bias. The bench applies the same rule to the linear adapter.
    fn weight_slices(&self, hosts: usize, layers: usize) -> Vec<usize> {
        let mut out: Vec<usize> = (self.input..self.input + hosts).collect();
        out.push(self.expand);
        out.extend(self.gate..self.gate + layers);
        out.extend(self.up..self.up + layers);
        out.push(self.head);
        out.sort();
        out
    }
}

/// Cached activations of one layer, used by the backward pass.
#[derive(Clone, Debug)]
struct LayerCache {
    /// The state entering the layer.
    h_in: Vec<f32>,
    /// The state leaving the layer: `h_in + SwiGLU(gate, up)`.
    h_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ForwardCache {
    host: usize,
    /// The L2-normalised hidden state that entered the projection.
    xn: Vec<f32>,
    /// The projection output, P wide.
    y: Vec<f32>,
    layers: Vec<LayerCache>,
    z: Vec<f32>,
    norm: f32,
}

impl ForwardCache {
    /// The state the head reads.
    fn final_state(&self) -> &[f32] {
        match self.layers.last() {
            Some(layer) => layer.h_out.as_slice(),
            None => self.y.as_slice(),
        }
    }
}

/// The state encoder.
#[derive(Clone, Debug)]
pub struct PmcStateModel {
    pub config: PmcStateConfig,
    pub host_dims: Vec<usize>,
    /// Per host: `[projection_dim, host_dim]`, row major.
    pub input: Vec<Vec<f32>>,
    pub input_bias: Vec<Vec<f32>>,
    /// Shared `[hidden_dim, projection_dim]`, row major.
    pub expand: Vec<f32>,
    pub expand_bias: Vec<f32>,
    /// Per layer `[hidden_dim, hidden_dim]`, row major.
    pub gate: Vec<Vec<f32>>,
    pub gate_bias: Vec<Vec<f32>>,
    /// Per layer `[hidden_dim, hidden_dim]`, row major.
    pub up: Vec<Vec<f32>>,
    pub up_bias: Vec<Vec<f32>>,
    /// `[key_dim, hidden_dim]`, row major.
    pub head: Vec<f32>,
    pub head_bias: Vec<f32>,
}

impl PmcStateModel {
    pub fn init(config: PmcStateConfig, host_dims: &[usize]) -> Result<PmcStateModel, String> {
        config.validate()?;
        if host_dims.is_empty() {
            return Err("the state encoder needs at least one host".to_string());
        }
        if host_dims.len() > 2 {
            return Err("the views protocol defines two hosts, not more".to_string());
        }
        for dim in host_dims {
            if *dim == 0 {
                return Err("a host dimension must be positive".to_string());
            }
        }
        let mut rng = Rng::new(config.seed);
        let p = config.projection_dim;
        let h = config.hidden_dim;
        let k = config.key_dim;
        let body_scale = 1.0 / (h as f32).sqrt();

        let mut input = Vec::with_capacity(host_dims.len());
        let mut input_bias = Vec::with_capacity(host_dims.len());
        for dim in host_dims {
            let scale = 1.0 / (*dim as f32).sqrt();
            let mut matrix = Vec::with_capacity(p * dim);
            for _ in 0..p * dim {
                matrix.push(rng.normal() * scale);
            }
            input.push(matrix);
            input_bias.push(vec![0.0; p]);
        }

        let mut expand = Vec::with_capacity(h * p);
        for _ in 0..h * p {
            expand.push(rng.normal() * body_scale);
        }
        let expand_bias = vec![0.0; h];

        let mut gate = Vec::with_capacity(config.layers);
        let mut gate_bias = Vec::with_capacity(config.layers);
        let mut up = Vec::with_capacity(config.layers);
        let mut up_bias = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let mut gate_matrix = Vec::with_capacity(h * h);
            for _ in 0..h * h {
                gate_matrix.push(rng.normal() * body_scale);
            }
            let mut up_matrix = Vec::with_capacity(h * h);
            for _ in 0..h * h {
                up_matrix.push(rng.normal() * body_scale);
            }
            gate.push(gate_matrix);
            gate_bias.push(vec![0.0; h]);
            up.push(up_matrix);
            up_bias.push(vec![0.0; h]);
        }

        let mut head = Vec::with_capacity(k * h);
        for _ in 0..k * h {
            head.push(rng.normal() * body_scale);
        }
        let head_bias = vec![0.0; k];

        Ok(PmcStateModel {
            config,
            host_dims: host_dims.to_vec(),
            input,
            input_bias,
            expand,
            expand_bias,
            gate,
            gate_bias,
            up,
            up_bias,
            head,
            head_bias,
        })
    }

    fn layout(&self) -> SliceLayout {
        SliceLayout::new(self.host_dims.len(), self.config.layers)
    }

    /// Parameter slices, in the order [`Grads::slices`] mirrors.
    pub fn params(&self) -> Vec<&[f32]> {
        let mut out: Vec<&[f32]> = Vec::with_capacity(self.layout().slices());
        for host in &self.input {
            out.push(host.as_slice());
        }
        for host in &self.input_bias {
            out.push(host.as_slice());
        }
        out.push(self.expand.as_slice());
        out.push(self.expand_bias.as_slice());
        for layer in &self.gate {
            out.push(layer.as_slice());
        }
        for layer in &self.gate_bias {
            out.push(layer.as_slice());
        }
        for layer in &self.up {
            out.push(layer.as_slice());
        }
        for layer in &self.up_bias {
            out.push(layer.as_slice());
        }
        out.push(self.head.as_slice());
        out.push(self.head_bias.as_slice());
        out
    }

    pub fn params_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out: Vec<&mut [f32]> = Vec::new();
        for host in &mut self.input {
            out.push(host.as_mut_slice());
        }
        for host in &mut self.input_bias {
            out.push(host.as_mut_slice());
        }
        out.push(self.expand.as_mut_slice());
        out.push(self.expand_bias.as_mut_slice());
        for layer in &mut self.gate {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.gate_bias {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.up {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.up_bias {
            out.push(layer.as_mut_slice());
        }
        out.push(self.head.as_mut_slice());
        out.push(self.head_bias.as_mut_slice());
        out
    }

    /// Digest of the parameter bytes, so a run can be reproduced.
    pub fn weight_digest(&self) -> String {
        let mut raw: Vec<u8> = Vec::new();
        for value in self.params().into_iter().flatten() {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        sha256_hex(&raw)
    }

    pub fn param_count(&self) -> usize {
        self.params().iter().map(|slice| slice.len()).sum()
    }

    /// The memory key a view produces.
    pub fn key(&self, host: usize, x: &[f32]) -> Result<Vec<f32>, String> {
        Ok(self.forward(host, x)?.z)
    }

    fn forward(&self, host: usize, x: &[f32]) -> Result<ForwardCache, String> {
        if host >= self.host_dims.len() {
            return Err(format!("host {host} is outside the encoder"));
        }
        let dim = self.host_dims[host];
        if x.len() != dim {
            return Err(format!(
                "host {host} expects {dim} values, got {}",
                x.len()
            ));
        }
        let xn = normalise(x)?;
        let p = self.config.projection_dim;
        let h = self.config.hidden_dim;
        let y = matvec(&self.input[host], &self.input_bias[host], &xn, p);
        let u = matvec(&self.expand, &self.expand_bias, &y, h);

        let mut state = u;
        let mut layers = Vec::with_capacity(self.config.layers);
        for index in 0..self.config.layers {
            let gate = matvec(&self.gate[index], &self.gate_bias[index], &state, h);
            let up = matvec(&self.up[index], &self.up_bias[index], &state, h);
            let mut out = vec![0.0f32; h];
            swiglu(&gate, &up, &mut out);
            let mut h_out = state.clone();
            for (cell, value) in h_out.iter_mut().zip(out.iter()) {
                *cell += *value;
            }
            let h_in = std::mem::replace(&mut state, h_out.clone());
            layers.push(LayerCache {
                h_in,
                h_out,
                gate,
                up,
            });
        }

        let z_raw = matvec(&self.head, &self.head_bias, &state, self.config.key_dim);
        let norm = z_raw.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("the encoder produced a zero or non-finite key".to_string());
        }
        let z = z_raw.iter().map(|value| value / norm).collect();
        Ok(ForwardCache {
            host,
            xn,
            y,
            layers,
            z,
            norm,
        })
    }

    fn backward(&self, cache: &ForwardCache, dz: &[f32]) -> Result<Grads, String> {
        let h = self.config.hidden_dim;
        let k = self.config.key_dim;
        if dz.len() != k {
            return Err(format!("the key gradient must have {k} values"));
        }
        let p = self.config.projection_dim;
        let mut grads = Grads::new(&self.host_dims, &self.config);

        // z = z_raw / ||z_raw||, so the normalisation gradient is applied
        // first, exactly as the bench does it.
        let mut dz_raw = vec![0.0f32; k];
        let z_dot: f32 = cache
            .z
            .iter()
            .zip(dz.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        for index in 0..k {
            dz_raw[index] = (dz[index] - cache.z[index] * z_dot) / cache.norm;
        }

        let mut dh = vec![0.0f32; h];
        {
            let final_state = cache.final_state();
            for row in 0..k {
                grads.head_bias[row] += dz_raw[row];
                for j in 0..h {
                    grads.head[row * h + j] += dz_raw[row] * final_state[j];
                    dh[j] += dz_raw[row] * self.head[row * h + j];
                }
            }
        }

        for index in (0..cache.layers.len()).rev() {
            let layer = &cache.layers[index];
            let mut dgate = vec![0.0f32; h];
            let mut dup = vec![0.0f32; h];
            swiglu_backward(&layer.gate, &layer.up, &dh, &mut dgate, &mut dup);
            let mut dh_in = dh.clone();
            for row in 0..h {
                grads.gate_bias[index][row] += dgate[row];
                grads.up_bias[index][row] += dup[row];
                let gate_source = &self.gate[index][row * h..(row + 1) * h];
                let up_source = &self.up[index][row * h..(row + 1) * h];
                for j in 0..h {
                    grads.gate[index][row * h + j] += dgate[row] * layer.h_in[j];
                    grads.up[index][row * h + j] += dup[row] * layer.h_in[j];
                    dh_in[j] += dgate[row] * gate_source[j] + dup[row] * up_source[j];
                }
            }
            dh = dh_in;
        }

        // The state the last backward pass produced is the gradient of the
        // expansion output; expand it back down to the projection width.
        let mut dy = vec![0.0f32; p];
        for row in 0..h {
            grads.expand_bias[row] += dh[row];
            for j in 0..p {
                grads.expand[row * p + j] += dh[row] * cache.y[j];
                dy[j] += dh[row] * self.expand[row * p + j];
            }
        }

        let dim = self.host_dims[cache.host];
        for row in 0..p {
            grads.input_bias[cache.host][row] += dy[row];
            let target = &mut grads.input[cache.host][row * dim..(row + 1) * dim];
            for (cell, value) in target.iter_mut().zip(cache.xn.iter()) {
                *cell += dy[row] * value;
            }
        }
        Ok(grads)
    }
}

/// Gradients, in the same order as [`PmcStateModel::params`].
#[derive(Clone, Debug)]
pub struct Grads {
    input: Vec<Vec<f32>>,
    input_bias: Vec<Vec<f32>>,
    expand: Vec<f32>,
    expand_bias: Vec<f32>,
    gate: Vec<Vec<f32>>,
    gate_bias: Vec<Vec<f32>>,
    up: Vec<Vec<f32>>,
    up_bias: Vec<Vec<f32>>,
    head: Vec<f32>,
    head_bias: Vec<f32>,
}

impl Grads {
    fn new(host_dims: &[usize], config: &PmcStateConfig) -> Grads {
        let p = config.projection_dim;
        let h = config.hidden_dim;
        let k = config.key_dim;
        let mut input = Vec::with_capacity(host_dims.len());
        let mut input_bias = Vec::with_capacity(host_dims.len());
        for dim in host_dims {
            input.push(vec![0.0f32; p * dim]);
            input_bias.push(vec![0.0f32; p]);
        }
        Grads {
            input,
            input_bias,
            expand: vec![0.0f32; h * p],
            expand_bias: vec![0.0f32; h],
            gate: (0..config.layers).map(|_| vec![0.0f32; h * h]).collect(),
            gate_bias: (0..config.layers).map(|_| vec![0.0f32; h]).collect(),
            up: (0..config.layers).map(|_| vec![0.0f32; h * h]).collect(),
            up_bias: (0..config.layers).map(|_| vec![0.0f32; h]).collect(),
            head: vec![0.0f32; k * h],
            head_bias: vec![0.0f32; k],
        }
    }

    pub fn slices(&self) -> Vec<&[f32]> {
        let mut out: Vec<&[f32]> = Vec::new();
        for host in &self.input {
            out.push(host.as_slice());
        }
        for host in &self.input_bias {
            out.push(host.as_slice());
        }
        out.push(self.expand.as_slice());
        out.push(self.expand_bias.as_slice());
        for layer in &self.gate {
            out.push(layer.as_slice());
        }
        for layer in &self.gate_bias {
            out.push(layer.as_slice());
        }
        for layer in &self.up {
            out.push(layer.as_slice());
        }
        for layer in &self.up_bias {
            out.push(layer.as_slice());
        }
        out.push(self.head.as_slice());
        out.push(self.head_bias.as_slice());
        out
    }

    fn slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut out: Vec<&mut [f32]> = Vec::new();
        for host in &mut self.input {
            out.push(host.as_mut_slice());
        }
        for host in &mut self.input_bias {
            out.push(host.as_mut_slice());
        }
        out.push(self.expand.as_mut_slice());
        out.push(self.expand_bias.as_mut_slice());
        for layer in &mut self.gate {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.gate_bias {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.up {
            out.push(layer.as_mut_slice());
        }
        for layer in &mut self.up_bias {
            out.push(layer.as_mut_slice());
        }
        out.push(self.head.as_mut_slice());
        out.push(self.head_bias.as_mut_slice());
        out
    }

    fn add(&mut self, other: &Grads) {
        for (target, source) in self.input.iter_mut().zip(other.input.iter()) {
            for (cell, value) in target.iter_mut().zip(source.iter()) {
                *cell += *value;
            }
        }
        for (target, source) in self.input_bias.iter_mut().zip(other.input_bias.iter()) {
            for (cell, value) in target.iter_mut().zip(source.iter()) {
                *cell += *value;
            }
        }
        for (cell, value) in self.expand.iter_mut().zip(other.expand.iter()) {
            *cell += *value;
        }
        for (cell, value) in self.expand_bias.iter_mut().zip(other.expand_bias.iter()) {
            *cell += *value;
        }
        for index in 0..self.gate.len().min(other.gate.len()) {
            for (cell, value) in self.gate[index].iter_mut().zip(other.gate[index].iter()) {
                *cell += *value;
            }
            for (cell, value) in self.gate_bias[index]
                .iter_mut()
                .zip(other.gate_bias[index].iter())
            {
                *cell += *value;
            }
            for (cell, value) in self.up[index].iter_mut().zip(other.up[index].iter()) {
                *cell += *value;
            }
            for (cell, value) in self.up_bias[index]
                .iter_mut()
                .zip(other.up_bias[index].iter())
            {
                *cell += *value;
            }
        }
        for (cell, value) in self.head.iter_mut().zip(other.head.iter()) {
            *cell += *value;
        }
        for (cell, value) in self.head_bias.iter_mut().zip(other.head_bias.iter()) {
            *cell += *value;
        }
    }
}

/// InfoNCE over a batch of keys, with the views of one fact as positives.
///
/// This is a port of `infonce_loss_and_vector_grad` in
/// `bench/substrate_swap.py`: the same stable softmax, the same loss, the
/// same symmetrised gradient.
pub fn infonce_loss_and_grad(
    z: &[Vec<f32>],
    labels: &[usize],
    temperature: f32,
) -> Result<(f32, Vec<Vec<f32>>), String> {
    let count = z.len();
    if count < 3 {
        return Err("InfoNCE needs at least three views".to_string());
    }
    if labels.len() != count {
        return Err("InfoNCE labels do not match the views".to_string());
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("InfoNCE temperature must be positive".to_string());
    }
    let width = z[0].len();
    let mut scores = vec![vec![0.0f32; count]; count];
    for i in 0..count {
        for j in 0..count {
            scores[i][j] = dot(&z[i], &z[j]) / temperature;
        }
    }
    let mut scalar = vec![vec![0.0f32; count]; count];
    let mut loss = 0.0f32;
    for anchor in 0..count {
        let positives: Vec<usize> = (0..count)
            .filter(|index| *index != anchor && labels[*index] == labels[anchor])
            .collect();
        if positives.is_empty() {
            return Err("InfoNCE anchor has no positive view".to_string());
        }
        let negatives: Vec<usize> = (0..count).filter(|index| *index != anchor).collect();
        let mut max_all = f32::NEG_INFINITY;
        for index in &negatives {
            if scores[anchor][*index] > max_all {
                max_all = scores[anchor][*index];
            }
        }
        let mut exp_all = vec![0.0f32; count];
        let mut sum_all = 0.0f32;
        for index in &negatives {
            exp_all[*index] = (scores[anchor][*index] - max_all).exp();
            sum_all += exp_all[*index];
        }
        let mut max_pos = f32::NEG_INFINITY;
        for index in &positives {
            if scores[anchor][*index] > max_pos {
                max_pos = scores[anchor][*index];
            }
        }
        let mut exp_pos = vec![0.0f32; count];
        let mut sum_pos = 0.0f32;
        for index in &positives {
            exp_pos[*index] = (scores[anchor][*index] - max_pos).exp();
            sum_pos += exp_pos[*index];
        }
        if !(sum_all > 0.0) || !(sum_pos > 0.0) {
            return Err("the InfoNCE softmax collapsed".to_string());
        }
        loss += -(sum_pos.ln() + max_pos) + (sum_all.ln() + max_all);
        for index in &negatives {
            let p_all = exp_all[*index] / sum_all;
            let p_pos = if exp_pos[*index] > 0.0 {
                exp_pos[*index] / sum_pos
            } else {
                0.0
            };
            scalar[anchor][*index] = p_all - p_pos;
        }
    }
    loss /= count as f32;
    let mut vector = vec![vec![0.0f32; width]; count];
    for i in 0..count {
        for j in 0..count {
            let coefficient = (scalar[i][j] + scalar[j][i]) / (temperature * count as f32);
            if coefficient == 0.0 {
                continue;
            }
            for k in 0..width {
                vector[i][k] += coefficient * z[j][k];
            }
        }
    }
    Ok((loss, vector))
}

/// The outcome of one training run.
#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub config: PmcStateConfig,
    pub host_dims: Vec<usize>,
    pub param_count: usize,
    pub weight_digest: String,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub epochs: usize,
    pub retrieval: RetrievalReport,
    pub folds: Vec<FoldReport>,
    pub elapsed_millis: u128,
}

/// Retrieval counts over one set of labels.
///
/// The keys are the host-A fact views; the queries are the four view kinds.
/// `host_b_question` is the cross-host, cross-view number the whole
/// mechanism exists to move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetrievalReport {
    pub labels: usize,
    pub host_a_fact: usize,
    pub host_a_question: usize,
    pub host_b_fact: usize,
    pub host_b_question: usize,
}

impl RetrievalReport {
    fn counted(rows: &[RetrievalRow]) -> RetrievalReport {
        let mut counts = RetrievalReport::default();
        for row in rows {
            if row.host_a_fact {
                counts.host_a_fact += 1;
            }
            if row.host_a_question {
                counts.host_a_question += 1;
            }
            if row.host_b_fact {
                counts.host_b_fact += 1;
            }
            if row.host_b_question {
                counts.host_b_question += 1;
            }
        }
        counts.labels = rows.len();
        counts
    }
}

/// One per-label retrieval row.
#[derive(Clone, Copy, Debug)]
struct RetrievalRow {
    label: usize,
    host_a_fact: bool,
    host_a_question: bool,
    host_b_fact: bool,
    host_b_question: bool,
}

/// One cross-validation fold.
#[derive(Clone, Debug)]
pub struct FoldReport {
    pub fold: usize,
    pub held_out: Vec<usize>,
    pub counts: RetrievalReport,
    pub final_loss: f32,
}

/// The views of a label set, in the order the trainer needs them.
fn select(entries: &[ViewEntry], labels: &[usize]) -> Vec<ViewEntry> {
    entries
        .iter()
        .filter(|entry| labels.contains(&entry.label))
        .cloned()
        .collect()
}

fn unique_labels(entries: &[ViewEntry]) -> Vec<usize> {
    let mut all: Vec<usize> = entries.iter().map(|entry| entry.label).collect();
    all.sort();
    all.dedup();
    all
}

/// Train the encoder on a label set and report what the run learned.
///
/// The labels in `train_labels` are the ones the loss sees; the retrieval is
/// measured over every label present in the views, so the numbers stay
/// comparable across folds.
pub fn train(
    config: &PmcStateConfig,
    entries: &[ViewEntry],
    train_labels: &[usize],
    folds: usize,
) -> Result<(PmcStateModel, TrainOutcome), String> {
    config.validate()?;
    let host_dims = host_dimensions(entries)?;
    let mut model = PmcStateModel::init(*config, &host_dims)?;
    let wanted = {
        let mut sorted = train_labels.to_vec();
        sorted.sort();
        sorted.dedup();
        sorted
    };
    let chosen = select(entries, &wanted);
    if chosen.is_empty() {
        return Err("no view matches the training labels".to_string());
    }
    for label in &wanted {
        let views = chosen
            .iter()
            .filter(|entry| entry.label == *label)
            .count();
        if views < 2 {
            return Err(format!(
                "label {label} has {views} view, InfoNCE needs at least two"
            ));
        }
    }

    let layout = model.layout();
    let mut moments: Vec<Vec<f32>> = model.params().iter().map(|s| vec![0.0; s.len()]).collect();
    let mut second: Vec<Vec<f32>> = model.params().iter().map(|s| vec![0.0; s.len()]).collect();
    let mut initial_loss = f32::NAN;
    let mut final_loss = f32::NAN;
    let started = std::time::Instant::now();

    for epoch in 1..=config.epochs {
        let mut caches = Vec::with_capacity(chosen.len());
        let mut keys = Vec::with_capacity(chosen.len());
        let mut labels = Vec::with_capacity(chosen.len());
        for entry in &chosen {
            let cache = model.forward(entry.host, &entry.data)?;
            keys.push(cache.z.clone());
            labels.push(entry.label);
            caches.push(cache);
        }
        let (loss, vector) = infonce_loss_and_grad(&keys, &labels, config.temperature)?;
        if epoch == 1 {
            initial_loss = loss;
        }
        final_loss = loss;
        if !loss.is_finite() {
            return Err("the training loss became non-finite".to_string());
        }

        let mut grads = Grads::new(&host_dims, config);
        for (index, cache) in caches.iter().enumerate() {
            // `backward` applies the L2 normalisation gradient itself, so the
            // caller hands over the raw key gradient from InfoNCE. Applying
            // it here as well would normalise the gradient twice.
            grads.add(&model.backward(cache, &vector[index])?);
        }

        // Weight decay on the matrices, then clipping, then Adam: the same
        // order the bench uses once the decay is factored out.
        let mut params = model.params_mut();
        let mut grad_slices = grads.slices_mut();
        let decay = config.weight_decay;
        if decay > 0.0 {
            for index in layout.weight_slices(host_dims.len(), config.layers) {
                for (cell, param) in grad_slices[index].iter_mut().zip(params[index].iter()) {
                    *cell += decay * param;
                }
            }
        }
        if config.max_grad_norm > 0.0 {
            grad_clip(&mut grad_slices, config.max_grad_norm);
        }
        for index in 0..params.len() {
            adam_step(
                params[index],
                grad_slices[index],
                &mut moments[index],
                &mut second[index],
                epoch as u32,
                config.learning_rate,
                0.9,
                0.999,
                1e-8,
            );
        }
    }

    let retrieval = RetrievalReport::counted(&retrieval_rows(&model, entries, &unique_labels(entries))?);
    let fold_reports = if folds >= 2 {
        cross_validate(config, entries, folds)?
    } else {
        Vec::new()
    };
    let param_count = model.param_count();
    let weight_digest = model.weight_digest();
    let elapsed = started.elapsed().as_millis();

    Ok((
        model,
        TrainOutcome {
            config: *config,
            host_dims,
            param_count,
            weight_digest,
            initial_loss,
            final_loss,
            epochs: config.epochs,
            retrieval,
            folds: fold_reports,
            elapsed_millis: elapsed,
        },
    ))
}

/// One retrieval row per label.
fn retrieval_rows(
    model: &PmcStateModel,
    entries: &[ViewEntry],
    labels: &[usize],
) -> Result<Vec<RetrievalRow>, String> {
    let mut index: HashMap<(usize, u8, usize), usize> = HashMap::new();
    for (position, entry) in entries.iter().enumerate() {
        index.insert((entry.host, entry.kind, entry.label), position);
    }
    let lookup = |host: usize, kind: u8, label: usize| -> Result<&ViewEntry, String> {
        match index.get(&(host, kind, label)) {
            Some(position) => Ok(&entries[*position]),
            None => Err(format!(
                "no host {host} view of kind {kind} for label {label}"
            )),
        }
    };
    let mut keys: Vec<(usize, Vec<f32>)> = Vec::with_capacity(labels.len());
    for label in labels {
        let entry = lookup(KEY_HOST, KIND_FACT, *label)?;
        keys.push((*label, model.key(entry.host, &entry.data)?));
    }
    let mut rows = Vec::with_capacity(labels.len());
    for label in labels {
        let query = |host: usize, kind: u8| -> Result<usize, String> {
            let entry = lookup(host, kind, *label)?;
            let key = model.key(entry.host, &entry.data)?;
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (position, (_, candidate)) in keys.iter().enumerate() {
                let score = dot(&key, candidate);
                if score > best_score {
                    best_score = score;
                    best = position;
                }
            }
            Ok(keys[best].0)
        };
        rows.push(RetrievalRow {
            label: *label,
            host_a_fact: query(KEY_HOST, KIND_FACT)? == *label,
            host_a_question: query(KEY_HOST, KIND_QUESTION)? == *label,
            host_b_fact: query(QUERY_HOST, KIND_FACT)? == *label,
            host_b_question: query(QUERY_HOST, KIND_QUESTION)? == *label,
        });
    }
    Ok(rows)
}

/// The label sets a cross-validation holds out, in fold order.
pub fn cv_folds(labels: &[usize], folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut shuffled = labels.to_vec();
    let mut rng = Rng::new(seed ^ 0x5EED_C0FFEE);
    for index in (1..shuffled.len()).rev() {
        let swap = ((rng.next_f32() * (index as f32 + 1.0)) as usize).min(index);
        shuffled.swap(index, swap);
    }
    let folds = folds.max(1).min(shuffled.len().max(1));
    let mut out = Vec::with_capacity(folds);
    for fold in 0..folds {
        let start = fold * shuffled.len() / folds;
        let end = (fold + 1) * shuffled.len() / folds;
        if start == end {
            continue;
        }
        out.push(shuffled[start..end].to_vec());
    }
    out
}

/// Cross-validate the encoder: train on every fold but one, then report the
/// retrieval over all the labels, counting only the held-out ones.
pub fn cross_validate(
    config: &PmcStateConfig,
    entries: &[ViewEntry],
    folds: usize,
) -> Result<Vec<FoldReport>, String> {
    let all = unique_labels(entries);
    let held_sets = cv_folds(&all, folds, config.seed);
    let mut reports = Vec::with_capacity(held_sets.len());
    for (fold, held) in held_sets.iter().enumerate() {
        if held.is_empty() {
            continue;
        }
        let training: Vec<usize> = all
            .iter()
            .filter(|label| !held.contains(label))
            .cloned()
            .collect();
        if training.is_empty() {
            return Err("a cross-validation fold holds out every label".to_string());
        }
        let (model, outcome) = train(config, entries, &training, 0)?;
        let rows = retrieval_rows(&model, entries, &all)?;
        let kept: Vec<RetrievalRow> = rows
            .into_iter()
            .filter(|row| held.contains(&row.label))
            .collect();
        reports.push(FoldReport {
            fold,
            held_out: held.clone(),
            counts: RetrievalReport::counted(&kept),
            final_loss: outcome.final_loss,
        });
    }
    Ok(reports)
}

/// Write a `PMCW` weights file.
pub fn write_weights(path: &Path, model: &PmcStateModel) -> Result<(), String> {
    let mut raw = Vec::new();
    raw.extend_from_slice(&WEIGHTS_MAGIC[..]);
    raw.extend_from_slice(&WEIGHTS_VERSION.to_le_bytes());
    raw.extend_from_slice(&(model.config.projection_dim as u32).to_le_bytes());
    raw.extend_from_slice(&(model.config.hidden_dim as u32).to_le_bytes());
    raw.extend_from_slice(&(model.config.layers as u32).to_le_bytes());
    raw.extend_from_slice(&(model.config.key_dim as u32).to_le_bytes());
    raw.extend_from_slice(&(model.host_dims.len() as u32).to_le_bytes());
    for dim in &model.host_dims {
        raw.extend_from_slice(&(*dim as u32).to_le_bytes());
    }
    for slice in model.params().into_iter().flatten() {
        raw.extend_from_slice(&slice.to_le_bytes());
    }
    let digest = sha256_hex(&raw);
    raw.extend_from_slice(&sha256(digest.as_bytes()));
    write_file(path, &raw)
}

/// Read a `PMCW` weights file, refusing anything that does not match the
/// configuration the caller asked for.
pub fn read_weights(path: &Path, config: &PmcStateConfig) -> Result<PmcStateModel, String> {
    let raw = read_file(path)?;
    if raw.len() < HEADER + DIGEST_BYTES {
        return Err(format!("{} is too short to be a weights file", path.display()));
    }
    if raw[0..4] != WEIGHTS_MAGIC[..] {
        return Err(format!("{} does not carry the weights magic", path.display()));
    }
    let version = read_u32(&raw, 4);
    if version != WEIGHTS_VERSION {
        return Err(format!("unsupported weights version {version}"));
    }
    let projection = read_u32(&raw, 8) as usize;
    let hidden = read_u32(&raw, 12) as usize;
    let layers = read_u32(&raw, 16) as usize;
    let key_dim = read_u32(&raw, 20) as usize;
    let hosts = read_u32(&raw, 24) as usize;
    if projection != config.projection_dim
        || hidden != config.hidden_dim
        || layers != config.layers
        || key_dim != config.key_dim
    {
        return Err(format!(
            "the weights file describes p{projection}h{hidden}l{layers}k{key_dim}, the config describes {}",
            config.architecture()
        ));
    }
    if hosts > 2 {
        return Err(format!("the weights file names {hosts} hosts, the protocol has two"));
    }
    let mut offset = HEADER;
    let mut host_dims = Vec::with_capacity(hosts);
    for _ in 0..hosts {
        if offset + 4 > raw.len() {
            return Err("the weights file is truncated in the host dims".to_string());
        }
        host_dims.push(read_u32(&raw, offset) as usize);
        offset += 4;
    }
    for (host, dim) in host_dims.iter().enumerate() {
        if *dim == 0 || *dim > MAX_HOST_DIM {
            return Err(format!(
                "the weights file names host {host} with an implausible dimension {dim}"
            ));
        }
    }
    // The declared shape is checked against the file length before anything
    // is allocated: a corrupted header must cost a refusal, not a gigabyte of
    // RAM.
    let expected = config.param_count(&host_dims) * 4 + offset + DIGEST_BYTES;
    if expected != raw.len() {
        return Err(format!(
            "the weights file declares {} parameter bytes but is {} bytes long",
            config.param_count(&host_dims) * 4,
            raw.len() - offset - DIGEST_BYTES
        ));
    }
    let mut model = PmcStateModel::init(*config, &host_dims)?;
    let payload = raw.len() - DIGEST_BYTES;
    let mut slices = model.params_mut();
    for (position, slice) in slices.iter_mut().enumerate() {
        let bytes = slice.len() * 4;
        if offset + bytes > payload {
            return Err(format!(
                "the weights file is truncated in parameter block {position}"
            ));
        }
        for (index, value) in slice.iter_mut().enumerate() {
            let start = offset + index * 4;
            *value = f32::from_le_bytes([
                raw[start],
                raw[start + 1],
                raw[start + 2],
                raw[start + 3],
            ]);
        }
        offset += bytes;
    }
    if offset != payload {
        return Err("the weights file has trailing parameter bytes".to_string());
    }
    let digest = sha256_hex(&raw[..payload]);
    if raw[payload..] != sha256(digest.as_bytes()) {
        return Err("the weights file digest does not match its payload".to_string());
    }
    Ok(model)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
}

/// `weights` is `[rows, width]`, `bias` is `[rows]`, `x` is `[width]`.
fn matvec(weights: &[f32], bias: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let width = x.len();
    assert_eq!(
        weights.len(),
        rows * width,
        "the matrix does not match the input width"
    );
    assert_eq!(bias.len(), rows, "the bias does not match the matrix rows");
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        out.push(bias[row] + dot(&weights[row * width..(row + 1) * width], x));
    }
    out
}

fn normalise(values: &[f32]) -> Result<Vec<f32>, String> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err("cannot normalise a zero or non-finite vector".to_string());
    }
    Ok(values.iter().map(|value| value / norm).collect())
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut raw = Vec::new();
    file.read_to_end(&mut raw)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(raw)
}

fn write_file(path: &Path, raw: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
    }
    let mut file = fs::File::create(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.write_all(raw)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(())
}

fn read_u32(raw: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        raw[offset],
        raw[offset + 1],
        raw[offset + 2],
        raw[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory for the file tests, honouring `TMPDIR` so the
    /// tests run on machines where the default temporary filesystem is full
    /// and a small one is not.
    fn test_dir() -> std::path::PathBuf {
        let root = match std::env::var_os("TMPDIR") {
            Some(value) => std::path::PathBuf::from(value),
            None => std::env::temp_dir(),
        };
        root.join("pmc-state-tests")
    }

    fn small_config() -> PmcStateConfig {
        PmcStateConfig {
            seed: 7,
            epochs: 3,
            learning_rate: 0.01,
            weight_decay: 0.0001,
            temperature: 0.1,
            projection_dim: 8,
            hidden_dim: 12,
            layers: 2,
            key_dim: 5,
            max_grad_norm: 1.0,
        }
    }

    fn synthetic() -> Vec<ViewEntry> {
        // Two hosts write the same four facts in different coordinate
        // systems: host 0 uses the vector itself, host 1 a fixed rotation
        // plus noise. Only a trainable per-host projection can undo that.
        let mut rng = Rng::new(11);
        let dim = 12usize;
        let mut entries = Vec::new();
        for label in 0..4usize {
            let base: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
            let norm = base.iter().map(|value| value * value).sum::<f32>().sqrt();
            let base: Vec<f32> = base.iter().map(|value| value / norm).collect();
            let mut rotated = vec![0.0f32; dim];
            for (index, value) in base.iter().enumerate() {
                rotated[(index + 5) % dim] = *value;
            }
            for host in 0..2 {
                let data: Vec<f32> = if host == 0 {
                    base.clone()
                } else {
                    rotated
                        .iter()
                        .map(|value| value + 0.01 * rng.normal())
                        .collect()
                };
                entries.push(ViewEntry {
                    host,
                    label,
                    kind: KIND_FACT,
                    data: data.clone(),
                });
                entries.push(ViewEntry {
                    host,
                    label,
                    kind: KIND_QUESTION,
                    data,
                });
            }
        }
        entries
    }

    fn loss_of(model: &PmcStateModel, entries: &[ViewEntry], temperature: f32) -> f32 {
        let mut keys = Vec::new();
        let mut labels = Vec::new();
        for entry in entries {
            keys.push(model.forward(entry.host, &entry.data).unwrap().z.clone());
            labels.push(entry.label);
        }
        infonce_loss_and_grad(&keys, &labels, temperature).unwrap().0
    }

    #[test]
    fn the_views_file_round_trips() {
        let entries = synthetic();
        let path = test_dir().join("views.bin");
        write_views(&path, &entries).unwrap();
        assert_eq!(
            read_views(&path).unwrap(),
            entries,
            "the views file lost information"
        );
        assert!(
            std::fs::remove_file(&path).is_ok(),
            "the test did not clean up after itself"
        );
    }

    #[test]
    fn the_weights_file_round_trips() {
        let model = PmcStateModel::init(small_config(), &[12, 12]).unwrap();
        let path = test_dir().join("weights.bin");
        write_weights(&path, &model).unwrap();
        let back = read_weights(&path, &small_config()).unwrap();
        assert_eq!(
            back.weight_digest(),
            model.weight_digest(),
            "the weights file changed the parameters"
        );
        assert!(
            std::fs::remove_file(&path).is_ok(),
            "the test did not clean up after itself"
        );
    }

    #[test]
    fn a_corrupted_weights_file_is_refused() {
        let model = PmcStateModel::init(small_config(), &[12, 12]).unwrap();
        let path = test_dir().join("weights-bad.bin");
        write_weights(&path, &model).unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        let middle = raw.len() / 2;
        raw[middle] ^= 0xff;
        std::fs::write(&path, &raw).unwrap();
        assert!(
            read_weights(&path, &small_config()).is_err(),
            "a corrupted payload was still accepted"
        );
        assert!(
            std::fs::remove_file(&path).is_ok(),
            "the test did not clean up after itself"
        );
    }

    #[test]
    fn a_mismatched_configuration_is_refused() {
        let model = PmcStateModel::init(small_config(), &[12, 12]).unwrap();
        let path = test_dir().join("weights-shape.bin");
        write_weights(&path, &model).unwrap();
        let other = PmcStateConfig {
            key_dim: 6,
            ..small_config()
        };
        assert!(
            read_weights(&path, &other).is_err(),
            "weights of the wrong shape were accepted"
        );
        assert!(
            std::fs::remove_file(&path).is_ok(),
            "the test did not clean up after itself"
        );
    }

    fn analytic_grads(model: &PmcStateModel, entries: &[ViewEntry], config: &PmcStateConfig) -> Grads {
        let mut caches = Vec::new();
        let mut keys = Vec::new();
        let mut labels = Vec::new();
        for entry in entries {
            let cache = model.forward(entry.host, &entry.data).unwrap();
            keys.push(cache.z.clone());
            labels.push(entry.label);
            caches.push(cache);
        }
        let (_, vector) = infonce_loss_and_grad(&keys, &labels, config.temperature).unwrap();
        let mut analytic = Grads::new(&model.host_dims, config);
        for (index, cache) in caches.iter().enumerate() {
            analytic.add(&model.backward(cache, &vector[index]).unwrap());
        }
        analytic
    }

    #[test]
    fn the_gradients_match_finite_differences() {
        let config = small_config();
        let mut model = PmcStateModel::init(config, &[12, 12]).unwrap();
        let entries = synthetic();
        let analytic = analytic_grads(&model, &entries, &config);

        let slices = analytic.slices();
        let mut checked = 0usize;
        let mut worst = 0.0f64;
        let step = 1e-3f64;
        for (slice, gradient) in slices.iter().enumerate() {
            let stride = (gradient.len() / 3).max(1);
            for position in (0..gradient.len()).step_by(stride) {
                let value = gradient[position] as f64;
                if value == 0.0 {
                    continue;
                }
                // Each perturbation is applied and released before the loss
                // is read, so the borrow stays out of the way.
                let original = {
                    let mut params = model.params_mut();
                    let original = params[slice][position];
                    params[slice][position] = original + step as f32;
                    original
                };
                let plus = loss_of(&model, &entries, config.temperature) as f64;
                {
                    let mut params = model.params_mut();
                    params[slice][position] = original - step as f32;
                }
                let minus = loss_of(&model, &entries, config.temperature) as f64;
                {
                    let mut params = model.params_mut();
                    params[slice][position] = original;
                }
                let numeric = (plus - minus) / (2.0 * step);
                let difference = (numeric - value).abs() / value.abs().max(1.0);
                if difference > worst {
                    worst = difference;
                }
                checked += 1;
            }
        }
        assert!(checked >= 10, "too few gradients were checked: {checked}");
        assert!(
            worst < 5e-2,
            "the analytic gradient disagrees with the numeric one by {worst}"
        );
    }

    #[test]
    fn the_synthetic_hosts_align_when_trained() {
        let config = PmcStateConfig {
            epochs: 400,
            learning_rate: 0.02,
            ..small_config()
        };
        let entries = synthetic();
        let (_, outcome) = train(&config, &entries, &[0, 1, 2, 3], 0).unwrap();
        assert!(
            outcome.final_loss < outcome.initial_loss,
            "training did not lower the loss: {} -> {}",
            outcome.initial_loss,
            outcome.final_loss
        );
        assert!(
            outcome.retrieval.host_b_question >= 3,
            "the cross-host view did not align: {} of {}",
            outcome.retrieval.host_b_question,
            outcome.retrieval.labels
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_weights() {
        let config = small_config();
        let entries = synthetic();
        let (_, first) = train(&config, &entries, &[0, 1, 2, 3], 0).unwrap();
        let (_, second) = train(&config, &entries, &[0, 1, 2, 3], 0).unwrap();
        assert_eq!(first.weight_digest, second.weight_digest);
        assert_eq!(first.retrieval, second.retrieval);
    }

    #[test]
    fn a_label_with_a_missing_view_is_refused() {
        let mut entries = synthetic();
        entries.retain(|entry| !(entry.host == 1 && entry.label == 3 && entry.kind == KIND_QUESTION));
        let result = train(&small_config(), &entries, &[0, 1, 2, 3], 0);
        assert!(
            result.is_err(),
            "a label that lost a view was trained anyway"
        );
    }

    #[test]
    fn the_cross_validation_reports_every_fold() {
        let config = small_config();
        let entries = synthetic();
        let folds = cross_validate(&config, &entries, 2).unwrap();
        assert_eq!(folds.len(), 2, "the cross-validation lost a fold");
        for fold in &folds {
            assert_eq!(
                fold.counts.labels,
                fold.held_out.len(),
                "the fold counted the wrong number of held-out labels"
            );
        }
    }

    #[test]
    fn the_parameter_count_matches_the_model() {
        let config = small_config();
        let expected = config.param_count(&[12, 12]);
        let model = PmcStateModel::init(config, &[12, 12]).unwrap();
        assert_eq!(
            model.param_count(),
            expected,
            "the parameter count does not describe the model"
        );
    }
}
