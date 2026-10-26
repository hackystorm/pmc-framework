// SPDX-License-Identifier: GPL-3.0-only

//! The trainer.
//!
//! Trains a small model on the synthetic memory task and reports what
//! happened. The Python implementation of the same mathematics lives in
//! `tests/test_rust_parity.py` and is the reference oracle: both must agree
//! on the final weights and on the metrics.

pub mod data;
pub mod export;
pub mod model;
pub mod optim;

pub use export::export_checkpoint;
pub use model::Mlp;

use std::time::Instant;

use pmc_core::{sha256_hex, Json, Rng, Tensor};

use data::MemoryTask;
use optim::AdamState;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub seed: u64,
    pub steps: usize,
    pub learning_rate: f32,
    pub hidden: usize,
    pub keys: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            seed: 20261026,
            steps: 200,
            learning_rate: 0.05,
            hidden: 32,
            keys: 16,
        }
    }
}

pub struct TrainReport {
    pub seed: u64,
    pub steps: usize,
    pub learning_rate: f32,
    pub hidden: usize,
    pub keys: usize,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub final_loss: f32,
    pub weight_digest: String,
    pub elapsed_millis: u128,
}

impl TrainReport {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("seed", Json::num(self.seed as f64)),
            ("steps", Json::num(self.steps as f64)),
            ("learning_rate", Json::num(self.learning_rate as f64)),
            ("hidden", Json::num(self.hidden as f64)),
            ("keys", Json::num(self.keys as f64)),
            ("initial_accuracy", Json::num(self.initial_accuracy as f64)),
            ("final_accuracy", Json::num(self.final_accuracy as f64)),
            ("final_loss", Json::num(self.final_loss as f64)),
            ("weight_digest", Json::str(self.weight_digest.clone())),
            ("elapsed_millis", Json::num(self.elapsed_millis as f64)),
        ])
    }
}

/// The index of the class in a one-hot target row.
///
/// Reading the label from column zero would silently train the model to
/// predict a single class, so the label is looked up properly.
pub(crate) fn target_index(targets: &Tensor, row: usize) -> usize {
    let cols = targets.cols();
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for index in 0..cols {
        let value = targets.get(row, index);
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best
}

fn accuracy(model: &Mlp, task: &MemoryTask) -> f32 {
    let mut correct = 0usize;
    for key in 0..task.keys {
        let input = task.input(key);
        let cache = model.forward(&input);
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for index in 0..task.keys {
            let value = cache.probs.get(0, index);
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        if best == task.mapping[key] {
            correct += 1;
        }
    }
    correct as f32 / task.keys as f32
}

fn loss(model: &Mlp, task: &MemoryTask, keys: &[usize]) -> f32 {
    let (inputs, targets) = task.batch(keys);
    let cache = model.forward(&inputs);
    let rows = cache.probs.rows();
    let mut total = 0.0f32;
    for i in 0..rows {
        let target = target_index(&targets, i);
        let probability = cache.probs.get(i, target).max(1e-9);
        total += -probability.ln();
    }
    total / rows as f32
}

pub fn train(config: &TrainConfig) -> (Mlp, TrainReport) {
    let started = Instant::now();
    let task = MemoryTask::new(config.keys, config.seed);
    let mut rng = Rng::new(config.seed);
    let mut model = Mlp::new(config.keys, config.hidden, config.keys, &mut rng);
    let initial_accuracy = accuracy(&model, &task);

    let mut optimiser = AdamState::new(&model.param_shapes());
    let all_keys: Vec<usize> = (0..config.keys).collect();

    for _ in 0..config.steps {
        let (inputs, targets) = task.batch(&all_keys);
        let cache = model.forward(&inputs);
        let grads = model.backward(&cache, &targets);
        let mut params: Vec<&mut Tensor> = vec![
            &mut model.w1,
            &mut model.b1,
            &mut model.w2,
            &mut model.b2,
        ];
        let gradient_refs: Vec<&Tensor> = vec![&grads.dw1, &grads.db1, &grads.dw2, &grads.db2];
        optimiser.step(
            &mut params,
            &gradient_refs,
            config.learning_rate,
            0.9,
            0.999,
            1e-8,
        );
    }

    let final_accuracy = accuracy(&model, &task);
    let final_loss = loss(&model, &task, &all_keys);
    let weight_digest = sha256_hex(&model.to_bytes());

    let report = TrainReport {
        seed: config.seed,
        steps: config.steps,
        learning_rate: config.learning_rate,
        hidden: config.hidden,
        keys: config.keys,
        initial_accuracy,
        final_accuracy,
        final_loss,
        weight_digest,
        elapsed_millis: started.elapsed().as_millis(),
    };

    (model, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trainer_learns_the_memory_task() {
        let config = TrainConfig {
            seed: 7,
            steps: 400,
            learning_rate: 0.05,
            hidden: 32,
            keys: 16,
        };
        let (_, report) = train(&config);
        assert!(
            report.final_accuracy > report.initial_accuracy,
            "training did not improve accuracy: {} -> {}",
            report.initial_accuracy,
            report.final_accuracy
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_weights() {
        let config = TrainConfig::default();
        let (_, first) = train(&config);
        let (_, second) = train(&config);
        assert_eq!(first.weight_digest, second.weight_digest);
    }
}

