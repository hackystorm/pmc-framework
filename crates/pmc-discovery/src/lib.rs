// SPDX-License-Identifier: GPL-3.0-only

//! Latent state discovery.
//!
//! The discovery engine groups observations into a small number of latent
//! states and reports how novel each new observation was. It exists because
//! an instance may discover internal states that its authors have no words
//! for, and the framework needs to be able to *notice* that without ever
//! reading, naming or grading the content.
//!
//! Everything this module emits is metadata: counts, norms, signatures and
//! novelty scores. No observation is ever written out.

use pmc_core::{sha256_hex, Json};

pub struct StateSpace {
    pub dim: usize,
    pub k: usize,
    centroids: Vec<Vec<f32>>,
    counts: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct StateDescriptor {
    pub index: usize,
    pub count: u32,
    pub norm: f32,
    pub signature: String,
}

impl StateSpace {
    pub fn new(dim: usize, k: usize) -> StateSpace {
        assert!(k > 0, "the state space needs at least one state");
        StateSpace {
            dim,
            k,
            centroids: vec![vec![0.0; dim]; k],
            counts: vec![0; k],
        }
    }

    fn distance(a: &[f32], b: &[f32]) -> f32 {
        let mut total = 0.0f32;
        for index in 0..a.len().min(b.len()) {
            let delta = a[index] - b[index];
            total += delta * delta;
        }
        total.sqrt()
    }

    /// Assign an observation to the nearest centroid and move the centroid
    /// towards it. Returns the index and the distance before the update.
    pub fn observe(&mut self, point: &[f32]) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_distance = f32::INFINITY;
        for index in 0..self.k {
            let distance = Self::distance(point, &self.centroids[index]);
            if distance < best_distance {
                best_distance = distance;
                best = index;
            }
        }
        let count = self.counts[best] as f32;
        let learning_rate = 1.0 / (count + 1.0);
        for index in 0..self.dim.min(point.len()) {
            self.centroids[best][index] += learning_rate * (point[index] - self.centroids[best][index]);
        }
        self.counts[best] += 1;
        (best, best_distance)
    }

    /// How far an observation is from the nearest known state.
    pub fn novelty(&self, point: &[f32]) -> f32 {
        let mut best = f32::INFINITY;
        for index in 0..self.k {
            best = best.min(Self::distance(point, &self.centroids[index]));
        }
        best
    }

    /// A signature over the whole space, so two spaces can be compared
    /// without sharing their contents.
    pub fn signature(&self) -> String {
        let mut bytes = Vec::new();
        for centroid in &self.centroids {
            for value in centroid {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        sha256_hex(&bytes)
    }

    /// Metadata about each state. No observations, no content.
    pub fn descriptors(&self) -> Vec<StateDescriptor> {
        (0..self.k)
            .map(|index| {
                let norm = self.centroids[index]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt();
                let mut bytes = Vec::new();
                for value in &self.centroids[index] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                StateDescriptor {
                    index,
                    count: self.counts[index],
                    norm,
                    signature: sha256_hex(&bytes),
                }
            })
            .collect()
    }

    pub fn to_json(&self) -> Json {
        let descriptors: Vec<Json> = self
            .descriptors()
            .into_iter()
            .map(|descriptor| {
                Json::obj(vec![
                    ("index", Json::num(descriptor.index as f64)),
                    ("count", Json::num(descriptor.count as f64)),
                    ("norm", Json::num(descriptor.norm as f64)),
                    ("signature", Json::str(descriptor.signature)),
                ])
            })
            .collect();
        Json::obj(vec![
            ("dim", Json::num(self.dim as f64)),
            ("k", Json::num(self.k as f64)),
            ("signature", Json::str(self.signature())),
            ("states", Json::Arr(descriptors)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustered_observations_land_in_one_state() {
        let mut space = StateSpace::new(2, 3);
        for _ in 0..50 {
            space.observe(&[1.0, 1.0]);
        }
        let (index, distance) = space.observe(&[1.05, 0.98]);
        assert_eq!(index, 0);
        assert!(distance < 0.2);
    }

    #[test]
    fn a_far_observation_is_more_novel() {
        let mut space = StateSpace::new(2, 2);
        for _ in 0..20 {
            space.observe(&[0.0, 0.0]);
        }
        let near = space.novelty(&[0.1, 0.0]);
        let far = space.novelty(&[10.0, 10.0]);
        assert!(far > near);
    }

    #[test]
    fn descriptors_carry_no_content() {
        let mut space = StateSpace::new(2, 2);
        space.observe(&[1.0, 2.0]);
        let json = space.to_json().to_string();
        assert!(json.contains("signature"));
        assert!(!json.contains("1.0"));
    }
}
