// SPDX-License-Identifier: GPL-3.0-only

//! Core primitives for the PMC trainer.
//!
//! Zero external dependencies, on purpose. A general-purpose Rust ML stack
//! would cost more in build time, disk and compilation on a four-core
//! machine than it returns in throughput for a twenty-million-parameter
//! core, and it would make the trainer harder to audit.

pub mod json;
pub mod rng;
pub mod sha256;
pub mod tensor;

pub use json::Json;
pub use rng::Rng;
pub use sha256::{sha256, sha256_hex, Sha256};
pub use tensor::Tensor;

/// Version of the artifact format produced by the trainer.
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;
