// SPDX-License-Identifier: GPL-3.0-only

//! Library surface of the `pmc` binary.
//!
//! The binary is a thin argument parser. Every operation lives in a module
//! here so it can be exercised from tests and from the FFI layer without
//! spawning a process.
//!
//! `fuse` is the only operation that writes a model: it appends a delta
//! vocabulary and PMC state tensors to a frozen GGUF checkpoint, then verifies
//! every byte it claims to have preserved before publishing the result.
//!
//! `state_memory` writes the other kind of state: a trained PMC state
//! encoder's parameters, fused into the same file as dense neural weights, so
//! a memory is part of the model rather than a JSON file beside it.

#![forbid(unsafe_code)]

pub mod fuse;
pub mod state_memory;
