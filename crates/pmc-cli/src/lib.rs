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

#![forbid(unsafe_code)]

pub mod fuse;
