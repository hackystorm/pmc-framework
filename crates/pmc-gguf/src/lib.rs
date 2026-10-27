// SPDX-License-Identifier: GPL-3.0-only

//! GGUF access for the PMC trainer.
//!
//! The trainer must read a frozen model's embedding table without ever
//! holding the whole file in memory. This crate opens a GGUF, exposes its
//! metadata and tensor table, and dequantizes single rows on demand.
//!
//! Interface contract (do not change without updating the orchestrator):
//!
//! ```text
//! GgufReader::open(path) -> Result<GgufReader, GgufError>
//! reader.tensor(name) -> Option<&TensorInfo>
//! reader.row_len(name) -> Option<usize>          // elements in one row
//! reader.read_row_f32(name, row, &mut out) -> Result<(), GgufError>
//! ```
//!
//! Row order is the raw GGUF order: `dims[0]` is the fastest-varying
//! dimension, so it is the length of one row. `gguf-py` prints `shape`
//! reversed because NumPy expects the slowest-varying dimension first.
//!
//! Supported quantization types are F32, F16, Q8_0, Q4_K, Q5_K and Q6_K.
//! Block layouts and arithmetic follow `llama.cpp/ggml/src/ggml-quants.c`.
//!
//! The writer extends an existing file without loading its tensors:
//!
//! ```text
//! GgufWriter::from_source(&reader) -> Result<GgufWriter, GgufError>
//! writer.append_vocab_tokens(tokens, scores, token_types)
//! writer.append_embedding_rows(rows)            // Q4_K/Q5_K/Q6_K target
//! writer.append_tensor_f32(name, dims, data)
//! writer.set_metadata(key, value)
//! writer.write_to(path)
//! ```
//!
//! Unknown metadata and unknown tensor types are preserved byte for byte;
//! only offsets and explicitly appended data move.

pub mod dequant;
pub mod quant;
pub mod reader;
pub mod types;
pub mod writer;

pub use dequant::{dequantize_row, f16_to_f32, f32_to_f16};
pub use quant::{quantize_row_q4_k, quantize_row_q5_k, quantize_row_q6_k};
pub use reader::GgufReader;
pub use types::{
    quant_block, quant_geometry, quant_name, GgufError, GgufValue, TensorInfo, GGML_TYPE_F16,
    GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, QK_K,
};
pub use writer::GgufWriter;
