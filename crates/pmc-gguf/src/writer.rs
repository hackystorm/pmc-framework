// SPDX-License-Identifier: GPL-3.0-only

//! Streaming GGUF writer for PMC fusion.
//!
//! The writer starts from a parsed [`GgufReader`], copies the source tensor
//! payloads in fixed-size chunks, and re-encodes the header after appending
//! metadata, tensors or vocabulary rows. Tensor offsets are recomputed from
//! `general.alignment`, because growing one tensor moves every tensor after it.
//!
//! No tensor or output file is held in memory. The only in-memory payloads are
//! the rows the caller appends explicitly (for example the vocabulary rows of a
//! dream) and the metadata header itself.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::quant::{quantize_row_q4_k, quantize_row_q5_k, quantize_row_q6_k};
use crate::reader::GgufReader;
use crate::types::{
    quant_geometry, GgufError, GgufValue, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian byte order.
const DEFAULT_ALIGNMENT: u64 = 32;
const COPY_CHUNK: usize = 4 * 1024 * 1024;
const ZERO_CHUNK: usize = 32 * 1024;

/// One tensor in the output table and where its bytes come from.
struct WriterTensor {
    name: String,
    dims: Vec<u64>,
    kind: u32,
    data: TensorData,
}

enum TensorData {
    /// Bytes copied from the source file at an absolute offset.
    Source { offset: u64, len: u64 },
    /// A source tensor followed by rows appended in memory.
    SourcePlus {
        offset: u64,
        len: u64,
        appended: Vec<u8>,
    },
    /// A tensor created from memory (for example `pmc.state.*`).
    Memory(Vec<u8>),
}

impl TensorData {
    fn len(&self, name: &str) -> Result<u64, GgufError> {
        match self {
            TensorData::Source { len, .. } => Ok(*len),
            TensorData::SourcePlus { len, appended, .. } => {
                let appended = u64::try_from(appended.len()).map_err(|_| GgufError::SizeOverflow {
                    name: name.to_string(),
                })?;
                len.checked_add(appended)
                    .ok_or_else(|| GgufError::SizeOverflow { name: name.to_string() })
            }
            TensorData::Memory(bytes) => {
                u64::try_from(bytes.len()).map_err(|_| GgufError::SizeOverflow {
                    name: name.to_string(),
                })
            }
        }
    }
}

/// GGUF writer that preserves the source tensors and extends the vocabulary.
pub struct GgufWriter {
    source_path: PathBuf,
    source_file_len: u64,
    version: u32,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<WriterTensor>,
    vocab_changed: bool,
    embeddings_changed: bool,
}

impl GgufWriter {
    /// Prepare a writer from an open source file.
    ///
    /// Metadata and the tensor table are copied into memory; tensor payloads
    /// stay in the source file and are streamed by [`GgufWriter::write_to`].
    pub fn from_source(reader: &GgufReader) -> Result<Self, GgufError> {
        let mut tensors = Vec::with_capacity(reader.tensors().len());
        for info in reader.tensors() {
            let len = tensor_nbytes(&info.name, &info.dims, info.kind)?;
            let offset = reader
                .data_start()
                .checked_add(info.offset)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: info.name.clone(),
                })?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: info.name.clone(),
                })?;
            if end > reader.file_len() {
                return Err(GgufError::Truncated {
                    name: info.name.clone(),
                    offset,
                    bytes: len,
                    file_len: reader.file_len(),
                });
            }
            tensors.push(WriterTensor {
                name: info.name.clone(),
                dims: info.dims.clone(),
                kind: info.kind,
                data: TensorData::Source { offset, len },
            });
        }

        Ok(GgufWriter {
            source_path: reader.path().to_path_buf(),
            source_file_len: reader.file_len(),
            version: reader.version(),
            metadata: reader.metadata().to_vec(),
            tensors,
            vocab_changed: false,
            embeddings_changed: false,
        })
    }

    /// Append a new F32 tensor.
    ///
    /// `dims` uses the GGUF wire order: `dims[0]` is the row length. The data
    /// slice must contain exactly `product(dims)` values.
    pub fn append_tensor_f32(
        &mut self,
        name: &str,
        dims: &[u64],
        data: &[f32],
    ) -> Result<(), GgufError> {
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(GgufError::DuplicateTensorName(name.to_string()));
        }
        if dims.len() > 8 {
            return Err(GgufError::TooManyDimensions(dims.len() as u32));
        }
        let expected = element_count(dims).ok_or_else(|| GgufError::SizeOverflow {
            name: name.to_string(),
        })?;
        if expected != data.len() as u64 {
            return Err(GgufError::InvalidTensorData {
                name: name.to_string(),
                expected: usize::try_from(expected).unwrap_or(usize::MAX),
                actual: data.len(),
            });
        }

        let capacity = data
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        self.tensors.push(WriterTensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            kind: GGML_TYPE_F32,
            data: TensorData::Memory(bytes),
        });
        Ok(())
    }

    /// Append vocabulary entries to the three tokenizer arrays.
    ///
    /// The arrays are extended in lockstep. If this model has no
    /// `tokenizer.ggml.scores` or `tokenizer.ggml.token_type` array, it is
    /// created with zero-filled entries for the existing vocabulary, so all
    /// arrays end up with the same length.
    pub fn append_vocab_tokens(
        &mut self,
        tokens: &[String],
        scores: &[f32],
        token_types: &[i32],
    ) -> Result<(), GgufError> {
        if tokens.len() != scores.len() || tokens.len() != token_types.len() {
            return Err(GgufError::VocabLengthMismatch {
                tokens: tokens.len(),
                scores: scores.len(),
                token_types: token_types.len(),
            });
        }
        if tokens.is_empty() {
            return Ok(());
        }

        let (tokens_elem, old_len) = self
            .array_info("tokenizer.ggml.tokens")?
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".to_string()))?;
        if tokens_elem != 8 {
            return Err(GgufError::InvalidMetadataType {
                key: "tokenizer.ggml.tokens".to_string(),
                expected: 8,
                actual: tokens_elem,
            });
        }

        if let Some((elem_type, len)) = self.array_info("tokenizer.ggml.scores")? {
            if elem_type != 6 {
                return Err(GgufError::InvalidMetadataType {
                    key: "tokenizer.ggml.scores".to_string(),
                    expected: 6,
                    actual: elem_type,
                });
            }
            if len != old_len {
                return Err(GgufError::MetadataLengthMismatch {
                    key: "tokenizer.ggml.scores".to_string(),
                    expected: old_len,
                    actual: len,
                });
            }
        }
        if let Some((elem_type, len)) = self.array_info("tokenizer.ggml.token_type")? {
            if elem_type != 5 {
                return Err(GgufError::InvalidMetadataType {
                    key: "tokenizer.ggml.token_type".to_string(),
                    expected: 5,
                    actual: elem_type,
                });
            }
            if len != old_len {
                return Err(GgufError::MetadataLengthMismatch {
                    key: "tokenizer.ggml.token_type".to_string(),
                    expected: old_len,
                    actual: len,
                });
            }
        }

        let new_tokens: Vec<GgufValue> = tokens
            .iter()
            .cloned()
            .map(GgufValue::String)
            .collect();
        let new_scores: Vec<GgufValue> = scores.iter().copied().map(GgufValue::F32).collect();
        let new_token_types: Vec<GgufValue> =
            token_types.iter().copied().map(GgufValue::I32).collect();

        self.append_to_array("tokenizer.ggml.tokens", 8, old_len, new_tokens, false)?;
        self.append_to_array("tokenizer.ggml.scores", 6, old_len, new_scores, true)?;
        self.append_to_array(
            "tokenizer.ggml.token_type",
            5,
            old_len,
            new_token_types,
            true,
        )?;

        let new_len = old_len
            .checked_add(tokens.len())
            .ok_or_else(|| GgufError::SizeOverflow {
                name: "tokenizer.ggml.tokens".to_string(),
            })?;
        self.update_arch_vocab_size(new_len)?;

        self.vocab_changed = true;
        Ok(())
    }

    /// Append embedding rows and grow the vocabulary dimension.
    ///
    /// The rows are quantized to the existing type of `token_embd.weight`
    /// (Q4_K, Q5_K or Q6_K). If an untied `output.weight` tensor exists, the
    /// same floating-point rows are also appended to it, quantized to its own
    /// type. A tied model has no `output.weight`, so there is nothing to do.
    pub fn append_embedding_rows(&mut self, rows: &[Vec<f32>]) -> Result<(), GgufError> {
        if rows.is_empty() {
            return Ok(());
        }

        let token_idx = self.tensor_index("token_embd.weight")?;
        let (token_row_len, token_kind) = self.embedding_shape(token_idx)?;

        let token_bytes =
            quantize_rows("token_embd.weight", rows, token_row_len, token_kind)?;

        let output_idx = self.tensor_index_opt("output.weight");
        let output_bytes = if let Some(index) = output_idx {
            let (output_row_len, output_kind) = self.embedding_shape(index)?;
            if output_row_len != token_row_len {
                return Err(GgufError::RowLengthMismatch {
                    name: "output.weight".to_string(),
                    expected: token_row_len,
                    actual: output_row_len,
                });
            }
            Some(quantize_rows(
                "output.weight",
                rows,
                output_row_len,
                output_kind,
            )?)
        } else {
            None
        };

        let row_count = rows.len() as u64;
        self.append_raw(token_idx, token_bytes);
        self.bump_rows(token_idx, row_count)?;

        if let Some(index) = output_idx {
            if let Some(bytes) = output_bytes {
                self.append_raw(index, bytes);
                self.bump_rows(index, row_count)?;
            }
        }

        self.embeddings_changed = true;
        Ok(())
    }

    /// Set or replace a metadata key.
    pub fn set_metadata(&mut self, key: &str, value: GgufValue) -> Result<(), GgufError> {
        if let Some(entry) = self.metadata.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value;
        } else {
            self.metadata.push((key.to_string(), value));
        }
        Ok(())
    }

    /// Write the fused GGUF to `path`.
    ///
    /// Tensor payloads are copied in 4 MiB chunks. The data section is laid
    /// out in tensor-table order with each tensor padded to
    /// `general.alignment`, matching the reference GGUF writers.
    pub fn write_to(&self, path: &Path) -> Result<(), GgufError> {
        self.validate_vocab()?;
        if path == self.source_path {
            return Err(GgufError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output path must differ from the source path",
            )));
        }

        let alignment = self.alignment()?;

        let placeholder = vec![0u64; self.tensors.len()];
        let header_probe = self.encode_header(&placeholder)?;
        let data_start =
            align_up(header_probe.len() as u64, alignment).ok_or_else(|| {
                GgufError::SizeOverflow {
                    name: "<header>".to_string(),
                }
            })?;

        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut lengths = Vec::with_capacity(self.tensors.len());
        let mut relative = 0u64;
        for tensor in &self.tensors {
            offsets.push(relative);
            let len = tensor.data.len(&tensor.name)?;
            lengths.push(len);
            let padded =
                align_up(len, alignment).ok_or_else(|| GgufError::SizeOverflow {
                    name: tensor.name.clone(),
                })?;
            relative = relative
                .checked_add(padded)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: tensor.name.clone(),
                })?;
        }

        let header = self.encode_header(&offsets)?;
        if header.len() as u64 != header_probe.len() as u64 {
            return Err(GgufError::SizeOverflow {
                name: "<header>".to_string(),
            });
        }

        let mut source = File::open(&self.source_path)?;
        let source_len = source.metadata()?.len();
        if source_len != self.source_file_len {
            return Err(GgufError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "source GGUF changed after it was opened",
            )));
        }

        let mut output = File::create(path)?;
        output.write_all(&header)?;
        write_zeros(
            &mut output,
            data_start
                .checked_sub(header.len() as u64)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: "<header>".to_string(),
                })?,
        )?;

        let mut copy_buffer = vec![0u8; COPY_CHUNK];
        for (index, tensor) in self.tensors.iter().enumerate() {
            match &tensor.data {
                TensorData::Source { offset, len } => {
                    copy_range(&mut source, &mut output, *offset, *len, &mut copy_buffer)?;
                }
                TensorData::SourcePlus {
                    offset,
                    len,
                    appended,
                } => {
                    copy_range(&mut source, &mut output, *offset, *len, &mut copy_buffer)?;
                    output.write_all(appended)?;
                }
                TensorData::Memory(bytes) => {
                    output.write_all(bytes)?;
                }
            }

            let padded = align_up(lengths[index], alignment).ok_or_else(|| {
                GgufError::SizeOverflow {
                    name: tensor.name.clone(),
                }
            })?;
            write_zeros(&mut output, padded - lengths[index])?;
        }

        Ok(())
    }

    fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, value)| value)
    }

    fn array_info(&self, key: &str) -> Result<Option<(u32, usize)>, GgufError> {
        match self.get_metadata(key) {
            None => Ok(None),
            Some(GgufValue::Array { elem_type, items }) => Ok(Some((*elem_type, items.len()))),
            Some(value) => Err(GgufError::InvalidMetadataType {
                key: key.to_string(),
                expected: 9,
                actual: value.type_id(),
            }),
        }
    }

    fn append_to_array(
        &mut self,
        key: &str,
        expected_type: u32,
        old_len: usize,
        new_items: Vec<GgufValue>,
        allow_create: bool,
    ) -> Result<(), GgufError> {
        for item in &new_items {
            if item.type_id() != expected_type {
                return Err(GgufError::InvalidMetadataType {
                    key: key.to_string(),
                    expected: expected_type,
                    actual: item.type_id(),
                });
            }
        }

        if let Some(index) = self.metadata.iter().position(|(k, _)| k == key) {
            match &mut self.metadata[index].1 {
                GgufValue::Array { elem_type, items } => {
                    if *elem_type != expected_type {
                        return Err(GgufError::InvalidMetadataType {
                            key: key.to_string(),
                            expected: expected_type,
                            actual: *elem_type,
                        });
                    }
                    if items.len() != old_len {
                        return Err(GgufError::MetadataLengthMismatch {
                            key: key.to_string(),
                            expected: old_len,
                            actual: items.len(),
                        });
                    }
                    items.extend(new_items);
                }
                value => {
                    return Err(GgufError::InvalidMetadataType {
                        key: key.to_string(),
                        expected: 9,
                        actual: value.type_id(),
                    });
                }
            }
            return Ok(());
        }

        if !allow_create {
            return Err(GgufError::MissingMetadata(key.to_string()));
        }

        let mut items = Vec::with_capacity(
            old_len
                .checked_add(new_items.len())
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: key.to_string(),
                })?,
        );
        for _ in 0..old_len {
            items.push(default_value(expected_type)?);
        }
        items.extend(new_items);
        self.metadata.push((
            key.to_string(),
            GgufValue::Array {
                elem_type: expected_type,
                items,
            },
        ));
        Ok(())
    }

    /// Keep `<architecture>.vocab_size` in sync when the source advertises it.
    /// The tokenizer list remains the authority, but a stale architecture
    /// field confuses tools that compare metadata before loading tensors.
    fn update_arch_vocab_size(&mut self, size: usize) -> Result<(), GgufError> {
        let arch = match self.get_metadata("general.architecture") {
            None => return Ok(()),
            Some(GgufValue::String(arch)) => arch.clone(),
            Some(value) => {
                return Err(GgufError::InvalidMetadataType {
                    key: "general.architecture".to_string(),
                    expected: 8,
                    actual: value.type_id(),
                });
            }
        };
        let key = format!("{arch}.vocab_size");
        let Some((_, value)) = self.metadata.iter_mut().find(|(k, _)| k == &key) else {
            return Ok(());
        };
        let size = u64::try_from(size).map_err(|_| GgufError::SizeOverflow {
            name: key.clone(),
        })?;
        match value {
            GgufValue::U32(v) => {
                *v = u32::try_from(size).map_err(|_| GgufError::SizeOverflow {
                    name: key.clone(),
                })?;
            }
            GgufValue::U64(v) => *v = size,
            GgufValue::I32(v) => {
                *v = i32::try_from(size).map_err(|_| GgufError::SizeOverflow {
                    name: key.clone(),
                })?;
            }
            GgufValue::I64(v) => {
                *v = i64::try_from(size).map_err(|_| GgufError::SizeOverflow {
                    name: key.clone(),
                })?;
            }
            other => {
                return Err(GgufError::InvalidMetadataType {
                    key,
                    expected: 4,
                    actual: other.type_id(),
                });
            }
        }
        Ok(())
    }

    fn tensor_index(&self, name: &str) -> Result<usize, GgufError> {
        self.tensor_index_opt(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    fn tensor_index_opt(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    fn embedding_shape(&self, index: usize) -> Result<(usize, u32), GgufError> {
        let tensor = &self.tensors[index];
        if tensor.dims.len() != 2 {
            return Err(GgufError::InvalidTensorShape {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
            });
        }
        let row_len =
            usize::try_from(tensor.dims[0]).map_err(|_| GgufError::SizeOverflow {
                name: tensor.name.clone(),
            })?;
        Ok((row_len, tensor.kind))
    }

    fn append_raw(&mut self, index: usize, bytes: Vec<u8>) {
        let tensor = &mut self.tensors[index];
        let old = std::mem::replace(&mut tensor.data, TensorData::Memory(Vec::new()));
        tensor.data = match old {
            TensorData::Source { offset, len } => TensorData::SourcePlus {
                offset,
                len,
                appended: bytes,
            },
            TensorData::SourcePlus {
                offset,
                len,
                mut appended,
            } => {
                appended.extend(bytes);
                TensorData::SourcePlus {
                    offset,
                    len,
                    appended,
                }
            }
            TensorData::Memory(mut existing) => {
                existing.extend(bytes);
                TensorData::Memory(existing)
            }
        };
    }

    fn bump_rows(&mut self, index: usize, count: u64) -> Result<(), GgufError> {
        let tensor = &mut self.tensors[index];
        tensor.dims[1] = tensor
            .dims[1]
            .checked_add(count)
            .ok_or_else(|| GgufError::SizeOverflow {
                name: tensor.name.clone(),
            })?;
        Ok(())
    }

    fn alignment(&self) -> Result<u64, GgufError> {
        let alignment = match self.get_metadata("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(value) => value
                .as_u64()
                .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
                .ok_or_else(|| GgufError::InvalidMetadataType {
                    key: "general.alignment".to_string(),
                    expected: 4,
                    actual: value.type_id(),
                })?,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }
        Ok(alignment)
    }

    fn validate_vocab(&self) -> Result<(), GgufError> {
        if !self.vocab_changed && !self.embeddings_changed {
            return Ok(());
        }

        let Some((elem_type, token_count)) = self.array_info("tokenizer.ggml.tokens")? else {
            return Ok(());
        };
        if elem_type != 8 {
            return Err(GgufError::InvalidMetadataType {
                key: "tokenizer.ggml.tokens".to_string(),
                expected: 8,
                actual: elem_type,
            });
        }

        if let Some(index) = self.tensor_index_opt("token_embd.weight") {
            let rows = tensor_rows(&self.tensors[index].dims).ok_or_else(|| {
                GgufError::SizeOverflow {
                    name: "token_embd.weight".to_string(),
                }
            })?;
            if rows != token_count as u64 {
                return Err(GgufError::VocabSizeMismatch {
                    tokens: token_count,
                    rows: usize::try_from(rows).unwrap_or(usize::MAX),
                });
            }
        }

        if self.vocab_changed {
            for key in ["tokenizer.ggml.scores", "tokenizer.ggml.token_type"] {
                if let Some((_, len)) = self.array_info(key)? {
                    if len != token_count {
                        return Err(GgufError::MetadataLengthMismatch {
                            key: key.to_string(),
                            expected: token_count,
                            actual: len,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn encode_header(&self, offsets: &[u64]) -> Result<Vec<u8>, GgufError> {
        if offsets.len() != self.tensors.len() {
            return Err(GgufError::SizeOverflow {
                name: "<tensor table>".to_string(),
            });
        }

        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for (key, value) in &self.metadata {
            push_string(&mut out, key);
            out.extend_from_slice(&value.type_id().to_le_bytes());
            encode_value(&mut out, value)?;
        }

        for (tensor, offset) in self.tensors.iter().zip(offsets) {
            push_string(&mut out, &tensor.name);
            out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.kind.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }

        Ok(out)
    }
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn encode_value(out: &mut Vec<u8>, value: &GgufValue) -> Result<(), GgufError> {
    match value {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => out.push(u8::from(*v)),
        GgufValue::String(v) => push_string(out, v),
        GgufValue::Array { elem_type, items } => {
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                if item.type_id() != *elem_type {
                    return Err(GgufError::InvalidMetadataType {
                        key: "<array element>".to_string(),
                        expected: *elem_type,
                        actual: item.type_id(),
                    });
                }
                encode_value(out, item)?;
            }
        }
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
    }
    Ok(())
}

fn default_value(type_id: u32) -> Result<GgufValue, GgufError> {
    match type_id {
        0 => Ok(GgufValue::U8(0)),
        1 => Ok(GgufValue::I8(0)),
        2 => Ok(GgufValue::U16(0)),
        3 => Ok(GgufValue::I16(0)),
        4 => Ok(GgufValue::U32(0)),
        5 => Ok(GgufValue::I32(0)),
        6 => Ok(GgufValue::F32(0.0)),
        7 => Ok(GgufValue::Bool(false)),
        8 => Ok(GgufValue::String(String::new())),
        10 => Ok(GgufValue::U64(0)),
        11 => Ok(GgufValue::I64(0)),
        12 => Ok(GgufValue::F64(0.0)),
        other => Err(GgufError::InvalidValueType(other)),
    }
}

fn quantize_rows(
    name: &str,
    rows: &[Vec<f32>],
    row_len: usize,
    kind: u32,
) -> Result<Vec<u8>, GgufError> {
    let (block_elems, block_bytes) =
        quant_geometry(kind).ok_or_else(|| GgufError::UnsupportedQuantType {
            name: name.to_string(),
            kind,
        })?;
    if row_len % block_elems != 0 {
        return Err(GgufError::InvalidRowLength {
            name: name.to_string(),
            row_len,
            block_elems,
        });
    }
    let row_bytes = row_len / block_elems * block_bytes;
    let total = rows
        .len()
        .checked_mul(row_bytes)
        .ok_or_else(|| GgufError::SizeOverflow {
            name: name.to_string(),
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| GgufError::SizeOverflow {
            name: name.to_string(),
        })?;

    for row in rows {
        if row.len() != row_len {
            return Err(GgufError::RowLengthMismatch {
                name: name.to_string(),
                expected: row_len,
                actual: row.len(),
            });
        }
        let start = out.len();
        out.resize(start + row_bytes, 0);
        let dst = &mut out[start..start + row_bytes];
        match kind {
            GGML_TYPE_Q4_K => quantize_row_q4_k(row, dst)?,
            GGML_TYPE_Q5_K => quantize_row_q5_k(row, dst)?,
            GGML_TYPE_Q6_K => quantize_row_q6_k(row, dst)?,
            _ => {
                return Err(GgufError::UnsupportedQuantType {
                    name: name.to_string(),
                    kind,
                });
            }
        }
    }
    Ok(out)
}

fn tensor_nbytes(name: &str, dims: &[u64], kind: u32) -> Result<u64, GgufError> {
    let (block_elems, block_bytes) =
        quant_geometry(kind).ok_or_else(|| GgufError::UnsupportedQuantType {
            name: name.to_string(),
            kind,
        })?;
    let elements = element_count(dims).ok_or_else(|| GgufError::SizeOverflow {
        name: name.to_string(),
    })?;
    let block_elems = block_elems as u64;
    if elements % block_elems != 0 {
        return Err(GgufError::InvalidElementCount {
            name: name.to_string(),
            elements,
            block_elems: block_elems as usize,
        });
    }
    (elements / block_elems)
        .checked_mul(block_bytes as u64)
        .ok_or_else(|| GgufError::SizeOverflow {
            name: name.to_string(),
        })
}

fn element_count(dims: &[u64]) -> Option<u64> {
    if dims.is_empty() {
        return Some(1);
    }
    let mut elements = 1u64;
    for &dim in dims {
        elements = elements.checked_mul(dim)?;
    }
    Some(elements)
}

fn tensor_rows(dims: &[u64]) -> Option<u64> {
    if dims.len() <= 1 {
        return Some(1);
    }
    let mut rows = 1u64;
    for &dim in &dims[1..] {
        rows = rows.checked_mul(dim)?;
    }
    Some(rows)
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn write_zeros(output: &mut File, mut count: u64) -> Result<(), GgufError> {
    let zeros = [0u8; ZERO_CHUNK];
    while count > 0 {
        let take = std::cmp::min(count, zeros.len() as u64) as usize;
        output.write_all(&zeros[..take])?;
        count -= take as u64;
    }
    Ok(())
}

fn copy_range(
    source: &mut File,
    output: &mut File,
    offset: u64,
    len: u64,
    buffer: &mut [u8],
) -> Result<(), GgufError> {
    source.seek(SeekFrom::Start(offset))?;
    let mut remaining = len;
    while remaining > 0 {
        let take = std::cmp::min(remaining, buffer.len() as u64) as usize;
        source.read_exact(&mut buffer[..take])?;
        output.write_all(&buffer[..take])?;
        remaining -= take as u64;
    }
    Ok(())
}
