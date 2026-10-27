// SPDX-License-Identifier: GPL-3.0-only

//! Random-access GGUF reader.
//!
//! Opening a file parses the header, the metadata key/value table and the
//! tensor table, then records where the data section starts. Tensor data is
//! never loaded eagerly: [`GgufReader::read_row_f32`] seeks to one row, reads
//! exactly that row's bytes and dequantizes it. [`GgufReader::read_tensor_f32`]
//! is the only API that reads a whole tensor and it refuses to do so without a
//! caller-provided element limit.
//!
//! Row order follows the GGUF wire format: raw dimensions are stored with
//! `dims[0]` as the fastest-varying dimension (the row length). `gguf-py`
//! reverses this list when it prints `shape`.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::dequant::dequantize_row;
use crate::types::{quant_block, GgufError, GgufValue, TensorInfo};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian byte order.
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DEPTH: u32 = 8;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;

/// Parsed GGUF file with random access to dequantized tensor rows.
pub struct GgufReader {
    file: File,
    path: PathBuf,
    version: u32,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<TensorInfo>,
    data_start: u64,
    alignment: u64,
    file_len: u64,
}

impl GgufReader {
    /// Parse the header, metadata and tensor table of `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let mut parser = Parser::new(BufReader::new(file), file_len);

        let magic = parser.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = parser.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = parser.read_u64()?;
        let kv_count = parser.read_u64()?;

        // Smallest possible tensor entry: 8-byte name length, empty name,
        // 4-byte n_dims, 4-byte type, 8-byte offset.
        let remaining = parser.remaining();
        if tensor_count > remaining / 24 {
            return Err(GgufError::ImplausibleCount {
                field: "tensor",
                count: tensor_count,
                remaining,
            });
        }
        // Smallest metadata entry: 8-byte key length, empty key, 4-byte type,
        // one-byte value.
        if kv_count > remaining / 13 {
            return Err(GgufError::ImplausibleCount {
                field: "metadata",
                count: kv_count,
                remaining,
            });
        }

        let mut metadata: Vec<(String, GgufValue)> = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = parser.read_string()?;
            let type_id = parser.read_u32()?;
            let value = parser.read_value(type_id, 0)?;
            if metadata.iter().any(|(existing, _)| existing == &key) {
                return Err(GgufError::DuplicateMetadataKey(key));
            }
            metadata.push((key, value));
        }

        let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = parser.read_string()?;
            let n_dims = parser.read_u32()?;
            if n_dims > 8 {
                return Err(GgufError::TooManyDimensions(n_dims));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(parser.read_u64()?);
            }
            let kind = parser.read_u32()?;
            let offset = parser.read_u64()?;
            if tensors.iter().any(|t| t.name == name) {
                return Err(GgufError::DuplicateTensorName(name));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                kind,
                offset,
            });
        }

        let alignment = metadata
            .iter()
            .find(|(key, _)| key == "general.alignment")
            .and_then(|(_, value)| {
                value
                    .as_u64()
                    .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        let header_end = parser.position();
        let data_start = align_up(header_end, alignment).ok_or(GgufError::SizeOverflow {
            name: "<header>".to_string(),
        })?;
        if data_start > file_len {
            return Err(GgufError::Truncated {
                name: "<header>".to_string(),
                offset: header_end,
                bytes: data_start - header_end,
                file_len,
            });
        }

        Ok(GgufReader {
            file: parser.into_inner(),
            path,
            version,
            metadata,
            tensors,
            data_start,
            alignment,
            file_len,
        })
    }

    /// Path of the source file, as passed to [`GgufReader::open`].
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// GGUF container version (2 or 3).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Byte offset of the tensor data section in the file.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Alignment declared by `general.alignment`, or 32 if absent.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Size of the file in bytes.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// All metadata entries in file order.
    pub fn metadata(&self) -> &[(String, GgufValue)] {
        &self.metadata
    }

    /// Look up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// All tensor descriptors in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor descriptor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Number of elements in one row of `name`, or `None` if absent.
    pub fn row_len(&self, name: &str) -> Option<usize> {
        self.tensor(name).and_then(TensorInfo::row_len)
    }

    /// Number of rows of `name`, or `None` if absent.
    pub fn rows(&self, name: &str) -> Option<usize> {
        self.tensor(name).and_then(TensorInfo::rows)
    }

    /// Total number of elements in `name`, or `None` if absent.
    pub fn element_count(&self, name: &str) -> Option<usize> {
        self.tensor(name).and_then(TensorInfo::elements)
    }

    /// Read row `row` of `name`, dequantized to `f32`.
    ///
    /// Exactly `row_len(name)` values are written to `out[..row_len]`. The
    /// call reads only the bytes of that row.
    pub fn read_row_f32(
        &mut self,
        name: &str,
        row: usize,
        out: &mut [f32],
    ) -> Result<(), GgufError> {
        let (kind, tensor_offset, row_len, rows) = {
            let info = self
                .tensor(name)
                .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
            let row_len = info
                .row_len()
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: name.to_string(),
                })?;
            let rows = info.rows().ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
            (info.kind, info.offset, row_len, rows)
        };

        if row >= rows {
            return Err(GgufError::RowOutOfBounds {
                name: name.to_string(),
                row,
                rows,
            });
        }
        if out.len() < row_len {
            return Err(GgufError::OutputTooSmall {
                name: name.to_string(),
                expected: row_len,
                actual: out.len(),
            });
        }

        let (block_elems, block_bytes) =
            quant_block(kind).ok_or_else(|| GgufError::UnsupportedQuantType {
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

        let row_offset = (row as u64)
            .checked_mul(row_bytes as u64)
            .and_then(|delta| tensor_offset.checked_add(delta))
            .and_then(|relative| self.data_start.checked_add(relative))
            .ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
        let end = row_offset
            .checked_add(row_bytes as u64)
            .ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
        if end > self.file_len {
            return Err(GgufError::Truncated {
                name: name.to_string(),
                offset: row_offset,
                bytes: row_bytes as u64,
                file_len: self.file_len,
            });
        }

        let mut raw = vec![0u8; row_bytes];
        self.file.seek(SeekFrom::Start(row_offset))?;
        self.file.read_exact(&mut raw)?;
        dequantize_row(kind, &raw, &mut out[..row_len])
    }

    /// Read an entire tensor dequantized to `f32`.
    ///
    /// Returns [`GgufError::MaxElemsExceeded`] without reading anything if the
    /// tensor has more than `max_elems` elements. This is the only method that
    /// can allocate proportional to the tensor size, and the guard is
    /// mandatory.
    pub fn read_tensor_f32(
        &mut self,
        name: &str,
        max_elems: usize,
    ) -> Result<Vec<f32>, GgufError> {
        let (elements, row_len, rows) = {
            let info = self
                .tensor(name)
                .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
            let elements = info.elements().ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
            let row_len = info.row_len().ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
            let rows = info.rows().ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_string(),
            })?;
            (elements, row_len, rows)
        };

        if elements > max_elems {
            return Err(GgufError::MaxElemsExceeded {
                name: name.to_string(),
                requested: elements,
                max: max_elems,
            });
        }

        let mut out = vec![0f32; elements];
        for row in 0..rows {
            let start = row * row_len;
            self.read_row_f32(name, row, &mut out[start..start + row_len])?;
        }
        Ok(out)
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

struct Parser {
    inner: BufReader<File>,
    pos: u64,
    file_len: u64,
}

impl Parser {
    fn new(inner: BufReader<File>, file_len: u64) -> Self {
        Parser {
            inner,
            pos: 0,
            file_len,
        }
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn remaining(&self) -> u64 {
        self.file_len.saturating_sub(self.pos)
    }

    fn into_inner(self) -> File {
        self.inner.into_inner()
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        if buf.len() as u64 > self.remaining() {
            return Err(GgufError::Truncated {
                name: "<header>".to_string(),
                offset: self.pos,
                bytes: buf.len() as u64,
                file_len: self.file_len,
            });
        }
        self.inner.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES || len > self.remaining() {
            return Err(GgufError::ImplausibleCount {
                field: "string length",
                count: len,
                remaining: self.remaining(),
            });
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| GgufError::InvalidString)
    }

    fn read_value(&mut self, type_id: u32, depth: u32) -> Result<GgufValue, GgufError> {
        if depth > MAX_DEPTH {
            return Err(GgufError::MetadataTooDeep);
        }
        match type_id {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_u8()? as i8)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_u16()? as i16)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_u32()? as i32)),
            6 => Ok(GgufValue::F32(f32::from_bits(self.read_u32()?))),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()?;
                let min_size = min_value_size(elem_type).ok_or(GgufError::InvalidValueType(
                    elem_type,
                ))?;
                if len
                    .checked_mul(min_size)
                    .map_or(true, |needed| needed > self.remaining())
                {
                    return Err(GgufError::ImplausibleCount {
                        field: "array length",
                        count: len,
                        remaining: self.remaining(),
                    });
                }
                let mut items = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    items.push(self.read_value(elem_type, depth + 1)?);
                }
                Ok(GgufValue::Array { elem_type, items })
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_u64()? as i64)),
            12 => Ok(GgufValue::F64(f64::from_bits(self.read_u64()?))),
            other => Err(GgufError::InvalidValueType(other)),
        }
    }
}

fn min_value_size(type_id: u32) -> Option<u64> {
    match type_id {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        8 => Some(8),  // string: u64 length prefix, empty payload
        9 => Some(12), // nested array: u32 element type + u64 length
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rounds_up() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(u64::MAX, 32), None);
    }

    #[test]
    fn value_sizes_cover_all_types() {
        for type_id in 0..=12 {
            assert!(min_value_size(type_id).is_some(), "type {type_id}");
        }
        assert!(min_value_size(13).is_none());
    }
}
