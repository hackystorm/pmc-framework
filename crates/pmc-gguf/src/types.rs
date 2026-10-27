// SPDX-License-Identifier: GPL-3.0-only

//! Core GGUF types: errors, metadata values, tensor descriptors and the
//! quantization constants used by the reader and the dequantizers.

use std::error::Error;
use std::fmt;
use std::io;

/// Errors produced while parsing a GGUF file or decoding a tensor row.
#[derive(Debug)]
pub enum GgufError {
    /// Underlying I/O error (short read, seek failure, ...).
    Io(io::Error),
    /// The file does not start with the `GGUF` magic.
    BadMagic(u32),
    /// The GGUF container version is not supported.
    UnsupportedVersion(u32),
    /// A metadata value uses an unknown type id.
    InvalidValueType(u32),
    /// A string field is not valid UTF-8.
    InvalidString,
    /// A metadata array is nested deeper than the parser allows.
    MetadataTooDeep,
    /// A tensor declares more dimensions than the format permits.
    TooManyDimensions(u32),
    /// A metadata key is duplicated.
    DuplicateMetadataKey(String),
    /// Two tensors share the same name.
    DuplicateTensorName(String),
    /// A count in the header is larger than the remaining file could hold.
    ImplausibleCount {
        field: &'static str,
        count: u64,
        remaining: u64,
    },
    /// A metadata key required for an operation is absent.
    MissingMetadata(String),
    /// A metadata value has the wrong GGUF type.
    InvalidMetadataType {
        key: String,
        expected: u32,
        actual: u32,
    },
    /// Existing metadata arrays have lengths that cannot be extended together.
    MetadataLengthMismatch {
        key: String,
        expected: usize,
        actual: usize,
    },
    /// The tokenizer arrays passed to the writer have different lengths.
    VocabLengthMismatch {
        tokens: usize,
        scores: usize,
        token_types: usize,
    },
    /// The tokenizer vocabulary size and the embedding table disagree.
    VocabSizeMismatch {
        tokens: usize,
        rows: usize,
    },
    /// A tensor name is not present in the tensor table.
    TensorNotFound(String),
    /// A tensor has a shape the writer cannot extend.
    InvalidTensorShape {
        name: String,
        dims: Vec<u64>,
    },
    /// The flat f32 buffer does not match the declared tensor dimensions.
    InvalidTensorData {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// A row has a different length than the tensor's row length.
    RowLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// The element count is not a multiple of the quantization block size.
    InvalidElementCount {
        name: String,
        elements: u64,
        block_elems: usize,
    },
    /// Requested row is outside the tensor.
    RowOutOfBounds {
        name: String,
        row: usize,
        rows: usize,
    },
    /// The destination slice is shorter than one row.
    OutputTooSmall {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// The tensor uses a quantization type this crate cannot decode.
    UnsupportedQuantType {
        name: String,
        kind: u32,
    },
    /// The row length is not a multiple of the quantization block size.
    InvalidRowLength {
        name: String,
        row_len: usize,
        block_elems: usize,
    },
    /// The file is truncated or an offset points past its end.
    Truncated {
        name: String,
        offset: u64,
        bytes: u64,
        file_len: u64,
    },
    /// A raw byte buffer is too small for the requested dequantization.
    BufferTooSmall {
        needed: usize,
        actual: usize,
    },
    /// A tensor is larger than the caller-provided element guard.
    MaxElemsExceeded {
        name: String,
        requested: usize,
        max: usize,
    },
    /// Arithmetic on sizes or offsets overflowed.
    SizeOverflow {
        name: String,
    },
    /// The alignment field is zero or not usable.
    InvalidAlignment(u64),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::BadMagic(m) => {
                write!(f, "not a GGUF file: magic 0x{m:08x}")
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v}")
            }
            GgufError::InvalidValueType(t) => {
                write!(f, "unknown GGUF metadata value type {t}")
            }
            GgufError::InvalidString => write!(f, "metadata string is not valid UTF-8"),
            GgufError::MetadataTooDeep => write!(f, "metadata arrays are nested too deeply"),
            GgufError::TooManyDimensions(n) => {
                write!(f, "tensor declares {n} dimensions, at most 8 are allowed")
            }
            GgufError::DuplicateMetadataKey(k) => {
                write!(f, "duplicate metadata key {k:?}")
            }
            GgufError::DuplicateTensorName(n) => write!(f, "duplicate tensor name {n:?}"),
            GgufError::ImplausibleCount {
                field,
                count,
                remaining,
            } => write!(
                f,
                "implausible {field} count {count} with only {remaining} bytes remaining"
            ),
            GgufError::MissingMetadata(k) => write!(f, "required metadata key {k:?} is missing"),
            GgufError::InvalidMetadataType {
                key,
                expected,
                actual,
            } => write!(
                f,
                "metadata {key:?} has type {actual}, expected {expected}"
            ),
            GgufError::MetadataLengthMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "metadata array {key:?} has length {actual}, expected {expected}"
            ),
            GgufError::VocabLengthMismatch {
                tokens,
                scores,
                token_types,
            } => write!(
                f,
                "vocabulary arrays disagree: tokens={tokens}, scores={scores}, token_types={token_types}"
            ),
            GgufError::VocabSizeMismatch { tokens, rows } => write!(
                f,
                "tokenizer has {tokens} tokens but token_embd.weight has {rows} rows"
            ),
            GgufError::TensorNotFound(n) => write!(f, "tensor {n:?} not found"),
            GgufError::InvalidTensorShape { name, dims } => write!(
                f,
                "tensor {name:?} has shape {dims:?}, which cannot be extended by rows"
            ),
            GgufError::InvalidTensorData {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} declares {expected} elements but got {actual}"
            ),
            GgufError::RowLengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} expects rows of {expected} elements, got {actual}"
            ),
            GgufError::InvalidElementCount {
                name,
                elements,
                block_elems,
            } => write!(
                f,
                "tensor {name:?} has {elements} elements, not a multiple of the block size {block_elems}"
            ),
            GgufError::RowOutOfBounds { name, row, rows } => {
                write!(f, "row {row} out of bounds for {name:?} ({rows} rows)")
            }
            GgufError::OutputTooSmall {
                name,
                expected,
                actual,
            } => write!(
                f,
                "output slice too small for {name:?}: need {expected} f32, got {actual}"
            ),
            GgufError::UnsupportedQuantType { name, kind } => write!(
                f,
                "tensor {name:?} uses unsupported quantization type {kind}"
            ),
            GgufError::InvalidRowLength {
                name,
                row_len,
                block_elems,
            } => write!(
                f,
                "tensor {name:?} row length {row_len} is not a multiple of the block size {block_elems}"
            ),
            GgufError::Truncated {
                name,
                offset,
                bytes,
                file_len,
            } => write!(
                f,
                "tensor {name:?} needs bytes {offset}..{} but the file is {file_len} bytes",
                offset.saturating_add(*bytes)
            ),
            GgufError::BufferTooSmall { needed, actual } => write!(
                f,
                "byte buffer too small: need {needed} bytes, got {actual}"
            ),
            GgufError::MaxElemsExceeded {
                name,
                requested,
                max,
            } => write!(
                f,
                "tensor {name:?} has {requested} elements, above the limit {max}"
            ),
            GgufError::SizeOverflow { name } => {
                write!(f, "size arithmetic overflow for tensor {name:?}")
            }
            GgufError::InvalidAlignment(a) => write!(f, "invalid GGUF alignment {a}"),
        }
    }
}

impl Error for GgufError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        GgufError::Io(value)
    }
}

/// A GGUF metadata value.
///
/// The variants mirror the scalar types of the GGUF specification plus
/// arrays. `Array` is recursive because GGUF arrays carry their element type
/// on the wire.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    /// GGUF arrays carry their element type on the wire, including when the
    /// array is empty or when its elements are themselves arrays.
    Array {
        elem_type: u32,
        items: Vec<GgufValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// GGUF type id of this value.
    pub fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array { .. } => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }

    /// Interpret an unsigned metadata value as `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(u64::from(*v)),
            GgufValue::U16(v) => Some(u64::from(*v)),
            GgufValue::U32(v) => Some(u64::from(*v)),
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Interpret a signed metadata value as `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufValue::I8(v) => Some(i64::from(*v)),
            GgufValue::I16(v) => Some(i64::from(*v)),
            GgufValue::I32(v) => Some(i64::from(*v)),
            GgufValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Borrow the value as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Construct an array value with an explicit wire element type.
    pub fn array(elem_type: u32, items: Vec<GgufValue>) -> Self {
        GgufValue::Array { elem_type, items }
    }

    /// Borrow the value as an array.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array { items, .. } => Some(items),
            _ => None,
        }
    }

    /// Borrow the array items for mutation.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<GgufValue>> {
        match self {
            GgufValue::Array { items, .. } => Some(items),
            _ => None,
        }
    }

    /// Element type id of an array value.
    pub fn array_elem_type(&self) -> Option<u32> {
        match self {
            GgufValue::Array { elem_type, .. } => Some(*elem_type),
            _ => None,
        }
    }

    /// Interpret a numeric metadata value as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::U8(v) => Some(f64::from(*v)),
            GgufValue::I8(v) => Some(f64::from(*v)),
            GgufValue::U16(v) => Some(f64::from(*v)),
            GgufValue::I16(v) => Some(f64::from(*v)),
            GgufValue::U32(v) => Some(f64::from(*v)),
            GgufValue::I32(v) => Some(f64::from(*v)),
            GgufValue::F32(v) => Some(f64::from(*v)),
            GgufValue::U64(v) => Some(*v as f64),
            GgufValue::I64(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Human-readable name of the value type.
    pub fn type_name(&self) -> &'static str {
        match self {
            GgufValue::U8(_) => "u8",
            GgufValue::I8(_) => "i8",
            GgufValue::U16(_) => "u16",
            GgufValue::I16(_) => "i16",
            GgufValue::U32(_) => "u32",
            GgufValue::I32(_) => "i32",
            GgufValue::F32(_) => "f32",
            GgufValue::Bool(_) => "bool",
            GgufValue::String(_) => "string",
            GgufValue::Array { .. } => "array",
            GgufValue::U64(_) => "u64",
            GgufValue::I64(_) => "i64",
            GgufValue::F64(_) => "f64",
        }
    }
}

/// Descriptor of one tensor in the tensor table.
///
/// `dims` keeps the on-disk order: `dims[0]` is the fastest-varying
/// dimension, i.e. the number of elements in one row. This is the opposite of
/// the `shape` printed by `gguf-py`, which reverses the list for NumPy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
}

impl TensorInfo {
    /// Elements in one row (`dims[0]`).
    pub fn row_len(&self) -> Option<usize> {
        if self.dims.is_empty() {
            return Some(1);
        }
        usize::try_from(self.dims[0]).ok()
    }

    /// Number of rows, i.e. the product of the dimensions after the first.
    pub fn rows(&self) -> Option<usize> {
        if self.dims.len() <= 1 {
            return Some(1);
        }
        let mut rows: usize = 1;
        for &d in &self.dims[1..] {
            rows = rows.checked_mul(usize::try_from(d).ok()?)?;
        }
        Some(rows)
    }

    /// Total number of elements in the tensor.
    pub fn elements(&self) -> Option<usize> {
        if self.dims.is_empty() {
            return Some(1);
        }
        let mut n: usize = 1;
        for &d in &self.dims {
            n = n.checked_mul(usize::try_from(d).ok()?)?;
        }
        Some(n)
    }
}

// ---------------------------------------------------------------------------
// Quantization type ids and block geometry.
//
// Type ids and block sizes are copied from llama.cpp `gguf-py/gguf/
// constants.py` (`GGMLQuantizationType` and `GGML_QUANT_SIZES`).
// ---------------------------------------------------------------------------

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
pub const GGML_TYPE_IQ2_XS: u32 = 17;
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
pub const GGML_TYPE_IQ1_S: u32 = 19;
pub const GGML_TYPE_IQ4_NL: u32 = 20;
pub const GGML_TYPE_IQ3_S: u32 = 21;
pub const GGML_TYPE_IQ2_S: u32 = 22;
pub const GGML_TYPE_IQ4_XS: u32 = 23;
pub const GGML_TYPE_I8: u32 = 24;
pub const GGML_TYPE_I16: u32 = 25;
pub const GGML_TYPE_I32: u32 = 26;
pub const GGML_TYPE_I64: u32 = 27;
pub const GGML_TYPE_F64: u32 = 28;
pub const GGML_TYPE_IQ1_M: u32 = 29;
pub const GGML_TYPE_BF16: u32 = 30;
pub const GGML_TYPE_TQ1_0: u32 = 34;
pub const GGML_TYPE_TQ2_0: u32 = 35;
pub const GGML_TYPE_MXFP4: u32 = 39;
pub const GGML_TYPE_NVFP4: u32 = 40;
pub const GGML_TYPE_Q1_0: u32 = 41;

/// Elements per K-quant super-block.
pub const QK_K: usize = 256;
/// Elements per legacy (non-K) 32-value block.
pub const QK8_0: usize = 32;

/// Bytes of one F16 value.
pub const F16_BYTES: usize = 2;
/// Bytes of one F32 value.
pub const F32_BYTES: usize = 4;

/// `(elements per block, bytes per block)` for every type this crate decodes.
pub fn quant_block(kind: u32) -> Option<(usize, usize)> {
    match kind {
        GGML_TYPE_F32 => Some((1, 4)),
        GGML_TYPE_F16 => Some((1, 2)),
        GGML_TYPE_Q8_0 => Some((32, 34)),
        GGML_TYPE_Q4_K => Some((QK_K, 2 + 2 + 12 + QK_K / 2)),
        GGML_TYPE_Q5_K => Some((QK_K, 2 + 2 + 12 + QK_K / 8 + QK_K / 2)),
        GGML_TYPE_Q6_K => Some((QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)),
        _ => None,
    }
}

/// `(elements per block, bytes per block)` for every type in the GGUF
/// `GGML_QUANT_SIZES` table.
///
/// The writer needs this for tensors it copies but does not decode: an IQ3_XXS
/// tensor must keep its bytes even though this crate has no IQ3_XXS
/// dequantizer.
pub fn quant_geometry(kind: u32) -> Option<(usize, usize)> {
    match kind {
        GGML_TYPE_F32 => Some((1, 4)),
        GGML_TYPE_F16 => Some((1, 2)),
        GGML_TYPE_Q4_0 => Some((32, 18)),
        GGML_TYPE_Q4_1 => Some((32, 20)),
        GGML_TYPE_Q5_0 => Some((32, 22)),
        GGML_TYPE_Q5_1 => Some((32, 24)),
        GGML_TYPE_Q8_0 => Some((32, 34)),
        GGML_TYPE_Q8_1 => Some((32, 40)),
        GGML_TYPE_Q2_K => Some((QK_K, 84)),
        GGML_TYPE_Q3_K => Some((QK_K, 110)),
        GGML_TYPE_Q4_K => Some((QK_K, 144)),
        GGML_TYPE_Q5_K => Some((QK_K, 176)),
        GGML_TYPE_Q6_K => Some((QK_K, 210)),
        GGML_TYPE_Q8_K => Some((QK_K, 292)),
        GGML_TYPE_IQ2_XXS => Some((QK_K, 66)),
        GGML_TYPE_IQ2_XS => Some((QK_K, 74)),
        GGML_TYPE_IQ3_XXS => Some((QK_K, 98)),
        GGML_TYPE_IQ1_S => Some((QK_K, 50)),
        GGML_TYPE_IQ4_NL => Some((32, 18)),
        GGML_TYPE_IQ3_S => Some((QK_K, 110)),
        GGML_TYPE_IQ2_S => Some((QK_K, 82)),
        GGML_TYPE_IQ4_XS => Some((QK_K, 136)),
        GGML_TYPE_I8 => Some((1, 1)),
        GGML_TYPE_I16 => Some((1, 2)),
        GGML_TYPE_I32 => Some((1, 4)),
        GGML_TYPE_I64 => Some((1, 8)),
        GGML_TYPE_F64 => Some((1, 8)),
        GGML_TYPE_IQ1_M => Some((QK_K, 56)),
        GGML_TYPE_BF16 => Some((1, 2)),
        GGML_TYPE_TQ1_0 => Some((QK_K, 54)),
        GGML_TYPE_TQ2_0 => Some((QK_K, 66)),
        GGML_TYPE_MXFP4 => Some((32, 17)),
        GGML_TYPE_NVFP4 => Some((64, 36)),
        GGML_TYPE_Q1_0 => Some((128, 18)),
        _ => None,
    }
}

/// Name of a GGML quantization type, or `"unknown"`.
pub fn quant_name(kind: u32) -> &'static str {
    match kind {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q8_1 => "Q8_1",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_IQ2_XXS => "IQ2_XXS",
        GGML_TYPE_IQ2_XS => "IQ2_XS",
        GGML_TYPE_IQ3_XXS => "IQ3_XXS",
        GGML_TYPE_IQ1_S => "IQ1_S",
        GGML_TYPE_IQ4_NL => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        GGML_TYPE_IQ2_S => "IQ2_S",
        GGML_TYPE_IQ4_XS => "IQ4_XS",
        GGML_TYPE_I8 => "I8",
        GGML_TYPE_I16 => "I16",
        GGML_TYPE_I32 => "I32",
        GGML_TYPE_I64 => "I64",
        GGML_TYPE_F64 => "F64",
        GGML_TYPE_IQ1_M => "IQ1_M",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_TQ1_0 => "TQ1_0",
        GGML_TYPE_TQ2_0 => "TQ2_0",
        GGML_TYPE_MXFP4 => "MXFP4",
        GGML_TYPE_NVFP4 => "NVFP4",
        GGML_TYPE_Q1_0 => "Q1_0",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_geometry_matches_gguf_py() {
        // GGML_QUANT_SIZES from gguf-py/gguf/constants.py.
        assert_eq!(quant_block(GGML_TYPE_F32), Some((1, 4)));
        assert_eq!(quant_block(GGML_TYPE_F16), Some((1, 2)));
        assert_eq!(quant_block(GGML_TYPE_Q8_0), Some((32, 34)));
        assert_eq!(quant_block(GGML_TYPE_Q4_K), Some((256, 144)));
        assert_eq!(quant_block(GGML_TYPE_Q5_K), Some((256, 176)));
        assert_eq!(quant_block(GGML_TYPE_Q6_K), Some((256, 210)));
        assert_eq!(quant_block(GGML_TYPE_IQ3_XXS), None);
    }

    #[test]
    fn writer_geometry_covers_all_quant_types() {
        assert_eq!(quant_geometry(GGML_TYPE_Q8_1), Some((32, 40)));
        assert_eq!(quant_geometry(GGML_TYPE_Q2_K), Some((256, 84)));
        assert_eq!(quant_geometry(GGML_TYPE_Q8_K), Some((256, 292)));
        assert_eq!(quant_geometry(GGML_TYPE_IQ3_XXS), Some((256, 98)));
        assert_eq!(quant_geometry(GGML_TYPE_IQ4_XS), Some((256, 136)));
        assert_eq!(quant_geometry(GGML_TYPE_MXFP4), Some((32, 17)));
        assert_eq!(quant_geometry(GGML_TYPE_NVFP4), Some((64, 36)));
        assert_eq!(quant_geometry(GGML_TYPE_Q1_0), Some((128, 18)));
        assert_eq!(quant_geometry(4), None);
    }

    #[test]
    fn array_value_keeps_element_type() {
        let value = GgufValue::array(8, vec![GgufValue::String("a".into())]);
        assert_eq!(value.type_id(), 9);
        assert_eq!(value.array_elem_type(), Some(8));
        assert_eq!(value.as_array().map(|items| items.len()), Some(1));
        let empty = GgufValue::array(5, Vec::new());
        assert_eq!(empty.array_elem_type(), Some(5));
        assert_eq!(empty.as_array().map(|items| items.len()), Some(0));
    }

    #[test]
    fn tensor_info_uses_raw_dim_order() {
        let t = TensorInfo {
            name: "token_embd.weight".into(),
            dims: vec![1024, 65536],
            kind: GGML_TYPE_Q4_K,
            offset: 0,
        };
        assert_eq!(t.row_len(), Some(1024));
        assert_eq!(t.rows(), Some(65536));
        assert_eq!(t.elements(), Some(1024 * 65536));
    }
}
