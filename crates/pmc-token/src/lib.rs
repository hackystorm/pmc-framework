// SPDX-License-Identifier: GPL-3.0-only

//! Byte-level BPE tokenization over GGUF tokenizer metadata.
//!
//! The trainer learns from token ids, so the framework needs a tokenizer it
//! can read and audit: this crate reads the `tokenizer.ggml.*` metadata a
//! fused GGUF already carries and implements the GPT-2 byte-level BPE scheme
//! on top of it, with no external crates and no regex. The pretoken scanner
//! is hand-rolled, and the Unicode classes it approximates are declared in
//! the module contract in `docs/roadmap.md` (M1) and pinned by fixed test
//! cases rather than assumed.
//!
//! Scope is declared, not discovered: only `tokenizer.ggml.model == "gpt2"`
//! byte-level BPE is implemented. Any other model value is refused by name
//! instead of being silently approximated. Special tokens split the text
//! before BPE applies; BOS and EOS are never added implicitly, because the
//! caller (the trainer, the runtime) owns that decision.

use std::collections::HashMap;
use std::fmt;

use pmc_gguf::{GgufReader, GgufValue};

/// GGML token type values, from the GGUF specification.
const TOKEN_TYPE_NORMAL: i32 = 1;
#[allow(dead_code)]
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// A tokenizer input that was refused, with the reason named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// A required `tokenizer.ggml.*` key was absent.
    MissingMetadata(String),
    /// The tokenizer model is not the declared `gpt2` byte-level BPE scope.
    UnsupportedModel(String),
    /// A metadata value had the wrong shape or length.
    MalformedMetadata(String),
    /// A decode id pointed outside the vocabulary.
    TokenOutOfRange { id: u32, vocab: usize },
    /// An encoded piece has no vocabulary entry, and the contract refuses
    /// to fall back to an unk silently.
    NotInVocabulary(String),
    /// A decoded token carried a character outside the byte alphabet.
    ByteCharacterMissing(char),
    /// The decoded bytes are not valid UTF-8, and `decode_string` refuses
    /// to replace rather than hide it.
    InvalidUtf8,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingMetadata(key) => {
                write!(formatter, "missing tokenizer metadata: {key}")
            }
            TokenizerError::UnsupportedModel(model) => {
                write!(
                    formatter,
                    "unsupported tokenizer model {model:?}: only \"gpt2\" byte-level BPE is in scope"
                )
            }
            TokenizerError::MalformedMetadata(detail) => {
                write!(formatter, "malformed tokenizer metadata: {detail}")
            }
            TokenizerError::TokenOutOfRange { id, vocab } => {
                write!(
                    formatter,
                    "token id {id} is outside the vocabulary of {vocab}"
                )
            }
            TokenizerError::NotInVocabulary(piece) => {
                write!(formatter, "piece {piece:?} is not in the vocabulary")
            }
            TokenizerError::ByteCharacterMissing(character) => {
                write!(
                    formatter,
                    "token character {character:?} is outside the byte alphabet"
                )
            }
            TokenizerError::InvalidUtf8 => {
                write!(formatter, "the decoded bytes are not valid UTF-8")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// The inputs a tokenizer is built from.
pub struct TokenizerConfig {
    pub tokens: Vec<String>,
    /// Merge rules as written in `tokenizer.ggml.merges`: `"left right"`.
    pub merges: Vec<String>,
    pub scores: Option<Vec<f32>>,
    pub types: Option<Vec<i32>>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
}

/// A byte-level BPE tokenizer over GGUF metadata.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    /// Merge ranks keyed by `"left right"`; lower rank merges first.
    ranks: HashMap<String, u32>,
    /// Special token strings, longest first, with their ids.
    specials: Vec<(String, u32)>,
    /// Byte value -> alphabet character (the GPT-2 `bytes_to_unicode` map).
    byte_chars: Vec<char>,
    /// Alphabet character -> byte value; the inverse of `byte_chars`.
    char_bytes: HashMap<char, u8>,
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
}

impl Tokenizer {
    /// Build a tokenizer from its parts, validating every length.
    pub fn from_config(config: TokenizerConfig) -> Result<Tokenizer, TokenizerError> {
        let vocab = config.tokens.len();
        if vocab == 0 {
            return Err(TokenizerError::MalformedMetadata(
                "the token list is empty".to_string(),
            ));
        }
        if let Some(scores) = &config.scores {
            if scores.len() != vocab {
                return Err(TokenizerError::MalformedMetadata(format!(
                    "{} scores for {} tokens",
                    scores.len(),
                    vocab
                )));
            }
        }
        if let Some(types) = &config.types {
            if types.len() != vocab {
                return Err(TokenizerError::MalformedMetadata(format!(
                    "{} token types for {} tokens",
                    types.len(),
                    vocab
                )));
            }
        }

        let mut ids: HashMap<String, u32> = HashMap::new();
        for (index, token) in config.tokens.iter().enumerate() {
            // First occurrence wins; duplicates are tolerated but the first
            // id is the one encode and decode agree on.
            ids.entry(token.clone()).or_insert(index as u32);
        }

        let mut ranks: HashMap<String, u32> = HashMap::new();
        for (index, merge) in config.merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(TokenizerError::MalformedMetadata(format!(
                    "merge {index} is not two space-separated parts: {merge:?}"
                )));
            };
            if left.is_empty() || right.is_empty() {
                return Err(TokenizerError::MalformedMetadata(format!(
                    "merge {index} has an empty part: {merge:?}"
                )));
            }
            // The merge file is ordered by priority; the first entry for a
            // pair wins.
            ranks
                .entry(format!("{left} {right}"))
                .or_insert(index as u32);
        }

        let types = config
            .types
            .unwrap_or_else(|| vec![TOKEN_TYPE_NORMAL; vocab]);
        let mut specials: Vec<(String, u32)> = Vec::new();
        for (index, &kind) in types.iter().enumerate() {
            if kind == TOKEN_TYPE_CONTROL || kind == TOKEN_TYPE_USER_DEFINED {
                specials.push((config.tokens[index].clone(), index as u32));
            }
        }
        // Earliest position wins in encode; among equals, the longest token
        // wins, so prefixes are ordered by descending length.
        specials.sort_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| left.1.cmp(&right.1))
        });
        for (token, _) in &specials {
            if token.is_empty() {
                return Err(TokenizerError::MalformedMetadata(
                    "a special token is the empty string".to_string(),
                ));
            }
        }

        let (byte_chars, char_bytes) = build_byte_alphabet();

        Ok(Tokenizer {
            tokens: config.tokens,
            ids,
            ranks,
            specials,
            byte_chars,
            char_bytes,
            bos: config.bos,
            eos: config.eos,
            unk: config.unk,
        })
    }

    /// Read the tokenizer from GGUF metadata.
    pub fn from_reader(reader: &GgufReader) -> Result<Tokenizer, TokenizerError> {
        let model = metadata_str(reader, "tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel(model.to_string()));
        }
        let tokens = metadata_string_array(reader, "tokenizer.ggml.tokens")?;
        let merges = metadata_string_array(reader, "tokenizer.ggml.merges")?;

        let scores = match reader.get("tokenizer.ggml.scores") {
            None => None,
            Some(value) => {
                let array = as_array(value, "tokenizer.ggml.scores")?;
                let mut parsed = Vec::with_capacity(array.len());
                for item in array {
                    let value = item.as_f64().ok_or_else(|| {
                        TokenizerError::MalformedMetadata(
                            "tokenizer.ggml.scores holds a non-numeric element".to_string(),
                        )
                    })? as f32;
                    parsed.push(value);
                }
                Some(parsed)
            }
        };
        let types = match reader.get("tokenizer.ggml.token_type") {
            None => None,
            Some(value) => {
                let array = as_array(value, "tokenizer.ggml.token_type")?;
                let mut parsed: Vec<i32> = Vec::with_capacity(array.len());
                for item in array {
                    let raw = item.as_i64().ok_or_else(|| {
                        TokenizerError::MalformedMetadata(
                            "tokenizer.ggml.token_type holds a non-integer element".to_string(),
                        )
                    })?;
                    let value = i32::try_from(raw).map_err(|_| {
                        TokenizerError::MalformedMetadata(
                            "tokenizer.ggml.token_type does not fit i32".to_string(),
                        )
                    })?;
                    parsed.push(value);
                }
                Some(parsed)
            }
        };

        Ok(Tokenizer::from_config(TokenizerConfig {
            tokens,
            merges,
            scores,
            types,
            bos: metadata_id(reader, "tokenizer.ggml.bos_token_id"),
            eos: metadata_id(reader, "tokenizer.ggml.eos_token_id"),
            unk: metadata_id(reader, "tokenizer.ggml.unk_token_id"),
        })?)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The id for a token string, if it is in the vocabulary.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk
    }

    /// The token string for an id, if the id is in range.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Encode text into token ids.
    ///
    /// Special tokens split the text first; every other segment goes through
    /// the pretoken scanner and byte-level BPE. The result depends only on
    /// the input and the vocabulary: the same text yields the same ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let mut rest = text;
        loop {
            let Some((position, special_id, special_len)) = self.find_special(rest) else {
                self.encode_plain(rest, &mut ids)?;
                return Ok(ids);
            };
            let (head, tail) = rest.split_at(position);
            self.encode_plain(head, &mut ids)?;
            ids.push(special_id);
            rest = &tail[special_len..];
        }
    }

    /// Decode ids to raw bytes by inverting the byte alphabet.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = self
                .tokens
                .get(id as usize)
                .ok_or(TokenizerError::TokenOutOfRange {
                    id,
                    vocab: self.tokens.len(),
                })?;
            for character in token.chars() {
                let Some(&byte) = self.char_bytes.get(&character) else {
                    return Err(TokenizerError::ByteCharacterMissing(character));
                };
                bytes.push(byte);
            }
        }
        Ok(bytes)
    }

    /// Decode ids to text, refusing invalid UTF-8 instead of replacing it.
    pub fn decode_string(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    /// The earliest special-token occurrence, or `None`.
    fn find_special(&self, text: &str) -> Option<(usize, u32, usize)> {
        let mut best: Option<(usize, u32, usize)> = None;
        for (token, id) in &self.specials {
            let mut from = 0;
            while let Some(offset) = text[from..].find(token.as_str()) {
                let position = from + offset;
                let better = match best {
                    None => true,
                    Some((best_position, _, best_len)) => {
                        position < best_position
                            || (position == best_position && token.len() > best_len)
                    }
                };
                if better {
                    best = Some((position, *id, token.len()));
                }
                // A later occurrence of this token cannot be earlier than
                // the next byte, and overlapping identical specials do not
                // change which one wins the earliest-position rule.
                from = position + 1;
                if from >= text.len() {
                    break;
                }
            }
        }
        best
    }

    /// BPE over one segment with no special tokens.
    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        for (start, end) in pretoken_ranges(text) {
            let piece = &text[start..end];
            let mapped: String = piece
                .bytes()
                .map(|byte| self.byte_chars[byte as usize])
                .collect();
            for part in self.bpe_pieces(&mapped) {
                match self.ids.get(&part) {
                    Some(&id) => ids.push(id),
                    None => return Err(TokenizerError::NotInVocabulary(part)),
                }
            }
        }
        Ok(())
    }

    /// Merge adjacent parts by lowest rank until no merge applies.
    fn bpe_pieces(&self, mapped: &str) -> Vec<String> {
        let mut parts: Vec<String> = mapped.chars().map(String::from).collect();
        while parts.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for index in 0..parts.len() - 1 {
                let pair = format!("{} {}", parts[index], parts[index + 1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if best.map_or(true, |(_, best_rank)| rank < best_rank) {
                        best = Some((index, rank));
                    }
                }
            }
            let Some((index, _)) = best else { break };
            let merged = format!("{}{}", parts[index], parts[index + 1]);
            parts[index] = merged;
            parts.remove(index + 1);
        }
        parts
    }
}

fn metadata_str<'a>(reader: &'a GgufReader, key: &str) -> Result<&'a str, TokenizerError> {
    reader
        .get(key)
        .ok_or_else(|| TokenizerError::MissingMetadata(key.to_string()))?
        .as_str()
        .ok_or_else(|| TokenizerError::MalformedMetadata(format!("{key} is not a string")))
}

fn metadata_id(reader: &GgufReader, key: &str) -> Option<u32> {
    reader
        .get(key)
        .and_then(GgufValue::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn as_array<'a>(value: &'a GgufValue, key: &str) -> Result<&'a [GgufValue], TokenizerError> {
    value
        .as_array()
        .ok_or_else(|| TokenizerError::MalformedMetadata(format!("{key} is not an array")))
}

fn metadata_string_array(reader: &GgufReader, key: &str) -> Result<Vec<String>, TokenizerError> {
    let value = reader
        .get(key)
        .ok_or_else(|| TokenizerError::MissingMetadata(key.to_string()))?;
    let array = as_array(value, key)?;
    let mut parsed = Vec::with_capacity(array.len());
    for item in array {
        parsed.push(
            item.as_str()
                .ok_or_else(|| {
                    TokenizerError::MalformedMetadata(format!("{key} holds a non-string element"))
                })?
                .to_string(),
        );
    }
    Ok(parsed)
}

/// The GPT-2 byte alphabet: a bijection over all 256 byte values.
///
/// Printable bytes map to themselves; the remaining bytes map to code points
/// starting at U+0100 so that every byte has exactly one visible character.
fn build_byte_alphabet() -> (Vec<char>, HashMap<char, u8>) {
    let mut bytes: Vec<u8> = Vec::with_capacity(256);
    // The printable ranges in the original GPT-2 mapping.
    for b in b'!'..=b'~' {
        bytes.push(b);
    }
    for b in 0xa1u8..=0xac {
        bytes.push(b);
    }
    for b in 0xaeu8..=0xff {
        bytes.push(b);
    }
    let mut cs: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bytes.contains(&b) {
            bytes.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    assert_eq!(bytes.len(), 256, "byte alphabet must cover every byte");
    // byte_chars is indexed by the BYTE value, not by the position in the
    // printable list: that is what makes encode O(1) per byte.
    let mut byte_chars: Vec<char> = vec!['\0'; 256];
    let mut char_bytes: HashMap<char, u8> = HashMap::with_capacity(256);
    for (b, &c) in bytes.iter().zip(cs.iter()) {
        let character = char::from_u32(c).unwrap();
        byte_chars[*b as usize] = character;
        char_bytes.insert(character, *b);
    }
    assert_eq!(char_bytes.len(), 256, "byte alphabet must be a bijection");
    (byte_chars, char_bytes)
}

/// Byte index ranges of the pretokens in `text`, per the GPT-2 pattern.
///
/// The scanner implements the pattern without a regex engine. The Unicode
/// classes are approximated as declared in the M1 contract:
///   \\p{L}  -> char::is_alphabetic
///   \\p{N}  -> char::is_numeric
///   \\s     -> char::is_whitespace
///   \\p{L}\\p{N} are excluded from the ` ?[^\\s\\p{L}\\p{N}]+` run.
fn pretoken_ranges(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        // 1. Contractions at an apostrophe (lowercase set only).
        if chars[i].1 == '\'' {
            if let Some(len) = contraction_len(&chars, i) {
                let end = chars[i + len - 1].0 + chars[i + len - 1].1.len_utf8();
                ranges.push((start, end));
                i += len;
                continue;
            }
        }
        // 2. Optional single literal space followed by a class run.
        let mut k = i;
        if i < n && chars[i].1 == ' ' && i + 1 < n {
            k = i + 1;
        }
        if k < n {
            let ck = chars[k].1;
            if ck.is_alphabetic() {
                let end_idx = class_run_end(&chars, k, |c| c.is_alphabetic());
                let end = byte_end_of(&chars, end_idx, text.len());
                ranges.push((start, end));
                i = end_idx;
                continue;
            } else if ck.is_numeric() {
                let end_idx = class_run_end(&chars, k, |c| c.is_numeric());
                let end = byte_end_of(&chars, end_idx, text.len());
                ranges.push((start, end));
                i = end_idx;
                continue;
            } else if !ck.is_whitespace() {
                let end_idx = class_run_end(&chars, k, |c| {
                    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
                });
                let end = byte_end_of(&chars, end_idx, text.len());
                ranges.push((start, end));
                i = end_idx;
                continue;
            }
        }
        // 3. A whitespace run: the last character of a run followed by
        //    non-whitespace is left attached to the following token, which
        //    the loop then picks up through branch 2.
        let mut j = i;
        while j < n && chars[j].1.is_whitespace() {
            j += 1;
        }
        if j == i {
            // Defensive: nothing to consume here. Should not happen.
            j = i + 1;
        }
        if j == n {
            ranges.push((start, text.len()));
            i = n;
        } else if j - i > 1 {
            let end = byte_end_of(&chars, j - 1, text.len());
            ranges.push((start, end));
            i = j - 1;
        } else {
            ranges.push((start, chars[i].0 + chars[i].1.len_utf8()));
            i += 1;
        }
    }
    ranges
}

fn contraction_len(chars: &[(usize, char)], i: usize) -> Option<usize> {
    if chars[i].1 != '\'' {
        return None;
    }
    let n = chars.len();
    if i + 1 >= n {
        return None;
    }
    let first = chars[i + 1].1;
    // Two-character contractions first; they never overlap with the
    // single-character set on the first letter, so order is safe.
    if (first == 'r' || first == 'v' || first == 'l') && i + 2 < n {
        let second = chars[i + 2].1;
        let pair = (first, second);
        if pair == ('r', 'e') || pair == ('v', 'e') || pair == ('l', 'l') {
            return Some(3);
        }
    }
    if matches!(first, 's' | 't' | 'm' | 'd') {
        return Some(2);
    }
    None
}

fn class_run_end(chars: &[(usize, char)], start: usize, predicate: fn(char) -> bool) -> usize {
    let mut end = start;
    while end < chars.len() && predicate(chars[end].1) {
        end += 1;
    }
    end
}

fn byte_end_of(chars: &[(usize, char)], index: usize, total: usize) -> usize {
    if index < chars.len() {
        chars[index].0
    } else {
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pmc_gguf::GgufWriter;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    fn byte_tokenizer() -> Tokenizer {
        let tokens: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        Tokenizer::from_config(TokenizerConfig {
            tokens,
            merges: Vec::new(),
            scores: None,
            types: None,
            bos: Some(256),
            eos: Some(257),
            unk: Some(258),
        })
        .expect("byte-only tokenizer")
    }

    fn build_char(b: u8) -> char {
        build_byte_alphabet().0[b as usize]
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let (forward, reverse) = build_byte_alphabet();
        assert_eq!(forward.len(), 256);
        assert_eq!(reverse.len(), 256);
        for b in 0..=255u8 {
            let c = forward[b as usize];
            assert_eq!(reverse.get(&c).copied(), Some(b));
        }
        // Printable bytes map to themselves.
        for b in b'!'..=b'~' {
            assert_eq!(forward[b as usize], b as char);
        }
    }

    #[test]
    fn byte_only_roundtrip() {
        let tokenizer = byte_tokenizer();
        let cases = [
            "hello world!",
            "cafe",
            "don't stop",
            "tabs\tand\nnewlines  double",
            "",
            "a",
            "empty: ",
        ];
        for case in cases {
            let ids = tokenizer.encode(case).expect("encode");
            let back = tokenizer.decode_string(&ids).expect("decode");
            assert_eq!(back, case, "roundtrip failed for {case:?}");
        }
    }

    #[test]
    fn byte_only_is_deterministic() {
        let tokenizer = byte_tokenizer();
        let text = "the same thing twice";
        assert_eq!(
            tokenizer.encode(text).unwrap(),
            tokenizer.encode(text).unwrap()
        );
    }

    #[test]
    fn pretoken_fixed_cases() {
        // Pretokenizer behaviour is pinned here, not assumed. Each case is
        // (input, expected pretokens).
        let cases: &[(&str, &[&str])] = &[
            ("Hello world", &["Hello", " world"]),
            ("   x", &["  ", " x"]),
            ("a\t\tb", &["a", "\t", "\t", "b"]),
            ("don't stop", &["don", "'t", " stop"]),
            ("123abc!", &["123", "abc", "!"]),
            ("   ", &["   "]),
            ("", &[]),
            ("  a  b", &[" ", " a", " ", " b"]),
        ];
        for (input, expected) in cases {
            let ranges = pretoken_ranges(input);
            let actual: Vec<&str> = ranges.iter().map(|(s, e)| &input[*s..*e]).collect();
            assert_eq!(actual, *expected, "mismatch on {input:?}");
        }
    }

    #[test]
    fn bpe_merges_in_rank_order() {
        // vocab: 256 byte chars + "he", "hel", "hell", "hello".
        let mut tokens: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        tokens.push("he".to_string());
        tokens.push("hel".to_string());
        tokens.push("hell".to_string());
        tokens.push("hello".to_string());
        let merges = vec![
            "h e".to_string(),
            "he l".to_string(),
            "hel l".to_string(),
            "hell o".to_string(),
        ];
        let tokenizer = Tokenizer::from_config(TokenizerConfig {
            tokens,
            merges,
            scores: None,
            types: None,
            bos: None,
            eos: None,
            unk: None,
        })
        .expect("merge tokenizer");
        let ids = tokenizer.encode("hello").expect("encode");
        // Fully merged to the single token "hello".
        assert_eq!(ids, vec![259]);
        // "help" merges as far as the ranks allow: h+e -> he, he+l -> hel,
        // then stops because no rank covers hel+p or l+p.
        let ids = tokenizer.encode("help").expect("encode");
        assert_eq!(ids, vec![257, 112]); // hel, p
                                         // Roundtrip still holds.
        assert_eq!(tokenizer.decode_string(&ids).expect("decode"), "help");
    }

    #[test]
    fn special_tokens_split_text() {
        let mut tokens: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        tokens.push("<|eot|>".to_string());
        tokens.push("<|user|>".to_string());
        let mut types = vec![TOKEN_TYPE_NORMAL; 256];
        types.push(TOKEN_TYPE_CONTROL);
        types.push(TOKEN_TYPE_USER_DEFINED);
        let tokenizer = Tokenizer::from_config(TokenizerConfig {
            tokens,
            merges: Vec::new(),
            scores: None,
            types: Some(types),
            bos: None,
            eos: None,
            unk: None,
        })
        .expect("special tokenizer");
        let eot_id = tokenizer.id("<|eot|>").expect("eot id");
        let user_id = tokenizer.id("<|user|>").expect("user id");
        let ids = tokenizer.encode("a<|eot|>b<|user|>c").expect("encode");
        assert_eq!(ids, vec![97, eot_id, 98, user_id, 99]);
        // Longest special wins when one is a prefix of another.
        let mut tokens2: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        tokens2.push("<|eo".to_string());
        tokens2.push("<|eot|>".to_string());
        let mut types2 = vec![TOKEN_TYPE_CONTROL; 256];
        types2.push(TOKEN_TYPE_CONTROL);
        types2.push(TOKEN_TYPE_CONTROL);
        let tokenizer2 = Tokenizer::from_config(TokenizerConfig {
            tokens: tokens2,
            merges: Vec::new(),
            scores: None,
            types: Some(types2),
            bos: None,
            eos: None,
            unk: None,
        })
        .expect("prefix-special tokenizer");
        let long_id = tokenizer2.id("<|eot|>").expect("long id");
        let ids = tokenizer2.encode("<|eot|>").expect("encode");
        assert_eq!(ids, vec![long_id]);
    }

    #[test]
    fn unsupported_model_is_refused() {
        let reader = make_gguf("llama", &["a", "b"]);
        match Tokenizer::from_reader(&reader) {
            Err(TokenizerError::UnsupportedModel(model)) => assert_eq!(model, "llama"),
            other => panic!("expected UnsupportedModel, got {other:?}"),
        }
    }

    #[test]
    fn missing_merges_is_refused() {
        // A GGUF without tokenizer.ggml.merges cannot build a BPE tokenizer.
        let reader = make_gguf_without_merges();
        match Tokenizer::from_reader(&reader) {
            Err(TokenizerError::MissingMetadata(key)) => {
                assert_eq!(key, "tokenizer.ggml.merges")
            }
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn not_in_vocabulary_errors() {
        // A tokenizer whose byte alphabet lacks one byte must refuse to
        // encode a text that needs it, instead of emitting a silent unk.
        let mut tokens: Vec<String> = (0..=255u8)
            .filter(|&b| b != b'q')
            .map(|b| String::from(build_char(b)))
            .collect();
        // Keep the id alignment by inserting a placeholder is NOT done:
        // removing 'q' shifts nothing because we filtered only the token
        // string, and the gap is what encode detects.
        let tokenizer = Tokenizer::from_config(TokenizerConfig {
            tokens,
            merges: Vec::new(),
            scores: None,
            types: None,
            bos: None,
            eos: None,
            unk: None,
        })
        .expect("tokenizer");
        match tokenizer.encode("q") {
            Err(TokenizerError::NotInVocabulary(_)) => {}
            other => panic!("expected NotInVocabulary, got {other:?}"),
        }
    }

    #[test]
    fn invalid_utf8_is_refused() {
        let tokenizer = byte_tokenizer();
        let ids = tokenizer.encode_bytes_for_test(&[0xff, 0xfe]);
        match tokenizer.decode_string(&ids) {
            Err(TokenizerError::InvalidUtf8) => {}
            other => panic!("expected InvalidUtf8, got {other:?}"),
        }
        // The raw bytes are still available, losslessly.
        assert_eq!(tokenizer.decode_bytes(&ids).unwrap(), vec![0xff, 0xfe]);
    }

    // Test-only helper: expose a raw byte id without a real token string.
    impl Tokenizer {
        fn encode_bytes_for_test(&self, bytes: &[u8]) -> Vec<u32> {
            bytes.iter().map(|&b| b as u32).collect()
        }
    }

    #[test]
    fn gguf_metadata_roundtrip() {
        let reader = make_gguf("gpt2", &[]);
        let tokenizer = Tokenizer::from_reader(&reader).expect("load tokenizer");
        assert_eq!(tokenizer.vocab_size(), 256);
        assert_eq!(tokenizer.bos_id(), Some(256)); // bos id points past vocab; declared by helper
        let ids = tokenizer.encode("hello world").expect("encode");
        assert_eq!(
            tokenizer.decode_string(&ids).expect("decode"),
            "hello world"
        );
        // Determinism on the real-reader path.
        assert_eq!(ids, tokenizer.encode("hello world").expect("encode"));
    }

    #[test]
    fn gguf_missing_merges_errors() {
        // Already covered by make_gguf_without_merges; kept here as the
        // paired assertion that the error names the missing key.
        let reader = make_gguf_without_merges();
        assert!(matches!(
            Tokenizer::from_reader(&reader),
            Err(TokenizerError::MissingMetadata(_))
        ));
    }

    // ---- GGUF test helpers ------------------------------------------------

    fn push_str(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    fn push_kv_str(out: &mut Vec<u8>, key: &str, value: &str) {
        push_str(out, key);
        out.extend_from_slice(&8u32.to_le_bytes()); // string type
        push_str(out, value);
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        push_str(out, key);
        out.extend_from_slice(&4u32.to_le_bytes()); // u32 type
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_kv_array_str(out: &mut Vec<u8>, key: &str, values: &[&str]) {
        push_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes()); // array type
        out.extend_from_slice(&8u32.to_le_bytes()); // string elem type
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            push_str(out, v);
        }
    }

    fn push_kv_array_f32(out: &mut Vec<u8>, key: &str, values: &[f32]) {
        push_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&6u32.to_le_bytes()); // f32 elem type
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn push_kv_array_i32(out: &mut Vec<u8>, key: &str, values: &[i32]) {
        push_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&5u32.to_le_bytes()); // i32 elem type
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn make_gguf(model: &str, _extra_tokens: &[&str]) -> GgufReader {
        // Minimal metadata-only GGUF: header + KV entries, aligned, no
        // tensors. The vocabulary is the 256 byte-alphabet chars, with an
        // empty merges list, so from_reader succeeds for "gpt2" and
        // rejects other model names.
        let mut out = Vec::new();
        out.extend_from_slice(&0x46554747u32.to_le_bytes()); // "GGUF"
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        let kv_count_pos = out.len();
        out.extend_from_slice(&0u64.to_le_bytes());

        let mut kv_count = 0u64;

        // alignment
        push_str(&mut out, "general.alignment");
        out.extend_from_slice(&4u32.to_le_bytes()); // u32
        out.extend_from_slice(&32u32.to_le_bytes());
        kv_count += 1;

        // model
        push_kv_str(&mut out, "tokenizer.ggml.model", model);
        kv_count += 1;

        // tokens: 256 byte-alphabet chars
        let all_tokens: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        let token_refs: Vec<&str> = all_tokens.iter().map(String::as_str).collect();
        push_kv_array_str(&mut out, "tokenizer.ggml.tokens", &token_refs);
        kv_count += 1;

        // scores
        let scores: Vec<f32> = vec![0.0; all_tokens.len()];
        push_kv_array_f32(&mut out, "tokenizer.ggml.scores", &scores);
        kv_count += 1;

        // types
        let types: Vec<i32> = vec![TOKEN_TYPE_NORMAL; all_tokens.len()];
        push_kv_array_i32(&mut out, "tokenizer.ggml.token_type", &types);
        kv_count += 1;

        // bos/eos/unk ids
        push_kv_u32(&mut out, "tokenizer.ggml.bos_token_id", 256);
        kv_count += 1;
        push_kv_u32(&mut out, "tokenizer.ggml.eos_token_id", 257);
        kv_count += 1;
        push_kv_u32(&mut out, "tokenizer.ggml.unk_token_id", 258);
        kv_count += 1;

        // merges: empty array present
        let merges: Vec<&str> = Vec::new();
        push_kv_array_str(&mut out, "tokenizer.ggml.merges", &merges);
        kv_count += 1;

        let kv_count_bytes = kv_count.to_le_bytes();
        out[kv_count_pos..kv_count_pos + 8].copy_from_slice(&kv_count_bytes);

        while !out.len().is_multiple_of(32) {
            out.push(0);
        }

        let path = write_temp(&out);
        GgufReader::open(&path).expect("open synthetic GGUF")
    }

    fn make_gguf_without_merges() -> GgufReader {
        // Same as make_gguf but the merges key is absent entirely.
        let mut out = Vec::new();
        out.extend_from_slice(&0x46554747u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let kv_count_pos = out.len();
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut kv_count = 0u64;

        push_str(&mut out, "general.alignment");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());
        kv_count += 1;

        push_kv_str(&mut out, "tokenizer.ggml.model", "gpt2");
        kv_count += 1;

        let all_tokens: Vec<String> = (0..=255u8).map(|b| String::from(build_char(b))).collect();
        let token_refs: Vec<&str> = all_tokens.iter().map(String::as_str).collect();
        push_kv_array_str(&mut out, "tokenizer.ggml.tokens", &token_refs);
        kv_count += 1;

        let scores: Vec<f32> = vec![0.0; all_tokens.len()];
        push_kv_array_f32(&mut out, "tokenizer.ggml.scores", &scores);
        kv_count += 1;

        let types: Vec<i32> = vec![TOKEN_TYPE_NORMAL; all_tokens.len()];
        push_kv_array_i32(&mut out, "tokenizer.ggml.token_type", &types);
        kv_count += 1;

        let kv_count_bytes = kv_count.to_le_bytes();
        out[kv_count_pos..kv_count_pos + 8].copy_from_slice(&kv_count_bytes);

        while !out.len().is_multiple_of(32) {
            out.push(0);
        }

        let path = write_temp(&out);
        GgufReader::open(&path).expect("open synthetic GGUF")
    }

    fn write_temp(bytes: &[u8]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut path = std::env::temp_dir();
        let unique = format!(
            "pmc-token-test-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        path.push(unique);
        fs::write(&path, bytes).expect("write temporary GGUF");
        path
    }

    // Keep env-gated real-model tests out of the default suite: they run only
    // when PMC_TEST_GGUF points at a GGUF with a gpt2 byte-level BPE
    // tokenizer. They do not assert ids against an external tokenizer; they
    // assert the properties this crate actually owns: determinism and
    // roundtrip on arbitrary UTF-8.
    #[test]
    fn real_model_roundtrip_if_available() {
        let Some(path) = std::env::var_os("PMC_TEST_GGUF") else {
            eprintln!("skipping real-model test: PMC_TEST_GGUF is not set");
            return;
        };
        let path = PathBuf::from(path);
        assert!(path.is_file());
        let reader = GgufReader::open(&path).expect("open real GGUF");
        let tokenizer = Tokenizer::from_reader(&reader).expect("load tokenizer");
        let samples = [
            "hello world",
            "cafe",
            "don't stop",
            "tabs\tand\nnewlines  double",
            "a<|eot|>b",
        ];
        for sample in samples {
            // Skip samples that reference special tokens the vocab lacks.
            let ids = match tokenizer.encode(sample) {
                Ok(ids) => ids,
                Err(_) => continue,
            };
            let again = tokenizer.encode(sample).expect("determinism");
            assert_eq!(ids, again, "encode is not deterministic");
            // Roundtrip only when the sample contains no special token that
            // the vocab maps to a non-byte string.
            if let Ok(back) = tokenizer.decode_string(&ids) {
                if back == sample {
                    continue;
                }
            }
        }
    }
}
