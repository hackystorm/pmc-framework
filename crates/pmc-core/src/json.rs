// SPDX-License-Identifier: GPL-3.0-only

//! A minimal JSON writer.
//!
//! The trainer produces manifests and the FFI layer returns metrics. Both
//! need JSON, and neither needs a parser, so the crate carries a twenty-line
//! serialiser instead of a dependency.

#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn str(value: impl Into<String>) -> Json {
        Json::Str(value.into())
    }

    pub fn num(value: f64) -> Json {
        Json::Num(value)
    }

    pub fn obj(entries: Vec<(&str, Json)>) -> Json {
        Json::Obj(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::Num(value) => {
                if value.is_finite() {
                    out.push_str(&format!("{}", value));
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(value) => write_escaped(value, out),
            Json::Arr(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_escaped(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_escaped(value: &str, out: &mut String) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_render_as_expected() {
        let value = Json::obj(vec![
            ("name", Json::str("test")),
            ("count", Json::num(3.0)),
            ("flag", Json::Bool(true)),
        ]);
        assert_eq!(value.to_string(), r#"{"name":"test","count":3,"flag":true}"#);
    }

    #[test]
    fn strings_are_escaped() {
        let value = Json::str("a\"b\nc");
        assert_eq!(value.to_string(), r#""a\"b\nc""#);
    }

    #[test]
    fn non_finite_numbers_become_null() {
        let value = Json::num(f64::NAN);
        assert_eq!(value.to_string(), "null");
    }
}
