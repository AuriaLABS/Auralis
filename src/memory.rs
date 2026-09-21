//! Experimental external-memory contract and deterministic in-memory reference backend.
//!
//! This module is deliberately independent from model.rs. Brain can run without
//! external memory; #42 may later consume this boundary through an opt-in adapter.

use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

pub const EXTERNAL_MEMORY_SCHEMA_VERSION: u32 = 1;
pub const MAX_MEMORY_ENTRIES: usize = 1_000_000;
pub const MAX_NAMESPACE_BYTES: usize = 256;
pub const MAX_KEY_BYTES: usize = 1024;
pub const MAX_VALUE_LEN: usize = 4096;
pub const MAX_QUERY_RESULTS: usize = 1_000_000;
pub const MAX_PERSISTED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryClass {
    Session,
    Persistent,
    Trainable,
}

impl MemoryClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Persistent => "persistent",
            Self::Trainable => "trainable",
        }
    }

    pub fn is_persisted(self) -> bool {
        !matches!(self, Self::Session)
    }
}

impl FromStr for MemoryClass {
    type Err = MemoryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "session" => Ok(Self::Session),
            "persistent" => Ok(Self::Persistent),
            "trainable" => Ok(Self::Trainable),
            other => Err(MemoryError::Format(format!(
                "unknown memory class {other}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryWrite {
    pub namespace: String,
    pub key: String,
    pub value: Vec<f32>,
    pub class: MemoryClass,
}

impl MemoryWrite {
    pub fn new(
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: Vec<f32>,
        class: MemoryClass,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            value,
            class,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRecord {
    pub id: u64,
    pub namespace: String,
    pub key: String,
    pub value: Vec<f32>,
    pub class: MemoryClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryQuery {
    pub namespace: Option<String>,
    pub key: Option<String>,
    pub class: Option<MemoryClass>,
    pub limit: usize,
}

impl MemoryQuery {
    pub fn exact(namespace: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            key: Some(key.into()),
            class: None,
            limit: 1,
        }
    }

    pub fn all(limit: usize) -> Self {
        Self {
            namespace: None,
            key: None,
            class: None,
            limit,
        }
    }

    pub fn with_class(mut self, class: MemoryClass) -> Self {
        self.class = Some(class);
        self
    }

    fn validate(&self) -> Result<(), MemoryError> {
        if self.limit == 0 || self.limit > MAX_QUERY_RESULTS {
            return Err(MemoryError::InvalidQuery(format!(
                "query limit must be in 1..={MAX_QUERY_RESULTS}"
            )));
        }
        if let Some(namespace) = &self.namespace {
            validate_namespace(namespace)?;
        }
        if let Some(key) = &self.key {
            validate_key(key)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryReset {
    Session,
    Class(MemoryClass),
    Namespace(String),
    All,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    InvalidCapacity(usize),
    AtCapacity { capacity: usize },
    InvalidNamespace(String),
    InvalidKey(String),
    InvalidValue(String),
    InvalidQuery(String),
    IdOverflow,
    TooLarge { bytes: usize, max: usize },
    Format(String),
    Io(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity(capacity) => write!(
                f,
                "memory capacity must be in 1..={MAX_MEMORY_ENTRIES}, got {capacity}"
            ),
            Self::AtCapacity { capacity } => {
                write!(f, "external memory is at capacity {capacity}")
            }
            Self::InvalidNamespace(msg) => write!(f, "invalid memory namespace: {msg}"),
            Self::InvalidKey(msg) => write!(f, "invalid memory key: {msg}"),
            Self::InvalidValue(msg) => write!(f, "invalid memory value: {msg}"),
            Self::InvalidQuery(msg) => write!(f, "invalid memory query: {msg}"),
            Self::IdOverflow => write!(f, "external memory id overflow"),
            Self::TooLarge { bytes, max } => {
                write!(f, "external memory snapshot too large: {bytes} > {max} bytes")
            }
            Self::Format(msg) => write!(f, "external memory format error: {msg}"),
            Self::Io(msg) => write!(f, "external memory I/O error: {msg}"),
        }
    }
}

impl std::error::Error for MemoryError {}

pub trait ExternalMemory {
    fn version(&self) -> u32 {
        EXTERNAL_MEMORY_SCHEMA_VERSION
    }

    fn capacity(&self) -> usize;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn write(&mut self, write: MemoryWrite) -> Result<u64, MemoryError>;
    fn read(&self, id: u64) -> Option<&MemoryRecord>;
    fn query<'a>(
        &'a self,
        query: &MemoryQuery,
    ) -> Result<Vec<&'a MemoryRecord>, MemoryError>;
    fn reset(&mut self, reset: &MemoryReset) -> Result<usize, MemoryError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct InMemoryExternalMemory {
    capacity: usize,
    next_id: u64,
    records: Vec<MemoryRecord>,
}

impl InMemoryExternalMemory {
    pub fn new(capacity: usize) -> Result<Self, MemoryError> {
        validate_capacity(capacity)?;
        Ok(Self {
            capacity,
            next_id: 0,
            records: Vec::with_capacity(capacity.min(1024)),
        })
    }

    pub fn records(&self) -> &[MemoryRecord] {
        &self.records
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn estimated_heap_bytes(&self) -> usize {
        let records = self
            .records
            .capacity()
            .saturating_mul(std::mem::size_of::<MemoryRecord>());
        self.records.iter().fold(records, |total, record| {
            total
                .saturating_add(record.namespace.capacity())
                .saturating_add(record.key.capacity())
                .saturating_add(
                    record
                        .value
                        .capacity()
                        .saturating_mul(std::mem::size_of::<f32>()),
                )
        })
    }

    pub fn encode(&self) -> Result<String, MemoryError> {
        validate_capacity(self.capacity)?;
        let persisted: Vec<&MemoryRecord> = self
            .records
            .iter()
            .filter(|record| record.class.is_persisted())
            .collect();

        let mut body = format!(
            "auralis_external_memory={}\ncapacity={}\nnext_id={}\npersisted_count={}\n",
            EXTERNAL_MEMORY_SCHEMA_VERSION,
            self.capacity,
            self.next_id,
            persisted.len(),
        );
        for record in persisted {
            validate_record(record)?;
            body.push_str("record=");
            body.push_str(&record.id.to_string());
            body.push(';');
            body.push_str(record.class.as_str());
            body.push(';');
            body.push_str(&encode_hex(record.namespace.as_bytes()));
            body.push(';');
            body.push_str(&encode_hex(record.key.as_bytes()));
            body.push(';');
            body.push_str(&encode_f32_bits(&record.value));
            body.push('\n');
        }

        let checksum = fingerprint_bytes(body.as_bytes());
        let encoded = format!("{body}checksum={checksum:016x}\n");
        if encoded.len() > MAX_PERSISTED_BYTES {
            return Err(MemoryError::TooLarge {
                bytes: encoded.len(),
                max: MAX_PERSISTED_BYTES,
            });
        }
        Ok(encoded)
    }

    pub fn decode(text: &str) -> Result<Self, MemoryError> {
        if text.len() > MAX_PERSISTED_BYTES {
            return Err(MemoryError::TooLarge {
                bytes: text.len(),
                max: MAX_PERSISTED_BYTES,
            });
        }

        let mut lines: Vec<&str> = text.lines().collect();
        let checksum_line = lines
            .pop()
            .ok_or_else(|| MemoryError::Format("missing checksum".into()))?;
        if lines.iter().any(|line| line.starts_with("checksum=")) {
            return Err(MemoryError::Format("duplicate checksum".into()));
        }
        let stored_checksum = checksum_line
            .strip_prefix("checksum=")
            .ok_or_else(|| MemoryError::Format("checksum must be the final field".into()))
            .and_then(parse_u64_hex)?;

        let mut body = lines.join("\n");
        body.push('\n');
        let computed_checksum = fingerprint_bytes(body.as_bytes());
        if stored_checksum != computed_checksum {
            return Err(MemoryError::Format(format!(
                "checksum mismatch: stored={stored_checksum:016x} computed={computed_checksum:016x}"
            )));
        }

        if lines.len() < 4 {
            return Err(MemoryError::Format("truncated memory snapshot".into()));
        }
        let version = parse_named_u32(lines[0], "auralis_external_memory")?;
        if version != EXTERNAL_MEMORY_SCHEMA_VERSION {
            return Err(MemoryError::Format(format!(
                "memory schema {version} is unsupported (expected {EXTERNAL_MEMORY_SCHEMA_VERSION})"
            )));
        }
        let capacity = parse_named_usize(lines[1], "capacity")?;
        validate_capacity(capacity)?;
        let next_id = parse_named_u64(lines[2], "next_id")?;
        let persisted_count = parse_named_usize(lines[3], "persisted_count")?;
        if persisted_count > capacity {
            return Err(MemoryError::Format(format!(
                "persisted_count {persisted_count} exceeds capacity {capacity}"
            )));
        }
        if lines.len() != 4 + persisted_count {
            return Err(MemoryError::Format(format!(
                "persisted_count mismatch: declared={persisted_count} actual={}",
                lines.len().saturating_sub(4)
            )));
        }

        let mut records = Vec::with_capacity(persisted_count);
        for line in &lines[4..] {
            let payload = line
                .strip_prefix("record=")
                .ok_or_else(|| MemoryError::Format("expected record field".into()))?;
            let parts: Vec<&str> = payload.split(';').collect();
            if parts.len() != 5 {
                return Err(MemoryError::Format(
                    "record must contain id;class;namespace;key;value".into(),
                ));
            }
            let id = parts[0]
                .parse::<u64>()
                .map_err(|_| MemoryError::Format("invalid record id".into()))?;
            let class: MemoryClass = parts[1].parse()?;
            if !class.is_persisted() {
                return Err(MemoryError::Format(
                    "session record is not valid in a persisted snapshot".into(),
                ));
            }
            let namespace = String::from_utf8(decode_hex(parts[2])?)
                .map_err(|_| MemoryError::Format("namespace is not valid UTF-8".into()))?;
            let key = String::from_utf8(decode_hex(parts[3])?)
                .map_err(|_| MemoryError::Format("key is not valid UTF-8".into()))?;
            let value = decode_f32_bits(parts[4])?;
            let record = MemoryRecord {
                id,
                namespace,
                key,
                value,
                class,
            };
            validate_record(&record)?;
            if records.iter().any(|existing: &MemoryRecord| existing.id == id) {
                return Err(MemoryError::Format(format!("duplicate record id {id}")));
            }
            records.push(record);
        }

        if let Some(max_id) = records.iter().map(|record| record.id).max() {
            if next_id <= max_id {
                return Err(MemoryError::Format(format!(
                    "next_id {next_id} must be greater than persisted record id {max_id}"
                )));
            }
        }

        Ok(Self {
            capacity,
            next_id,
            records,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), MemoryError> {
        let encoded = self.encode()?;
        fs::write(path.as_ref(), encoded)
            .map_err(|e| MemoryError::Io(format!("{}: {e}", path.as_ref().display())))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let metadata = fs::metadata(path.as_ref())
            .map_err(|e| MemoryError::Io(format!("{}: {e}", path.as_ref().display())))?;
        let bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
        if bytes > MAX_PERSISTED_BYTES {
            return Err(MemoryError::TooLarge {
                bytes,
                max: MAX_PERSISTED_BYTES,
            });
        }
        let text = fs::read_to_string(path.as_ref())
            .map_err(|e| MemoryError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::decode(&text)
    }

    pub fn snapshot_fingerprint(&self) -> Result<u64, MemoryError> {
        Ok(fingerprint_bytes(self.encode()?.as_bytes()))
    }
}

impl ExternalMemory for InMemoryExternalMemory {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn write(&mut self, write: MemoryWrite) -> Result<u64, MemoryError> {
        validate_write(&write)?;
        if self.records.len() >= self.capacity {
            return Err(MemoryError::AtCapacity {
                capacity: self.capacity,
            });
        }
        let next_id = self.next_id.checked_add(1).ok_or(MemoryError::IdOverflow)?;
        let id = self.next_id;
        self.records.push(MemoryRecord {
            id,
            namespace: write.namespace,
            key: write.key,
            value: write.value,
            class: write.class,
        });
        self.next_id = next_id;
        Ok(id)
    }

    fn read(&self, id: u64) -> Option<&MemoryRecord> {
        self.records.iter().find(|record| record.id == id)
    }

    fn query<'a>(
        &'a self,
        query: &MemoryQuery,
    ) -> Result<Vec<&'a MemoryRecord>, MemoryError> {
        query.validate()?;
        let mut out = Vec::with_capacity(query.limit.min(self.records.len()));
        for record in &self.records {
            if query
                .namespace
                .as_deref()
                .is_some_and(|namespace| namespace != record.namespace)
            {
                continue;
            }
            if query
                .key
                .as_deref()
                .is_some_and(|key| key != record.key)
            {
                continue;
            }
            if query.class.is_some_and(|class| class != record.class) {
                continue;
            }
            out.push(record);
            if out.len() == query.limit {
                break;
            }
        }
        Ok(out)
    }

    fn reset(&mut self, reset: &MemoryReset) -> Result<usize, MemoryError> {
        if let MemoryReset::Namespace(namespace) = reset {
            validate_namespace(namespace)?;
        }
        let before = self.records.len();
        match reset {
            MemoryReset::Session => {
                self.records
                    .retain(|record| record.class != MemoryClass::Session);
            }
            MemoryReset::Class(class) => {
                self.records.retain(|record| record.class != *class);
            }
            MemoryReset::Namespace(namespace) => {
                self.records
                    .retain(|record| record.namespace != *namespace);
            }
            MemoryReset::All => {
                self.records.clear();
                self.next_id = 0;
            }
        }
        Ok(before - self.records.len())
    }
}

fn validate_capacity(capacity: usize) -> Result<(), MemoryError> {
    if capacity == 0 || capacity > MAX_MEMORY_ENTRIES {
        return Err(MemoryError::InvalidCapacity(capacity));
    }
    Ok(())
}

fn validate_namespace(namespace: &str) -> Result<(), MemoryError> {
    if namespace.is_empty() {
        return Err(MemoryError::InvalidNamespace("must not be empty".into()));
    }
    if namespace.len() > MAX_NAMESPACE_BYTES {
        return Err(MemoryError::InvalidNamespace(format!(
            "{} bytes exceeds {MAX_NAMESPACE_BYTES}",
            namespace.len()
        )));
    }
    Ok(())
}

fn validate_key(key: &str) -> Result<(), MemoryError> {
    if key.is_empty() {
        return Err(MemoryError::InvalidKey("must not be empty".into()));
    }
    if key.len() > MAX_KEY_BYTES {
        return Err(MemoryError::InvalidKey(format!(
            "{} bytes exceeds {MAX_KEY_BYTES}",
            key.len()
        )));
    }
    Ok(())
}

fn validate_write(write: &MemoryWrite) -> Result<(), MemoryError> {
    validate_namespace(&write.namespace)?;
    validate_key(&write.key)?;
    validate_value(&write.value)
}

fn validate_record(record: &MemoryRecord) -> Result<(), MemoryError> {
    validate_namespace(&record.namespace)?;
    validate_key(&record.key)?;
    validate_value(&record.value)
}

fn validate_value(value: &[f32]) -> Result<(), MemoryError> {
    if value.is_empty() || value.len() > MAX_VALUE_LEN {
        return Err(MemoryError::InvalidValue(format!(
            "length must be in 1..={MAX_VALUE_LEN}, got {}",
            value.len()
        )));
    }
    if value.iter().any(|x| !x.is_finite()) {
        return Err(MemoryError::InvalidValue(
            "all values must be finite".into(),
        ));
    }
    Ok(())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn decode_hex(text: &str) -> Result<Vec<u8>, MemoryError> {
    if text.len() % 2 != 0 {
        return Err(MemoryError::Format("hex field has odd length".into()));
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Result<u8, MemoryError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(MemoryError::Format("invalid hex digit".into())),
    }
}

fn encode_f32_bits(values: &[f32]) -> String {
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:08x}", value.to_bits()));
    }
    out
}

fn decode_f32_bits(text: &str) -> Result<Vec<f32>, MemoryError> {
    if text.is_empty() {
        return Err(MemoryError::InvalidValue("empty encoded value".into()));
    }
    let count = text.split(',').count();
    if count > MAX_VALUE_LEN {
        return Err(MemoryError::InvalidValue(format!(
            "encoded value length {count} exceeds {MAX_VALUE_LEN}"
        )));
    }
    text.split(',')
        .map(|part| {
            if part.len() != 8 {
                return Err(MemoryError::Format(
                    "f32 bits must contain exactly 8 hex digits".into(),
                ));
            }
            let bits = u32::from_str_radix(part, 16)
                .map_err(|_| MemoryError::Format("invalid f32 bits".into()))?;
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err(MemoryError::InvalidValue(
                    "decoded value is not finite".into(),
                ));
            }
            Ok(value)
        })
        .collect()
}

fn parse_named_u32(line: &str, key: &str) -> Result<u32, MemoryError> {
    named_value(line, key)?
        .parse()
        .map_err(|_| MemoryError::Format(format!("invalid {key}")))
}

fn parse_named_usize(line: &str, key: &str) -> Result<usize, MemoryError> {
    named_value(line, key)?
        .parse()
        .map_err(|_| MemoryError::Format(format!("invalid {key}")))
}

fn parse_named_u64(line: &str, key: &str) -> Result<u64, MemoryError> {
    named_value(line, key)?
        .parse()
        .map_err(|_| MemoryError::Format(format!("invalid {key}")))
}

fn named_value<'a>(line: &'a str, key: &str) -> Result<&'a str, MemoryError> {
    line.strip_prefix(key)
        .and_then(|rest| rest.strip_prefix('='))
        .ok_or_else(|| MemoryError::Format(format!("expected {key} field")))
}

fn parse_u64_hex(value: &str) -> Result<u64, MemoryError> {
    if value.len() != 16 {
        return Err(MemoryError::Format(
            "checksum must contain exactly 16 hex digits".into(),
        ));
    }
    u64::from_str_radix(value, 16)
        .map_err(|_| MemoryError::Format("invalid checksum".into()))
}

fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(
        namespace: &str,
        key: &str,
        value: f32,
        class: MemoryClass,
    ) -> MemoryWrite {
        MemoryWrite::new(namespace, key, vec![value, -value], class)
    }

    #[test]
    fn deterministic_write_read_query_and_version() {
        let mut a = InMemoryExternalMemory::new(8).unwrap();
        let mut b = InMemoryExternalMemory::new(8).unwrap();
        for memory in [&mut a, &mut b] {
            assert_eq!(memory.version(), 1);
            assert_eq!(
                memory.write(write("s1", "alpha", 1.25, MemoryClass::Session)),
                Ok(0)
            );
            assert_eq!(
                memory.write(write("shared", "beta", 2.5, MemoryClass::Persistent)),
                Ok(1)
            );
            assert_eq!(
                memory.write(write("weights", "gamma", 3.75, MemoryClass::Trainable)),
                Ok(2)
            );
        }
        assert_eq!(a, b);
        assert_eq!(a.read(1).unwrap().key, "beta");
        assert!(a.read(99).is_none());

        let hit = a.query(&MemoryQuery::exact("shared", "beta")).unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].id, 1);
        assert_eq!(hit[0].value, vec![2.5, -2.5]);

        let trainable = a
            .query(&MemoryQuery::all(8).with_class(MemoryClass::Trainable))
            .unwrap();
        assert_eq!(trainable.len(), 1);
        assert_eq!(trainable[0].key, "gamma");
    }

    #[test]
    fn capacity_is_explicit_and_rejects_without_eviction() {
        let mut memory = InMemoryExternalMemory::new(2).unwrap();
        memory
            .write(write("s", "a", 1.0, MemoryClass::Persistent))
            .unwrap();
        memory
            .write(write("s", "b", 2.0, MemoryClass::Persistent))
            .unwrap();
        let before = memory.clone();
        assert_eq!(
            memory.write(write("s", "c", 3.0, MemoryClass::Persistent)),
            Err(MemoryError::AtCapacity { capacity: 2 })
        );
        assert_eq!(memory, before);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn session_reset_prevents_cross_session_contamination() {
        let mut memory = InMemoryExternalMemory::new(8).unwrap();
        memory
            .write(write("session-a", "secret", 1.0, MemoryClass::Session))
            .unwrap();
        memory
            .write(write("shared", "fact", 2.0, MemoryClass::Persistent))
            .unwrap();
        memory
            .write(write("weights", "slot", 3.0, MemoryClass::Trainable))
            .unwrap();

        assert_eq!(memory.reset(&MemoryReset::Session).unwrap(), 1);
        assert!(memory
            .query(&MemoryQuery::exact("session-a", "secret"))
            .unwrap()
            .is_empty());
        assert_eq!(
            memory
                .query(&MemoryQuery::exact("shared", "fact"))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            memory
                .query(&MemoryQuery::exact("weights", "slot"))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn snapshot_excludes_session_and_roundtrips_bits_exactly() {
        let mut memory = InMemoryExternalMemory::new(8).unwrap();
        memory
            .write(write("temp", "x", 0.5, MemoryClass::Session))
            .unwrap();
        memory
            .write(MemoryWrite::new(
                "persistent",
                "bits",
                vec![f32::from_bits(0x3f00_0001), -0.0],
                MemoryClass::Persistent,
            ))
            .unwrap();
        memory
            .write(write("trainable", "z", 4.0, MemoryClass::Trainable))
            .unwrap();

        let text = memory.encode().unwrap();
        assert_eq!(text.matches("record=").count(), 2);
        assert!(!text.contains(&encode_hex(b"temp")));

        let restored = InMemoryExternalMemory::decode(&text).unwrap();
        assert_eq!(restored.capacity(), 8);
        assert_eq!(restored.next_id(), 3);
        assert_eq!(restored.len(), 2);
        assert_eq!(
            restored.records()[0].value[0].to_bits(),
            0x3f00_0001
        );
        assert_eq!(restored.records()[0].value[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(
            restored.snapshot_fingerprint().unwrap(),
            fingerprint_bytes(text.as_bytes())
        );
    }

    #[test]
    fn persistence_is_deterministic_for_same_operation_sequence() {
        fn build() -> InMemoryExternalMemory {
            let mut memory = InMemoryExternalMemory::new(16).unwrap();
            memory
                .write(write("s", "session", 1.0, MemoryClass::Session))
                .unwrap();
            memory
                .write(write("p", "fact", 2.0, MemoryClass::Persistent))
                .unwrap();
            memory
                .write(write("t", "slot", 3.0, MemoryClass::Trainable))
                .unwrap();
            memory
        }
        assert_eq!(build().encode().unwrap(), build().encode().unwrap());
    }

    #[test]
    fn corrupt_or_incompatible_snapshots_fail_closed() {
        let mut memory = InMemoryExternalMemory::new(4).unwrap();
        memory
            .write(write("p", "fact", 1.0, MemoryClass::Persistent))
            .unwrap();
        let good = memory.encode().unwrap();

        let corrupt = good.replacen("3f800000", "40000000", 1);
        assert!(matches!(
            InMemoryExternalMemory::decode(&corrupt),
            Err(MemoryError::Format(message)) if message.contains("checksum mismatch")
        ));

        let future = good.replacen(
            "auralis_external_memory=1",
            "auralis_external_memory=2",
            1,
        );
        let lines: Vec<&str> = future.lines().collect();
        let mut body = lines[..lines.len() - 1].join("\n");
        body.push('\n');
        let future = format!(
            "{body}checksum={:016x}\n",
            fingerprint_bytes(body.as_bytes())
        );
        assert!(matches!(
            InMemoryExternalMemory::decode(&future),
            Err(MemoryError::Format(message)) if message.contains("unsupported")
        ));
    }

    #[test]
    fn invalid_inputs_and_query_limits_fail_without_mutation() {
        assert!(InMemoryExternalMemory::new(0).is_err());
        let mut memory = InMemoryExternalMemory::new(2).unwrap();
        let before = memory.clone();
        assert!(memory
            .write(MemoryWrite::new("", "k", vec![1.0], MemoryClass::Session))
            .is_err());
        assert!(memory
            .write(MemoryWrite::new("n", "", vec![1.0], MemoryClass::Session))
            .is_err());
        assert!(memory
            .write(MemoryWrite::new(
                "n",
                "k",
                vec![f32::NAN],
                MemoryClass::Session,
            ))
            .is_err());
        assert_eq!(memory, before);
        assert!(memory.query(&MemoryQuery::all(0)).is_err());
    }

    #[test]
    fn save_load_roundtrip_and_reset_all_are_exact() {
        let mut memory = InMemoryExternalMemory::new(4).unwrap();
        memory
            .write(write("p", "a", 1.0, MemoryClass::Persistent))
            .unwrap();
        memory
            .write(write("t", "b", 2.0, MemoryClass::Trainable))
            .unwrap();

        let path = std::env::temp_dir().join(format!(
            "auralis-memory-{}-{}.txt",
            std::process::id(),
            memory.snapshot_fingerprint().unwrap()
        ));
        memory.save(&path).unwrap();
        let loaded = InMemoryExternalMemory::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(memory.encode().unwrap(), loaded.encode().unwrap());

        assert_eq!(memory.reset(&MemoryReset::All).unwrap(), 2);
        assert!(memory.is_empty());
        assert_eq!(memory.next_id(), 0);
        assert_eq!(
            memory
                .write(write("p", "again", 3.0, MemoryClass::Persistent))
                .unwrap(),
            0
        );
    }
}
