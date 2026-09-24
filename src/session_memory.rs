//! #98 session memory decoupled from model weights.
//!
//! Optional short-term store. Memory off keeps the runtime callable.
//! Two SessionMemory values never share records. Eviction is explicit
//! evict-oldest when capacity is reached.

use crate::agent_eval::redact;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

pub const SESSION_MEMORY_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MEMORY_CAPACITY: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Event,
    Note,
    ToolResult,
}

impl MemoryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Event => "event",
            Self::Note => "note",
            Self::ToolResult => "tool-result",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "event" => Some(Self::Event),
            "note" => Some(Self::Note),
            "tool-result" => Some(Self::ToolResult),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: String,
    pub kind: MemoryKind,
    pub tag: String,
    pub text: String,
    pub seq: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryError {
    IncompatibleSchema(u32),
    Malformed(&'static str),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleSchema(v) => write!(f, "incompatible session memory schema {v}"),
            Self::Malformed(msg) => write!(f, "malformed memory: {msg}"),
        }
    }
}

impl Error for MemoryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMemory {
    pub schema_version: u32,
    pub session_id: String,
    enabled: bool,
    capacity: usize,
    records: VecDeque<MemoryRecord>,
    next_seq: u64,
    evicted: u64,
    hits: u64,
    misses: u64,
}

impl SessionMemory {
    pub fn new(session_id: impl Into<String>, capacity: usize) -> Self {
        Self {
            schema_version: SESSION_MEMORY_SCHEMA_VERSION,
            session_id: session_id.into(),
            enabled: true,
            capacity: capacity.max(1),
            records: VecDeque::new(),
            next_seq: 1,
            evicted: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn off(session_id: impl Into<String>) -> Self {
        let mut memory = Self::new(session_id, DEFAULT_MEMORY_CAPACITY);
        memory.enabled = false;
        memory
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn remember(&mut self, kind: MemoryKind, tag: &str, text: &str) -> Option<MemoryRecord> {
        if !self.enabled {
            return None;
        }
        if self.records.len() >= self.capacity {
            self.records.pop_front();
            self.evicted += 1;
        }
        let record = MemoryRecord {
            id: format!("{}-m{}", self.session_id, self.next_seq),
            kind,
            tag: tag.to_string(),
            text: redact(text),
            seq: self.next_seq,
        };
        self.next_seq += 1;
        self.records.push_back(record.clone());
        Some(record)
    }

    pub fn retrieve(
        &mut self,
        kind: Option<MemoryKind>,
        tag: Option<&str>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        if !self.enabled {
            self.misses += 1;
            return Vec::new();
        }
        let mut matched: Vec<MemoryRecord> = self
            .records
            .iter()
            .rev()
            .filter(|record| kind.map(|k| record.kind == k).unwrap_or(true))
            .filter(|record| tag.map(|t| record.tag == t).unwrap_or(true))
            .take(limit)
            .cloned()
            .collect();
        if matched.is_empty() {
            self.misses += 1;
        } else {
            self.hits += 1;
        }
        matched.reverse();
        matched
    }

    pub fn reset(&mut self) {
        self.records.clear();
        self.next_seq = 1;
        self.evicted = 0;
        self.hits = 0;
        self.misses = 0;
    }

    pub fn canonical(&self) -> String {
        let mut out = format!(
            "auralis_session_memory={}\nsession={}\nenabled={}\ncapacity={}\n",
            self.schema_version,
            self.session_id,
            self.enabled as u8,
            self.capacity
        );
        for record in &self.records {
            out.push_str(&format!(
                "{}|{}|{}|{}|{}\n",
                record.seq,
                record.kind.as_str(),
                escape(&record.tag),
                escape(&record.text),
                record.id
            ));
        }
        out
    }

    pub fn load(blob: &str) -> Result<Self, MemoryError> {
        let mut lines = blob.lines();
        let schema = lines
            .next()
            .and_then(|l| l.strip_prefix("auralis_session_memory="))
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or(MemoryError::Malformed("missing schema"))?;
        if schema != SESSION_MEMORY_SCHEMA_VERSION {
            return Err(MemoryError::IncompatibleSchema(schema));
        }
        let session = lines
            .next()
            .and_then(|l| l.strip_prefix("session="))
            .ok_or(MemoryError::Malformed("missing session"))?
            .to_string();
        let enabled = lines
            .next()
            .and_then(|l| l.strip_prefix("enabled="))
            .map(|v| v == "1")
            .ok_or(MemoryError::Malformed("missing enabled"))?;
        let capacity = lines
            .next()
            .and_then(|l| l.strip_prefix("capacity="))
            .and_then(|v| v.parse().ok())
            .ok_or(MemoryError::Malformed("missing capacity"))?;
        let mut memory = if enabled {
            SessionMemory::new(session, capacity)
        } else {
            SessionMemory::off(session)
        };
        memory.capacity = capacity.max(1);
        let mut max_seq = 0u64;
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(5, '|');
            let seq = parts
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or(MemoryError::Malformed("bad seq"))?;
            let kind = parts
                .next()
                .and_then(MemoryKind::parse)
                .ok_or(MemoryError::Malformed("bad kind"))?;
            let tag = unescape(parts.next().unwrap_or(""));
            let text = unescape(parts.next().unwrap_or(""));
            let id = parts.next().unwrap_or("").to_string();
            memory.records.push_back(MemoryRecord {
                id,
                kind,
                tag,
                text,
                seq,
            });
            max_seq = max_seq.max(seq);
        }
        memory.next_seq = max_seq + 1;
        Ok(memory)
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\n', "\\n").replace('|', "\\|")
}

fn unescape(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('|') => out.push('|'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_retrieve_reset_and_replay() {
        let mut memory = SessionMemory::new("s-98", 8);
        memory.remember(MemoryKind::Note, "goal", "find city");
        memory.remember(MemoryKind::ToolResult, "lookup", "paris");
        memory.remember(MemoryKind::Event, "user", "thanks");
        let by_tag = memory.retrieve(None, Some("lookup"), 4);
        assert_eq!(by_tag.len(), 1);
        assert_eq!(by_tag[0].text, "paris");
        let recent = memory.retrieve(None, None, 2);
        assert_eq!(recent[0].tag, "lookup");
        assert_eq!(recent[1].tag, "user");
        let blob = memory.canonical();
        let loaded = SessionMemory::load(&blob).unwrap();
        assert_eq!(loaded.retrieve(None, None, 8), memory.retrieve(None, None, 8));
        memory.reset();
        assert_eq!(memory.len(), 0);
        assert_eq!(memory.retrieve(None, None, 8).len(), 0);
    }

    #[test]
    fn evict_oldest_is_explicit() {
        let mut memory = SessionMemory::new("s-98", 2);
        memory.remember(MemoryKind::Note, "a", "one");
        memory.remember(MemoryKind::Note, "b", "two");
        memory.remember(MemoryKind::Note, "c", "three");
        assert_eq!(memory.len(), 2);
        assert_eq!(memory.evicted(), 1);
        let texts: Vec<_> = memory.retrieve(None, None, 8).into_iter().map(|r| r.text).collect();
        assert_eq!(texts, vec!["two", "three"]);
    }

    #[test]
    fn sessions_do_not_share_and_off_is_complete() {
        let mut a = SessionMemory::new("a", 4);
        let mut b = SessionMemory::new("b", 4);
        a.remember(MemoryKind::Note, "secret", "alpha");
        b.remember(MemoryKind::Note, "secret", "beta");
        assert_eq!(a.retrieve(None, Some("secret"), 1)[0].text, "alpha");
        assert_eq!(b.retrieve(None, Some("secret"), 1)[0].text, "beta");
        assert!(!a.canonical().contains("beta"));
        let mut off = SessionMemory::off("c");
        assert!(!off.enabled());
        assert!(off.remember(MemoryKind::Note, "x", "ignored").is_none());
        assert!(off.retrieve(None, None, 4).is_empty());
        assert_eq!(off.misses(), 1);
    }

    #[test]
    fn secrets_redacted_and_schema_rejected() {
        let mut memory = SessionMemory::new("s-98", 4);
        memory.remember(MemoryKind::Note, "auth", "password=hunter2");
        assert!(!memory.canonical().contains("hunter2"));
        let err = SessionMemory::load("auralis_session_memory=9\nsession=x\nenabled=1\ncapacity=4\n")
            .unwrap_err();
        assert_eq!(err, MemoryError::IncompatibleSchema(9));
    }
}
