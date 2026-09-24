//! #97 versioned event stream with backpressure and cancel.
//!
//! In-process only. Sequence is monotonic. The buffer is bounded: a full
//! buffer rejects the push instead of growing. After cancel, mutative
//! events are refused so no orphan mutation is recorded.

use crate::agent_eval::redact;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

pub const EVENT_STREAM_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_STREAM_CAPACITY: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamEventKind {
    Token,
    Plan,
    Tool,
    Permission,
    Error,
    Cancel,
}

impl StreamEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Plan => "plan",
            Self::Tool => "tool",
            Self::Permission => "permission",
            Self::Error => "error",
            Self::Cancel => "cancel",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "token" => Some(Self::Token),
            "plan" => Some(Self::Plan),
            "tool" => Some(Self::Tool),
            "permission" => Some(Self::Permission),
            "error" => Some(Self::Error),
            "cancel" => Some(Self::Cancel),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEvent {
    pub seq: u64,
    pub kind: StreamEventKind,
    pub payload: String,
    pub mutative: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushResult {
    Accepted,
    Backpressured,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamError {
    IncompatibleSchema(u32),
    Malformed(&'static str),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleSchema(v) => write!(f, "incompatible event stream schema {v}"),
            Self::Malformed(msg) => write!(f, "malformed stream: {msg}"),
        }
    }
}

impl Error for StreamError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventStream {
    pub schema_version: u32,
    pub session_id: String,
    capacity: usize,
    events: VecDeque<StreamEvent>,
    next_seq: u64,
    cancelled: bool,
    dropped: u64,
}

impl EventStream {
    pub fn new(session_id: impl Into<String>, capacity: usize) -> Self {
        Self {
            schema_version: EVENT_STREAM_SCHEMA_VERSION,
            session_id: session_id.into(),
            capacity: capacity.max(1),
            events: VecDeque::new(),
            next_seq: 1,
            cancelled: false,
            dropped: 0,
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn push(&mut self, kind: StreamEventKind, payload: &str, mutative: bool) -> PushResult {
        if self.cancelled && mutative {
            return PushResult::Cancelled;
        }
        if self.events.len() >= self.capacity {
            self.dropped += 1;
            return PushResult::Backpressured;
        }
        let event = StreamEvent {
            seq: self.next_seq,
            kind,
            payload: redact(payload),
            mutative,
        };
        self.next_seq += 1;
        if kind == StreamEventKind::Cancel {
            self.cancelled = true;
        }
        self.events.push_back(event);
        PushResult::Accepted
    }

    pub fn cancel(&mut self, reason: &str) -> PushResult {
        self.push(StreamEventKind::Cancel, reason, false)
    }

    pub fn take(&mut self, max: usize) -> Vec<StreamEvent> {
        let n = max.min(self.events.len());
        self.events.drain(..n).collect()
    }

    pub fn recorded(&self) -> Vec<StreamEvent> {
        self.events.iter().cloned().collect()
    }

    pub fn canonical(&self) -> String {
        let mut out = format!(
            "auralis_stream={}\nsession={}\ncancelled={}\ndropped={}\n",
            self.schema_version,
            self.session_id,
            self.cancelled as u8,
            self.dropped
        );
        for event in &self.events {
            out.push_str(&format!(
                "{}|{}|{}|{}\n",
                event.seq,
                event.kind.as_str(),
                event.mutative as u8,
                escape(&event.payload)
            ));
        }
        out
    }

    pub fn load(blob: &str) -> Result<Self, StreamError> {
        let mut lines = blob.lines();
        let schema = lines
            .next()
            .and_then(|l| l.strip_prefix("auralis_stream="))
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or(StreamError::Malformed("missing schema"))?;
        if schema != EVENT_STREAM_SCHEMA_VERSION {
            return Err(StreamError::IncompatibleSchema(schema));
        }
        let session = lines
            .next()
            .and_then(|l| l.strip_prefix("session="))
            .ok_or(StreamError::Malformed("missing session"))?
            .to_string();
        let cancelled = lines
            .next()
            .and_then(|l| l.strip_prefix("cancelled="))
            .and_then(|v| Some(v == "1"))
            .ok_or(StreamError::Malformed("missing cancelled"))?;
        let dropped = lines
            .next()
            .and_then(|l| l.strip_prefix("dropped="))
            .and_then(|v| v.parse().ok())
            .ok_or(StreamError::Malformed("missing dropped"))?;
        let mut stream = EventStream::new(session, DEFAULT_STREAM_CAPACITY);
        stream.cancelled = cancelled;
        stream.dropped = dropped;
        let mut max_seq = 0u64;
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(4, '|');
            let seq = parts
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or(StreamError::Malformed("bad seq"))?;
            let kind = parts
                .next()
                .and_then(StreamEventKind::parse)
                .ok_or(StreamError::Malformed("bad kind"))?;
            let mutative = parts.next() == Some("1");
            let payload = unescape(parts.next().unwrap_or(""));
            stream.events.push_back(StreamEvent {
                seq,
                kind,
                payload,
                mutative,
            });
            max_seq = max_seq.max(seq);
        }
        stream.next_seq = max_seq + 1;
        Ok(stream)
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
    fn events_stay_ordered_and_replay() {
        let mut stream = EventStream::new("s-97", 8);
        assert_eq!(stream.push(StreamEventKind::Token, "hel", false), PushResult::Accepted);
        assert_eq!(stream.push(StreamEventKind::Token, "lo", false), PushResult::Accepted);
        assert_eq!(stream.push(StreamEventKind::Plan, "lookup", false), PushResult::Accepted);
        assert_eq!(stream.push(StreamEventKind::Tool, "ref.echo", true), PushResult::Accepted);
        assert_eq!(
            stream.push(StreamEventKind::Permission, "allow", false),
            PushResult::Accepted
        );
        let recorded = stream.recorded();
        let seqs: Vec<u64> = recorded.iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![1, 2, 3, 4, 5]);
        let loaded = EventStream::load(&stream.canonical()).unwrap();
        assert_eq!(loaded.recorded(), recorded);
    }

    #[test]
    fn backpressure_does_not_grow_or_reorder() {
        let mut stream = EventStream::new("s-97", 2);
        assert_eq!(stream.push(StreamEventKind::Token, "a", false), PushResult::Accepted);
        assert_eq!(stream.push(StreamEventKind::Token, "b", false), PushResult::Accepted);
        assert_eq!(stream.push(StreamEventKind::Token, "c", false), PushResult::Backpressured);
        assert_eq!(stream.len(), 2);
        assert_eq!(stream.dropped(), 1);
        let payloads: Vec<_> = stream.recorded().into_iter().map(|e| e.payload).collect();
        assert_eq!(payloads, vec!["a", "b"]);
        let taken = stream.take(1);
        assert_eq!(taken[0].payload, "a");
        assert_eq!(stream.push(StreamEventKind::Token, "c", false), PushResult::Accepted);
        let payloads: Vec<_> = stream.recorded().into_iter().map(|e| e.payload).collect();
        assert_eq!(payloads, vec!["b", "c"]);
    }

    #[test]
    fn cancel_blocks_mutative_events() {
        let mut stream = EventStream::new("s-97", 8);
        assert_eq!(stream.cancel("client-disconnect"), PushResult::Accepted);
        assert!(stream.cancelled());
        assert_eq!(
            stream.push(StreamEventKind::Tool, "ref.add", true),
            PushResult::Cancelled
        );
        assert_eq!(
            stream.push(StreamEventKind::Error, "late-error", false),
            PushResult::Accepted
        );
        assert!(!stream
            .recorded()
            .iter()
            .any(|e| e.kind == StreamEventKind::Tool));
    }

    #[test]
    fn secrets_are_redacted_and_incompatible_schema_fails() {
        let mut stream = EventStream::new("s-97", 4);
        stream.push(StreamEventKind::Token, "token=abc123 visible", false);
        let blob = stream.canonical();
        assert!(!blob.contains("abc123"));
        assert!(blob.contains("[REDACTED]"));
        let err = EventStream::load("auralis_stream=9\nsession=x\ncancelled=0\ndropped=0\n").unwrap_err();
        assert_eq!(err, StreamError::IncompatibleSchema(9));
    }
}
