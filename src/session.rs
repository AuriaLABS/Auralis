//! #48 versioned local session. Ephemeral REPL state, never model weights.
//!
//! Tools and planner stay off by default. Recording a tool intent does not
//! invoke #44/#45. Secrets are redacted before an event is kept.

use crate::agent_eval::redact;
use std::error::Error;
use std::fmt;

pub const SESSION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Interrupted,
    Closed,
}

impl SessionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Interrupted => "interrupted",
            Self::Closed => "closed",
        }
    }

    fn parse(value: &str) -> Result<Self, SessionError> {
        match value {
            "active" => Ok(Self::Active),
            "interrupted" => Ok(Self::Interrupted),
            "closed" => Ok(Self::Closed),
            other => Err(SessionError::Invalid(format!("status {other}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    User,
    Assistant,
    System,
    ToolIntent,
    Error,
    Interrupt,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::ToolIntent => "tool-intent",
            Self::Error => "error",
            Self::Interrupt => "interrupt",
        }
    }

    fn parse(value: &str) -> Result<Self, SessionError> {
        match value {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "system" => Ok(Self::System),
            "tool-intent" => Ok(Self::ToolIntent),
            "error" => Ok(Self::Error),
            "interrupt" => Ok(Self::Interrupt),
            other => Err(SessionError::Invalid(format!("kind {other}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionEvent {
    pub seq: u32,
    pub kind: EventKind,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Session {
    pub session_id: String,
    pub schema_version: u32,
    pub generation: u32,
    pub status: SessionStatus,
    pub tools_enabled: bool,
    pub planner_enabled: bool,
    pub model_checkpoint: String,
    pub events: Vec<SessionEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError {
    IncompatibleSchema(u32),
    Invalid(String),
    Closed,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleSchema(v) => write!(f, "incompatible session schema {v}"),
            Self::Invalid(detail) => write!(f, "invalid session: {detail}"),
            Self::Closed => write!(f, "session is closed"),
        }
    }
}

impl Error for SessionError {}

impl Session {
    pub fn new(session_id: impl Into<String>, model_checkpoint: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            schema_version: SESSION_SCHEMA_VERSION,
            generation: 0,
            status: SessionStatus::Active,
            tools_enabled: false,
            planner_enabled: false,
            model_checkpoint: model_checkpoint.into(),
            events: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.events.clear();
        self.generation = self.generation.saturating_add(1);
        self.status = SessionStatus::Active;
    }

    pub fn close(&mut self) {
        self.status = SessionStatus::Closed;
    }

    pub fn interrupt(&mut self) -> Result<(), SessionError> {
        self.push(EventKind::Interrupt, "interrupt")?;
        self.status = SessionStatus::Interrupted;
        Ok(())
    }

    pub fn set_tools_enabled(&mut self, enabled: bool) {
        self.tools_enabled = enabled;
    }

    pub fn set_planner_enabled(&mut self, enabled: bool) {
        self.planner_enabled = enabled;
    }

    pub fn push_user(&mut self, text: &str) -> Result<(), SessionError> {
        self.push(EventKind::User, text)
    }

    pub fn push_assistant(&mut self, text: &str) -> Result<(), SessionError> {
        self.push(EventKind::Assistant, text)
    }

    pub fn record_error(&mut self, detail: &str) -> Result<(), SessionError> {
        self.push(EventKind::Error, detail)
    }

    pub fn record_tool_intent(&mut self, spec: &str) -> Result<(), SessionError> {
        let payload = if self.tools_enabled {
            format!("intent:{spec}")
        } else {
            format!("tools-disabled:{spec}")
        };
        self.push(EventKind::ToolIntent, &payload)
    }

    pub fn record_plan_intent(&mut self, spec: &str) -> Result<(), SessionError> {
        let payload = if self.planner_enabled {
            format!("plan:{spec}")
        } else {
            format!("planner-disabled:{spec}")
        };
        self.push(EventKind::System, &payload)
    }

    fn push(&mut self, kind: EventKind, payload: &str) -> Result<(), SessionError> {
        if self.status == SessionStatus::Closed {
            return Err(SessionError::Closed);
        }
        let seq = self.events.len() as u32;
        self.events.push(SessionEvent {
            seq,
            kind,
            payload: redact(payload),
        });
        if self.status == SessionStatus::Interrupted && kind != EventKind::Interrupt {
            self.status = SessionStatus::Active;
        }
        Ok(())
    }

    pub fn canonical(&self) -> String {
        let mut out = format!(
            "auralis_session={}\nsession_id={}\ngeneration={}\nstatus={}\ntools={}\nplanner={}\ncheckpoint={}\n",
            self.schema_version,
            esc(&self.session_id),
            self.generation,
            self.status.as_str(),
            if self.tools_enabled { 1 } else { 0 },
            if self.planner_enabled { 1 } else { 0 },
            esc(&self.model_checkpoint),
        );
        for event in &self.events {
            out.push_str(&format!(
                "event={}|{}|{}\n",
                event.seq,
                event.kind.as_str(),
                esc(&event.payload)
            ));
        }
        out
    }

    pub fn export(&self) -> String {
        self.canonical()
    }

    pub fn load(blob: &str) -> Result<Self, SessionError> {
        let mut schema = None;
        let mut session_id = None;
        let mut generation = None;
        let mut status = None;
        let mut tools = None;
        let mut planner = None;
        let mut checkpoint = None;
        let mut events = Vec::new();
        for raw in blob.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(value) = line.strip_prefix("auralis_session=") {
                let parsed: u32 = value
                    .parse()
                    .map_err(|_| SessionError::Invalid("schema".into()))?;
                if parsed != SESSION_SCHEMA_VERSION {
                    return Err(SessionError::IncompatibleSchema(parsed));
                }
                schema = Some(parsed);
            } else if let Some(value) = line.strip_prefix("session_id=") {
                session_id = Some(unesc(value));
            } else if let Some(value) = line.strip_prefix("generation=") {
                generation = Some(
                    value
                        .parse()
                        .map_err(|_| SessionError::Invalid("generation".into()))?,
                );
            } else if let Some(value) = line.strip_prefix("status=") {
                status = Some(SessionStatus::parse(value)?);
            } else if let Some(value) = line.strip_prefix("tools=") {
                tools = Some(value == "1");
            } else if let Some(value) = line.strip_prefix("planner=") {
                planner = Some(value == "1");
            } else if let Some(value) = line.strip_prefix("checkpoint=") {
                checkpoint = Some(unesc(value));
            } else if let Some(value) = line.strip_prefix("event=") {
                let parts: Vec<&str> = value.splitn(3, '|').collect();
                if parts.len() != 3 {
                    return Err(SessionError::Invalid("event".into()));
                }
                let seq: u32 = parts[0]
                    .parse()
                    .map_err(|_| SessionError::Invalid("event seq".into()))?;
                events.push(SessionEvent {
                    seq,
                    kind: EventKind::parse(parts[1])?,
                    payload: unesc(parts[2]),
                });
            } else {
                return Err(SessionError::Invalid(format!("line {line}")));
            }
        }
        let session = Self {
            session_id: session_id.ok_or_else(|| SessionError::Invalid("session_id".into()))?,
            schema_version: schema.ok_or_else(|| SessionError::Invalid("schema".into()))?,
            generation: generation.ok_or_else(|| SessionError::Invalid("generation".into()))?,
            status: status.ok_or_else(|| SessionError::Invalid("status".into()))?,
            tools_enabled: tools.unwrap_or(false),
            planner_enabled: planner.unwrap_or(false),
            model_checkpoint: checkpoint.unwrap_or_else(|| "none".into()),
            events,
        };
        if session.canonical().contains("SECRET")
            || session.canonical().contains("sk-live-")
            || session.canonical().contains("password=")
        {
            return Err(SessionError::Invalid("unredacted secret".into()));
        }
        Ok(session)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplOutcome {
    Reply(String),
    Exported(String),
    Loaded { session_id: String, events: usize },
    Exit,
}

pub struct SessionRepl {
    pub session: Session,
}

impl SessionRepl {
    pub fn new(session_id: impl Into<String>, checkpoint: impl Into<String>) -> Self {
        Self {
            session: Session::new(session_id, checkpoint),
        }
    }

    pub fn handle(&mut self, line: &str) -> Result<ReplOutcome, SessionError> {
        let trimmed = line.trim();
        if trimmed == "/salir" || trimmed == "/exit" {
            self.session.close();
            return Ok(ReplOutcome::Exit);
        }
        if trimmed == "/reset" {
            let checkpoint = self.session.model_checkpoint.clone();
            self.session.reset();
            return Ok(ReplOutcome::Reply(format!(
                "reset generation={} checkpoint={checkpoint}",
                self.session.generation
            )));
        }
        if trimmed == "/export" || trimmed == "/save" {
            return Ok(ReplOutcome::Exported(self.session.export()));
        }
        if let Some(blob) = trimmed.strip_prefix("/load ") {
            let loaded = Session::load(blob)?;
            self.session = loaded;
            return Ok(ReplOutcome::Loaded {
                session_id: self.session.session_id.clone(),
                events: self.session.events.len(),
            });
        }
        if trimmed == "/interrupt" {
            self.session.interrupt()?;
            return Ok(ReplOutcome::Reply("interrupted".into()));
        }
        if trimmed == "/tools on" {
            self.session.set_tools_enabled(true);
            return Ok(ReplOutcome::Reply("tools=on".into()));
        }
        if trimmed == "/tools off" {
            self.session.set_tools_enabled(false);
            return Ok(ReplOutcome::Reply("tools=off".into()));
        }
        if trimmed == "/planner on" {
            self.session.set_planner_enabled(true);
            return Ok(ReplOutcome::Reply("planner=on".into()));
        }
        if trimmed == "/planner off" {
            self.session.set_planner_enabled(false);
            return Ok(ReplOutcome::Reply("planner=off".into()));
        }
        if let Some(spec) = trimmed.strip_prefix("/tool ") {
            self.session.record_tool_intent(spec)?;
            let note = if self.session.tools_enabled {
                "intent recorded; not executed"
            } else {
                "tools-disabled"
            };
            return Ok(ReplOutcome::Reply(note.into()));
        }
        if let Some(spec) = trimmed.strip_prefix("/plan ") {
            self.session.record_plan_intent(spec)?;
            let note = if self.session.planner_enabled {
                "plan intent recorded; planner not invoked"
            } else {
                "planner-disabled"
            };
            return Ok(ReplOutcome::Reply(note.into()));
        }
        self.session.push_user(trimmed)?;
        let reply = format!("ack:{}", self.session.events.len());
        self.session.push_assistant(&reply)?;
        Ok(ReplOutcome::Reply(reply))
    }
}

fn esc(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unesc(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('|') => out.push('|'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
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
    fn create_reset_save_load_roundtrip() {
        let mut session = Session::new("sess-48", "ckpt-name-only");
        session.push_user("hola").unwrap();
        session.push_assistant("ack:1").unwrap();
        let blob = session.export();
        let loaded = Session::load(&blob).unwrap();
        assert_eq!(loaded, session);
        session.reset();
        assert!(session.events.is_empty());
        assert_eq!(session.generation, 1);
        assert_eq!(session.model_checkpoint, "ckpt-name-only");
        assert_eq!(session.session_id, "sess-48");
        let after_reset = Session::load(&session.export()).unwrap();
        assert_eq!(after_reset.events.len(), 0);
        assert_eq!(after_reset.generation, 1);
    }

    #[test]
    fn incompatible_schema_is_rejected() {
        let blob = "auralis_session=2\nsession_id=x\ngeneration=0\nstatus=active\ntools=0\nplanner=0\ncheckpoint=none\n";
        match Session::load(blob) {
            Err(SessionError::IncompatibleSchema(2)) => {}
            other => panic!("expected schema reject, got {other:?}"),
        }
    }

    #[test]
    fn secrets_are_not_serialized() {
        let mut session = Session::new("sess-secret", "none");
        session
            .push_user("token=abc123 password=hunter2 SECRET")
            .unwrap();
        let blob = session.export();
        assert!(!blob.contains("abc123"));
        assert!(!blob.contains("hunter2"));
        assert!(!blob.contains("SECRET"));
        assert!(blob.contains("[REDACTED]"));
    }

    #[test]
    fn partial_error_does_not_drop_history() {
        let mut session = Session::new("sess-err", "none");
        session.push_user("keep-me").unwrap();
        session.record_error("boom").unwrap();
        assert_eq!(session.events.len(), 2);
        assert_eq!(session.events[0].payload, "keep-me");
        assert_eq!(session.events[1].kind, EventKind::Error);
    }

    #[test]
    fn repl_works_with_tools_and_planner_disabled() {
        let mut repl = SessionRepl::new("sess-repl", "none");
        assert!(!repl.session.tools_enabled);
        assert!(!repl.session.planner_enabled);
        let tool = repl.handle("/tool lookup city-1").unwrap();
        assert_eq!(tool, ReplOutcome::Reply("tools-disabled".into()));
        let plan = repl.handle("/plan lookup").unwrap();
        assert_eq!(plan, ReplOutcome::Reply("planner-disabled".into()));
        let talk = repl.handle("hola").unwrap();
        assert!(matches!(talk, ReplOutcome::Reply(_)));
        assert_eq!(repl.session.events.len(), 4);
        let exported = match repl.handle("/export").unwrap() {
            ReplOutcome::Exported(blob) => blob,
            other => panic!("{other:?}"),
        };
        let mut other = SessionRepl::new("tmp", "other-ckpt");
        other.handle(&format!("/load {exported}")).unwrap();
        assert_eq!(other.session.session_id, "sess-repl");
        assert_eq!(other.session.events.len(), 4);
        assert_eq!(other.session.model_checkpoint, "none");
    }

    #[test]
    fn fixture_order_is_stable() {
        let fixture = "auralis_session=1\nsession_id=fixture-48\ngeneration=0\nstatus=active\ntools=0\nplanner=0\ncheckpoint=none\nevent=0|user|one\nevent=1|assistant|ack:1\n";
        let loaded = Session::load(fixture).unwrap();
        assert_eq!(loaded.export(), fixture);
        assert_eq!(loaded.events[0].payload, "one");
        assert_eq!(loaded.events[1].seq, 1);
    }
}
