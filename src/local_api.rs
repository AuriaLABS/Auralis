//! #50 local agent API. Same SessionRepl runtime as the REPL.
//!
//! Bind is loopback-only when a socket is used. No public internet, no
//! multi-tenant, no OAuth. Tools are listed, not executed here.

use crate::session::{ReplOutcome, SessionError, SessionRepl};
use crate::tool_registry::ToolRegistry;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::Instant;

pub const LOCAL_API_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiLimits {
    pub max_body_bytes: usize,
    pub max_sessions: usize,
    pub max_concurrent: usize,
    pub max_wall_ms: u64,
}

impl Default for ApiLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: 8_192,
            max_sessions: 32,
            max_concurrent: 4,
            max_wall_ms: 1_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiRequest {
    pub method: String,
    pub path: String,
    pub body: String,
    pub correlation_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiResponse {
    pub status: u16,
    pub body: String,
    pub correlation_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiError {
    NotFound,
    Conflict(String),
    TooLarge,
    Busy,
    Timeout,
    Incompatible(u32),
    Session(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::Conflict(msg) => write!(f, "conflict: {msg}"),
            Self::TooLarge => write!(f, "payload too large"),
            Self::Busy => write!(f, "too many concurrent sessions"),
            Self::Timeout => write!(f, "request exceeded wall budget"),
            Self::Incompatible(v) => write!(f, "incompatible api schema {v}"),
            Self::Session(msg) => write!(f, "session: {msg}"),
        }
    }
}

impl Error for ApiError {}

pub struct LocalAgentApi {
    sessions: BTreeMap<String, SessionRepl>,
    limits: ApiLimits,
    next_id: u64,
    in_flight: u32,
}

impl LocalAgentApi {
    pub fn new(limits: ApiLimits) -> Self {
        Self {
            sessions: BTreeMap::new(),
            limits,
            next_id: 1,
            in_flight: 0,
        }
    }

    pub fn handle(&mut self, request: ApiRequest) -> ApiResponse {
        let started = Instant::now();
        if request.body.len() > self.limits.max_body_bytes {
            return error_response(413, &request.correlation_id, ApiError::TooLarge);
        }
        if self.in_flight as usize >= self.limits.max_concurrent {
            return error_response(429, &request.correlation_id, ApiError::Busy);
        }
        self.in_flight += 1;
        let response = self.dispatch(&request);
        self.in_flight = self.in_flight.saturating_sub(1);
        if started.elapsed().as_millis() as u64 > self.limits.max_wall_ms {
            return error_response(504, &request.correlation_id, ApiError::Timeout);
        }
        response
    }

    fn dispatch(&mut self, request: &ApiRequest) -> ApiResponse {
        let method = request.method.to_uppercase();
        let path = request.path.trim_end_matches('/');
        match (method.as_str(), path) {
            ("GET", "/v1/version") => ok(
                &request.correlation_id,
                &format!("schema={LOCAL_API_SCHEMA_VERSION}"),
            ),
            ("GET", "/v1/tools") => {
                let listing = ToolRegistry::with_reference_tools()
                    .map(|reg| {
                        reg.list()
                            .into_iter()
                            .map(|t| format!("{}@{}:{}:{}", t.name, t.version, t.capability, t.mutative as u8))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                ok(&request.correlation_id, &listing)
            }
            ("GET", "/v1/capabilities") => ok(
                &request.correlation_id,
                "session,tools,messages,close",
            ),
            ("POST", "/v1/sessions") => self.create_session(request),
            _ if path.starts_with("/v1/sessions/") => self.session_route(&method, path, request),
            _ => error_response(404, &request.correlation_id, ApiError::NotFound),
        }
    }

    fn create_session(&mut self, request: &ApiRequest) -> ApiResponse {
        if self.sessions.len() >= self.limits.max_sessions {
            return error_response(429, &request.correlation_id, ApiError::Busy);
        }
        let id = if request.body.starts_with("id=") {
            request.body.trim_start_matches("id=").trim().to_string()
        } else {
            let id = format!("api-{}", self.next_id);
            self.next_id += 1;
            id
        };
        if self.sessions.contains_key(&id) {
            return error_response(
                409,
                &request.correlation_id,
                ApiError::Conflict(id),
            );
        }
        let checkpoint = "none";
        self.sessions
            .insert(id.clone(), SessionRepl::new(id.clone(), checkpoint));
        ok(&request.correlation_id, &format!("session={id}"))
    }

    fn session_route(&mut self, method: &str, path: &str, request: &ApiRequest) -> ApiResponse {
        let rest = &path["/v1/sessions/".len()..];
        let (id, suffix) = match rest.split_once('/') {
            Some((id, suffix)) => (id, suffix),
            None => (rest, ""),
        };
        match (method, suffix) {
            ("GET", "") => match self.sessions.get(id) {
                Some(repl) => ok(&request.correlation_id, &repl.session.export()),
                None => error_response(404, &request.correlation_id, ApiError::NotFound),
            },
            ("POST", "messages") => self.message(id, request),
            ("POST", "close") => {
                if let Some(repl) = self.sessions.get_mut(id) {
                    let _ = repl.handle("/salir");
                }
                self.sessions.remove(id);
                ok(&request.correlation_id, "closed")
            }
            _ => error_response(404, &request.correlation_id, ApiError::NotFound),
        }
    }

    fn message(&mut self, id: &str, request: &ApiRequest) -> ApiResponse {
        let Some(repl) = self.sessions.get_mut(id) else {
            return error_response(404, &request.correlation_id, ApiError::NotFound);
        };
        match repl.handle(&request.body) {
            Ok(ReplOutcome::Reply(text)) => ok(&request.correlation_id, &text),
            Ok(ReplOutcome::Exported(blob)) => ok(&request.correlation_id, &blob),
            Ok(ReplOutcome::Loaded { session_id, events }) => ok(
                &request.correlation_id,
                &format!("loaded={session_id} events={events}"),
            ),
            Ok(ReplOutcome::Exit) => {
                self.sessions.remove(id);
                ok(&request.correlation_id, "closed")
            }
            Err(SessionError::IncompatibleSchema(v)) => {
                error_response(409, &request.correlation_id, ApiError::Incompatible(v))
            }
            Err(err) => error_response(400, &request.correlation_id, ApiError::Session(err.to_string())),
        }
    }
}

fn ok(correlation: &str, body: &str) -> ApiResponse {
    ApiResponse {
        status: 200,
        body: body.to_string(),
        correlation_id: correlation.to_string(),
    }
}

fn error_response(status: u16, correlation: &str, error: ApiError) -> ApiResponse {
    ApiResponse {
        status,
        body: format!("error={error}"),
        correlation_id: correlation.to_string(),
    }
}

/// Serve one loopback request. Bind only `127.0.0.1`.
pub fn serve_one(listener: &TcpListener, api: &mut LocalAgentApi) -> std::io::Result<()> {
    let (stream, addr) = listener.accept()?;
    if !addr.ip().is_loopback() {
        return Ok(());
    }
    handle_http(stream, api)
}

fn handle_http(mut stream: TcpStream, api: &mut LocalAgentApi) -> std::io::Result<()> {
    let mut buf = vec![0u8; 16_384];
    let n = stream.read(&mut buf)?;
    let raw = String::from_utf8_lossy(&buf[..n]);
    let mut lines = raw.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut correlation = "missing".to_string();
    let mut content_len = 0usize;
    for line in lines.by_ref() {
        if line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix("X-Correlation-Id: ") {
            correlation = value.to_string();
        }
        if let Some(value) = line.strip_prefix("Content-Length: ") {
            content_len = value.parse().unwrap_or(0);
        }
    }
    let header_end = raw.find("\r\n\r\n").map(|i| i + 4).unwrap_or(raw.len());
    let body = raw.get(header_end..header_end + content_len).unwrap_or("").to_string();
    let response = api.handle(ApiRequest {
        method,
        path,
        body,
        correlation_id: correlation.clone(),
    });
    let out = format!(
        "HTTP/1.1 {} OK\r\nX-Correlation-Id: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.status,
        response.correlation_id,
        response.body.len(),
        response.body
    );
    stream.write_all(out.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionRepl;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn req(method: &str, path: &str, body: &str, corr: &str) -> ApiRequest {
        ApiRequest {
            method: method.into(),
            path: path.into(),
            body: body.into(),
            correlation_id: corr.into(),
        }
    }

    #[test]
    fn create_message_close_and_list_tools() {
        let mut api = LocalAgentApi::new(ApiLimits::default());
        let created = api.handle(req("POST", "/v1/sessions", "id=s-50", "c1"));
        assert_eq!(created.status, 200);
        assert_eq!(created.correlation_id, "c1");
        let reply = api.handle(req("POST", "/v1/sessions/s-50/messages", "hola", "c2"));
        assert_eq!(reply.status, 200);
        assert!(reply.body.starts_with("ack:"));
        let tools = api.handle(req("GET", "/v1/tools", "", "c3"));
        assert!(tools.body.contains("ref.echo@1"));
        let closed = api.handle(req("POST", "/v1/sessions/s-50/close", "", "c4"));
        assert_eq!(closed.body, "closed");
        let missing = api.handle(req("GET", "/v1/sessions/s-50", "", "c5"));
        assert_eq!(missing.status, 404);
    }

    #[test]
    fn repl_and_api_share_semantics() {
        let mut repl = SessionRepl::new("eq-50", "none");
        let via_repl = match repl.handle("hola").unwrap() {
            crate::session::ReplOutcome::Reply(text) => text,
            other => panic!("{other:?}"),
        };
        let mut api = LocalAgentApi::new(ApiLimits::default());
        api.handle(req("POST", "/v1/sessions", "id=eq-50", "c"));
        let via_api = api.handle(req("POST", "/v1/sessions/eq-50/messages", "hola", "c"));
        assert_eq!(via_api.body, via_repl);
        let exported = api.handle(req("GET", "/v1/sessions/eq-50", "", "c"));
        assert_eq!(exported.body, repl.session.export());
    }

    #[test]
    fn sessions_do_not_mix() {
        let mut api = LocalAgentApi::new(ApiLimits::default());
        api.handle(req("POST", "/v1/sessions", "id=a", "c"));
        api.handle(req("POST", "/v1/sessions", "id=b", "c"));
        api.handle(req("POST", "/v1/sessions/a/messages", "alpha", "c"));
        api.handle(req("POST", "/v1/sessions/b/messages", "beta", "c"));
        let a = api.handle(req("GET", "/v1/sessions/a", "", "c"));
        let b = api.handle(req("GET", "/v1/sessions/b", "", "c"));
        assert!(a.body.contains("alpha"));
        assert!(!a.body.contains("beta"));
        assert!(b.body.contains("beta"));
        assert!(!b.body.contains("alpha"));
    }

    #[test]
    fn limits_and_incompatible_schema() {
        let mut api = LocalAgentApi::new(ApiLimits {
            max_body_bytes: 4,
            max_sessions: 1,
            max_concurrent: 4,
            max_wall_ms: 1_000,
        });
        let large = api.handle(req("POST", "/v1/sessions", "id=toolong", "c"));
        assert_eq!(large.status, 413);
        let mut api = LocalAgentApi::new(ApiLimits::default());
        api.handle(req("POST", "/v1/sessions", "id=s", "c"));
        let bad = api.handle(req(
            "POST",
            "/v1/sessions/s/messages",
            "/load auralis_session=9\nsession_id=x\ngeneration=0\nstatus=active\ntools=0\nplanner=0\ncheckpoint=none\n",
            "c",
        ));
        assert_eq!(bad.status, 409);
        assert!(bad.body.contains("incompatible"));
    }

    #[test]
    fn loopback_http_preserves_correlation_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut api = LocalAgentApi::new(ApiLimits::default());
            serve_one(&listener, &mut api).unwrap();
        });
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let body = "id=http-50";
        let payload = format!(
            "POST /v1/sessions HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Correlation-Id: corr-50\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(payload.as_bytes()).unwrap();
        let _ = stream.shutdown(Shutdown::Write);
        let mut buf = String::new();
        stream.read_to_string(&mut buf).unwrap();
        assert!(buf.contains("X-Correlation-Id: corr-50"));
        assert!(buf.contains("session=http-50"));
        handle.join().unwrap();
    }
}
