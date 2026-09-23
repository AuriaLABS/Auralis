//! Versioned, typed protocol for tool definitions, calls and responses.
//! No real shell/network/browser execution lives in this module.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const TOOL_PROTOCOL_SCHEMA_VERSION: u32 = 1;
pub const MAX_TOOL_NAME_LEN: usize = 64;
pub const MAX_CALL_ID_LEN: usize = 96;
pub const MAX_ARGS: usize = 64;
pub const MAX_STRING_ARG_LEN: usize = 16 * 1024;
pub const MAX_RESULT_BYTES: usize = 1024 * 1024;
pub const MAX_TIMEOUT_MS: u64 = 300_000;
const MAX_PROTOCOL_TEXT_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolArgKind {
    String { min_len: usize, max_len: usize },
    Integer { min: i64, max: i64 },
    Boolean,
    Enum { variants: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolArgSpec {
    pub name: String,
    pub required: bool,
    pub kind: ToolArgKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    pub schema_version: u32,
    pub name: String,
    pub version: u32,
    pub capability: String,
    pub mutative: bool,
    pub timeout_ms: u64,
    pub max_result_bytes: usize,
    pub args: Vec<ToolArgSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolValue {
    String(String),
    Integer(i64),
    Boolean(bool),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolArgument {
    pub name: String,
    pub value: ToolValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRequest {
    pub schema_version: u32,
    pub call_id: String,
    pub tool_name: String,
    pub tool_version: u32,
    pub arguments: Vec<ToolArgument>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolErrorSeverity {
    Recoverable,
    Fatal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolResult {
    pub schema_version: u32,
    pub call_id: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolError {
    pub schema_version: u32,
    pub call_id: String,
    pub severity: ToolErrorSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolResponse {
    Result(ToolResult),
    Error(ToolError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolProtocolError {
    Decode(String),
    Definition(String),
    Request(String),
}

impl fmt::Display for ToolProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(v) => write!(f, "tool protocol decode error: {v}"),
            Self::Definition(v) => write!(f, "invalid tool definition: {v}"),
            Self::Request(v) => write!(f, "invalid tool request: {v}"),
        }
    }
}
impl std::error::Error for ToolProtocolError {}

impl ToolErrorSeverity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Recoverable => "recoverable",
            Self::Fatal => "fatal",
        }
    }
    fn parse(text: &str) -> Result<Self, ToolProtocolError> {
        match text {
            "recoverable" => Ok(Self::Recoverable),
            "fatal" => Ok(Self::Fatal),
            other => Err(ToolProtocolError::Decode(format!(
                "unknown tool error severity {other}"
            ))),
        }
    }
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        version: u32,
        capability: impl Into<String>,
        mutative: bool,
        timeout_ms: u64,
        max_result_bytes: usize,
        args: Vec<ToolArgSpec>,
    ) -> Result<Self, ToolProtocolError> {
        let out = Self {
            schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
            name: name.into(),
            version,
            capability: capability.into(),
            mutative,
            timeout_ms,
            max_result_bytes,
            args,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<(), ToolProtocolError> {
        if self.schema_version != TOOL_PROTOCOL_SCHEMA_VERSION {
            return Err(ToolProtocolError::Definition(format!(
                "schema {} is unsupported (expected {})",
                self.schema_version, TOOL_PROTOCOL_SCHEMA_VERSION
            )));
        }
        validate_identifier("tool name", &self.name, MAX_TOOL_NAME_LEN)?;
        validate_identifier("capability", &self.capability, MAX_TOOL_NAME_LEN)?;
        if self.version == 0 {
            return Err(ToolProtocolError::Definition(
                "tool version must be positive".into(),
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(ToolProtocolError::Definition(format!(
                "timeout_ms must be in 1..={MAX_TIMEOUT_MS}"
            )));
        }
        if self.max_result_bytes == 0 || self.max_result_bytes > MAX_RESULT_BYTES {
            return Err(ToolProtocolError::Definition(format!(
                "max_result_bytes must be in 1..={MAX_RESULT_BYTES}"
            )));
        }
        if self.args.len() > MAX_ARGS {
            return Err(ToolProtocolError::Definition("too many arguments".into()));
        }

        let mut names = BTreeSet::new();
        for spec in &self.args {
            validate_identifier("argument name", &spec.name, MAX_TOOL_NAME_LEN)?;
            if !names.insert(spec.name.as_str()) {
                return Err(ToolProtocolError::Definition(format!(
                    "duplicate argument definition {}",
                    spec.name
                )));
            }
            match &spec.kind {
                ToolArgKind::String { min_len, max_len } => {
                    if min_len > max_len || *max_len > MAX_STRING_ARG_LEN {
                        return Err(ToolProtocolError::Definition(format!(
                            "invalid string bounds for {}",
                            spec.name
                        )));
                    }
                }
                ToolArgKind::Integer { min, max } if min > max => {
                    return Err(ToolProtocolError::Definition(format!(
                        "invalid integer bounds for {}",
                        spec.name
                    )));
                }
                ToolArgKind::Enum { variants } => {
                    if variants.is_empty() || variants.len() > MAX_ARGS {
                        return Err(ToolProtocolError::Definition(format!(
                            "invalid enum variant count for {}",
                            spec.name
                        )));
                    }
                    let mut seen = BTreeSet::new();
                    for variant in variants {
                        validate_identifier("enum variant", variant, MAX_TOOL_NAME_LEN)?;
                        if !seen.insert(variant.as_str()) {
                            return Err(ToolProtocolError::Definition(format!(
                                "duplicate enum variant {variant}"
                            )));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> String {
        self.validate().expect("valid tool definition");
        let mut out = String::new();
        field(&mut out, "auralis_tool_definition", &self.schema_version.to_string());
        field(&mut out, "name", &escape(&self.name));
        field(&mut out, "version", &self.version.to_string());
        field(&mut out, "capability", &escape(&self.capability));
        field(&mut out, "mutative", bool_text(self.mutative));
        field(&mut out, "timeout_ms", &self.timeout_ms.to_string());
        field(&mut out, "max_result_bytes", &self.max_result_bytes.to_string());
        field(&mut out, "arg_count", &self.args.len().to_string());
        for (i, spec) in self.args.iter().enumerate() {
            field(&mut out, &format!("arg.{i}.name"), &escape(&spec.name));
            field(&mut out, &format!("arg.{i}.required"), bool_text(spec.required));
            match &spec.kind {
                ToolArgKind::String { min_len, max_len } => {
                    field(&mut out, &format!("arg.{i}.kind"), "string");
                    field(&mut out, &format!("arg.{i}.min_len"), &min_len.to_string());
                    field(&mut out, &format!("arg.{i}.max_len"), &max_len.to_string());
                }
                ToolArgKind::Integer { min, max } => {
                    field(&mut out, &format!("arg.{i}.kind"), "integer");
                    field(&mut out, &format!("arg.{i}.min"), &min.to_string());
                    field(&mut out, &format!("arg.{i}.max"), &max.to_string());
                }
                ToolArgKind::Boolean => field(&mut out, &format!("arg.{i}.kind"), "boolean"),
                ToolArgKind::Enum { variants } => {
                    field(&mut out, &format!("arg.{i}.kind"), "enum");
                    field(
                        &mut out,
                        &format!("arg.{i}.variant_count"),
                        &variants.len().to_string(),
                    );
                    for (j, variant) in variants.iter().enumerate() {
                        field(
                            &mut out,
                            &format!("arg.{i}.variant.{j}"),
                            &escape(variant),
                        );
                    }
                }
            }
        }
        out
    }

    pub fn decode(text: &str) -> Result<Self, ToolProtocolError> {
        let mut m = parse_fields(text)?;
        let schema_version = take_parse(&mut m, "auralis_tool_definition")?;
        let name = unescape(&take(&mut m, "name")?)?;
        let version = take_parse(&mut m, "version")?;
        let capability = unescape(&take(&mut m, "capability")?)?;
        let mutative = parse_bool(&take(&mut m, "mutative")?)?;
        let timeout_ms = take_parse(&mut m, "timeout_ms")?;
        let max_result_bytes = take_parse(&mut m, "max_result_bytes")?;
        let count: usize = take_parse(&mut m, "arg_count")?;
        if count > MAX_ARGS {
            return Err(ToolProtocolError::Decode("arg_count exceeds limit".into()));
        }
        let mut args = Vec::with_capacity(count);
        for i in 0..count {
            let name = unescape(&take(&mut m, &format!("arg.{i}.name"))?)?;
            let required = parse_bool(&take(&mut m, &format!("arg.{i}.required"))?)?;
            let kind = match take(&mut m, &format!("arg.{i}.kind"))?.as_str() {
                "string" => ToolArgKind::String {
                    min_len: take_parse(&mut m, &format!("arg.{i}.min_len"))?,
                    max_len: take_parse(&mut m, &format!("arg.{i}.max_len"))?,
                },
                "integer" => ToolArgKind::Integer {
                    min: take_parse(&mut m, &format!("arg.{i}.min"))?,
                    max: take_parse(&mut m, &format!("arg.{i}.max"))?,
                },
                "boolean" => ToolArgKind::Boolean,
                "enum" => {
                    let n: usize = take_parse(&mut m, &format!("arg.{i}.variant_count"))?;
                    if n > MAX_ARGS {
                        return Err(ToolProtocolError::Decode(
                            "enum variant count exceeds limit".into(),
                        ));
                    }
                    let mut variants = Vec::with_capacity(n);
                    for j in 0..n {
                        variants.push(unescape(&take(
                            &mut m,
                            &format!("arg.{i}.variant.{j}"),
                        )?)?);
                    }
                    ToolArgKind::Enum { variants }
                }
                other => {
                    return Err(ToolProtocolError::Decode(format!(
                        "unknown argument kind {other}"
                    )))
                }
            };
            args.push(ToolArgSpec { name, required, kind });
        }
        reject_leftovers(m)?;
        let out = Self {
            schema_version,
            name,
            version,
            capability,
            mutative,
            timeout_ms,
            max_result_bytes,
            args,
        };
        out.validate()?;
        Ok(out)
    }
}

impl ToolRequest {
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_version: u32,
        arguments: Vec<ToolArgument>,
    ) -> Self {
        Self {
            schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            tool_version,
            arguments,
        }
    }

    pub fn validate_against(&self, definition: &ToolDefinition) -> Result<(), ToolProtocolError> {
        definition.validate()?;
        if self.schema_version != TOOL_PROTOCOL_SCHEMA_VERSION {
            return Err(ToolProtocolError::Request("unsupported request schema".into()));
        }
        validate_identifier("call id", &self.call_id, MAX_CALL_ID_LEN)
            .map_err(|e| ToolProtocolError::Request(e.to_string()))?;
        if self.tool_name != definition.name || self.tool_version != definition.version {
            return Err(ToolProtocolError::Request("tool identity mismatch".into()));
        }
        if self.arguments.len() > MAX_ARGS {
            return Err(ToolProtocolError::Request("too many arguments".into()));
        }
        let specs = definition
            .args
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeSet::new();
        for arg in &self.arguments {
            if !seen.insert(arg.name.as_str()) {
                return Err(ToolProtocolError::Request(format!(
                    "duplicate argument {}",
                    arg.name
                )));
            }
            let spec = specs.get(arg.name.as_str()).ok_or_else(|| {
                ToolProtocolError::Request(format!("unknown argument {}", arg.name))
            })?;
            validate_value(spec, &arg.value)?;
        }
        for spec in &definition.args {
            if spec.required && !seen.contains(spec.name.as_str()) {
                return Err(ToolProtocolError::Request(format!(
                    "missing required argument {}",
                    spec.name
                )));
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> String {
        let mut out = String::new();
        field(&mut out, "auralis_tool_request", &self.schema_version.to_string());
        field(&mut out, "call_id", &escape(&self.call_id));
        field(&mut out, "tool_name", &escape(&self.tool_name));
        field(&mut out, "tool_version", &self.tool_version.to_string());
        field(&mut out, "arg_count", &self.arguments.len().to_string());
        for (i, arg) in self.arguments.iter().enumerate() {
            field(&mut out, &format!("arg.{i}.name"), &escape(&arg.name));
            match &arg.value {
                ToolValue::String(v) => {
                    field(&mut out, &format!("arg.{i}.type"), "string");
                    field(&mut out, &format!("arg.{i}.value"), &escape(v));
                }
                ToolValue::Integer(v) => {
                    field(&mut out, &format!("arg.{i}.type"), "integer");
                    field(&mut out, &format!("arg.{i}.value"), &v.to_string());
                }
                ToolValue::Boolean(v) => {
                    field(&mut out, &format!("arg.{i}.type"), "boolean");
                    field(&mut out, &format!("arg.{i}.value"), bool_text(*v));
                }
            }
        }
        out
    }

    pub fn decode(text: &str) -> Result<Self, ToolProtocolError> {
        let mut m = parse_fields(text)?;
        let schema_version = take_parse(&mut m, "auralis_tool_request")?;
        let call_id = unescape(&take(&mut m, "call_id")?)?;
        let tool_name = unescape(&take(&mut m, "tool_name")?)?;
        let tool_version = take_parse(&mut m, "tool_version")?;
        let count: usize = take_parse(&mut m, "arg_count")?;
        if count > MAX_ARGS {
            return Err(ToolProtocolError::Decode("arg_count exceeds limit".into()));
        }
        let mut arguments = Vec::with_capacity(count);
        for i in 0..count {
            let name = unescape(&take(&mut m, &format!("arg.{i}.name"))?)?;
            let raw = take(&mut m, &format!("arg.{i}.value"))?;
            let value = match take(&mut m, &format!("arg.{i}.type"))?.as_str() {
                "string" => ToolValue::String(unescape(&raw)?),
                "integer" => ToolValue::Integer(raw.parse().map_err(|_| {
                    ToolProtocolError::Decode(format!("invalid integer for {name}"))
                })?),
                "boolean" => ToolValue::Boolean(parse_bool(&raw)?),
                other => {
                    return Err(ToolProtocolError::Decode(format!(
                        "unknown request argument type {other}"
                    )))
                }
            };
            arguments.push(ToolArgument { name, value });
        }
        reject_leftovers(m)?;
        Ok(Self {
            schema_version,
            call_id,
            tool_name,
            tool_version,
            arguments,
        })
    }
}

impl ToolResponse {
    pub fn result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Result(ToolResult {
            schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
            call_id: call_id.into(),
            content: content.into(),
        })
    }

    pub fn error(
        call_id: impl Into<String>,
        severity: ToolErrorSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Error(ToolError {
            schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
            call_id: call_id.into(),
            severity,
            code: code.into(),
            message: message.into(),
        })
    }

    pub fn call_id(&self) -> &str {
        match self {
            Self::Result(v) => &v.call_id,
            Self::Error(v) => &v.call_id,
        }
    }

    pub fn validate_for(
        &self,
        definition: &ToolDefinition,
        expected_call_id: &str,
    ) -> Result<(), ToolProtocolError> {
        if self.call_id() != expected_call_id {
            return Err(ToolProtocolError::Request(
                "response call id mismatch".into(),
            ));
        }
        match self {
            Self::Result(v) => {
                if v.schema_version != TOOL_PROTOCOL_SCHEMA_VERSION {
                    return Err(ToolProtocolError::Request(
                        "unsupported result schema".into(),
                    ));
                }
                if v.content.len() > definition.max_result_bytes {
                    return Err(ToolProtocolError::Request(
                        "result exceeds max_result_bytes".into(),
                    ));
                }
            }
            Self::Error(v) => {
                if v.schema_version != TOOL_PROTOCOL_SCHEMA_VERSION {
                    return Err(ToolProtocolError::Request(
                        "unsupported error schema".into(),
                    ));
                }
                validate_identifier("error code", &v.code, MAX_TOOL_NAME_LEN)
                    .map_err(|e| ToolProtocolError::Request(e.to_string()))?;
                if v.message.chars().count() > 4096 {
                    return Err(ToolProtocolError::Request(
                        "tool error message exceeds limit".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> String {
        let mut out = String::new();
        match self {
            Self::Result(v) => {
                field(&mut out, "auralis_tool_result", &v.schema_version.to_string());
                field(&mut out, "call_id", &escape(&v.call_id));
                field(&mut out, "content", &escape(&v.content));
            }
            Self::Error(v) => {
                field(&mut out, "auralis_tool_error", &v.schema_version.to_string());
                field(&mut out, "call_id", &escape(&v.call_id));
                field(&mut out, "severity", v.severity.as_str());
                field(&mut out, "code", &escape(&v.code));
                field(&mut out, "message", &escape(&v.message));
            }
        }
        out
    }

    pub fn decode(text: &str) -> Result<Self, ToolProtocolError> {
        let mut m = parse_fields(text)?;
        if m.contains_key("auralis_tool_result") {
            let schema_version = take_parse(&mut m, "auralis_tool_result")?;
            let call_id = unescape(&take(&mut m, "call_id")?)?;
            let content = unescape(&take(&mut m, "content")?)?;
            reject_leftovers(m)?;
            return Ok(Self::Result(ToolResult {
                schema_version,
                call_id,
                content,
            }));
        }
        if m.contains_key("auralis_tool_error") {
            let schema_version = take_parse(&mut m, "auralis_tool_error")?;
            let call_id = unescape(&take(&mut m, "call_id")?)?;
            let severity = ToolErrorSeverity::parse(&take(&mut m, "severity")?)?;
            let code = unescape(&take(&mut m, "code")?)?;
            let message = unescape(&take(&mut m, "message")?)?;
            reject_leftovers(m)?;
            return Ok(Self::Error(ToolError {
                schema_version,
                call_id,
                severity,
                code,
                message,
            }));
        }
        Err(ToolProtocolError::Decode(
            "missing tool result/error schema header".into(),
        ))
    }
}

pub struct ValidatedToolCall<'a> {
    definition: &'a ToolDefinition,
    request: &'a ToolRequest,
}
impl<'a> ValidatedToolCall<'a> {
    pub fn definition(&self) -> &'a ToolDefinition {
        self.definition
    }
    pub fn request(&self) -> &'a ToolRequest {
        self.request
    }
}

pub trait ToolExecutor {
    fn execute(&mut self, call: ValidatedToolCall<'_>) -> ToolResponse;
}

pub fn invoke_validated<E: ToolExecutor>(
    definition: &ToolDefinition,
    request: &ToolRequest,
    executor: &mut E,
) -> ToolResponse {
    if let Err(error) = request.validate_against(definition) {
        return ToolResponse::error(
            request.call_id.clone(),
            ToolErrorSeverity::Fatal,
            "protocol-validation",
            error.to_string(),
        );
    }
    let response = executor.execute(ValidatedToolCall {
        definition,
        request,
    });
    match response.validate_for(definition, &request.call_id) {
        Ok(()) => response,
        Err(error) => ToolResponse::error(
            request.call_id.clone(),
            ToolErrorSeverity::Fatal,
            "invalid-tool-response",
            error.to_string(),
        ),
    }
}

#[derive(Clone, Debug)]
pub enum MockMode {
    EchoArgument(String),
    FixedResult(String),
    FixedError {
        severity: ToolErrorSeverity,
        code: String,
        message: String,
    },
}

#[derive(Clone, Debug)]
pub struct DeterministicMockExecutor {
    pub invocations: usize,
    pub mode: MockMode,
}
impl DeterministicMockExecutor {
    pub fn new(mode: MockMode) -> Self {
        Self {
            invocations: 0,
            mode,
        }
    }
}
impl ToolExecutor for DeterministicMockExecutor {
    fn execute(&mut self, call: ValidatedToolCall<'_>) -> ToolResponse {
        self.invocations += 1;
        match &self.mode {
            MockMode::EchoArgument(name) => {
                let content = call
                    .request()
                    .arguments
                    .iter()
                    .find(|a| a.name == *name)
                    .map(|a| match &a.value {
                        ToolValue::String(v) => v.clone(),
                        ToolValue::Integer(v) => v.to_string(),
                        ToolValue::Boolean(v) => v.to_string(),
                    })
                    .unwrap_or_default();
                ToolResponse::result(call.request().call_id.clone(), content)
            }
            MockMode::FixedResult(v) => {
                ToolResponse::result(call.request().call_id.clone(), v.clone())
            }
            MockMode::FixedError {
                severity,
                code,
                message,
            } => ToolResponse::error(
                call.request().call_id.clone(),
                *severity,
                code.clone(),
                message.clone(),
            ),
        }
    }
}

fn validate_value(spec: &ToolArgSpec, value: &ToolValue) -> Result<(), ToolProtocolError> {
    match (&spec.kind, value) {
        (ToolArgKind::String { min_len, max_len }, ToolValue::String(v)) => {
            let n = v.chars().count();
            if n < *min_len || n > *max_len {
                return Err(ToolProtocolError::Request(format!(
                    "argument {} string length out of bounds",
                    spec.name
                )));
            }
        }
        (ToolArgKind::Integer { min, max }, ToolValue::Integer(v)) => {
            if v < min || v > max {
                return Err(ToolProtocolError::Request(format!(
                    "argument {} integer out of bounds",
                    spec.name
                )));
            }
        }
        (ToolArgKind::Boolean, ToolValue::Boolean(_)) => {}
        (ToolArgKind::Enum { variants }, ToolValue::String(v)) => {
            if !variants.iter().any(|x| x == v) {
                return Err(ToolProtocolError::Request(format!(
                    "argument {} enum value is not allowed",
                    spec.name
                )));
            }
        }
        _ => {
            return Err(ToolProtocolError::Request(format!(
                "argument {} has wrong type",
                spec.name
            )))
        }
    }
    Ok(())
}

fn validate_identifier(
    label: &str,
    text: &str,
    max_len: usize,
) -> Result<(), ToolProtocolError> {
    if text.is_empty() || text.len() > max_len {
        return Err(ToolProtocolError::Definition(format!(
            "{label} length must be in 1..={max_len}"
        )));
    }
    if !text.bytes().all(|b| {
        b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')
    }) {
        return Err(ToolProtocolError::Definition(format!(
            "{label} contains invalid characters"
        )));
    }
    Ok(())
}

fn bool_text(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}
fn parse_bool(v: &str) -> Result<bool, ToolProtocolError> {
    match v {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ToolProtocolError::Decode("invalid boolean".into())),
    }
}
fn field(out: &mut String, key: &str, value: &str) {
    out.push_str(key);
    out.push('=');
    out.push_str(value);
    out.push('\n');
}
fn parse_fields(text: &str) -> Result<BTreeMap<String, String>, ToolProtocolError> {
    if text.len() > MAX_PROTOCOL_TEXT_BYTES {
        return Err(ToolProtocolError::Decode(
            "protocol text exceeds size limit".into(),
        ));
    }
    let mut out = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| {
            ToolProtocolError::Decode(format!("invalid line {}", i + 1))
        })?;
        if key.is_empty() {
            return Err(ToolProtocolError::Decode("empty field name".into()));
        }
        if out.insert(key.to_string(), value.to_string()).is_some() {
            return Err(ToolProtocolError::Decode(format!(
                "duplicate field {key}"
            )));
        }
    }
    Ok(out)
}
fn take(
    fields: &mut BTreeMap<String, String>,
    key: &str,
) -> Result<String, ToolProtocolError> {
    fields
        .remove(key)
        .ok_or_else(|| ToolProtocolError::Decode(format!("missing field {key}")))
}
fn take_parse<T>(
    fields: &mut BTreeMap<String, String>,
    key: &str,
) -> Result<T, ToolProtocolError>
where
    T: std::str::FromStr,
{
    take(fields, key)?
        .parse()
        .map_err(|_| ToolProtocolError::Decode(format!("invalid field {key}")))
}
fn reject_leftovers(fields: BTreeMap<String, String>) -> Result<(), ToolProtocolError> {
    if let Some((key, _)) = fields.into_iter().next() {
        return Err(ToolProtocolError::Decode(format!("unknown field {key}")));
    }
    Ok(())
}
fn escape(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric()
            || matches!(b, b'_' | b'-' | b'.' | b' ' | b':' | b'/')
        {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex(b >> 4));
            out.push(hex(b & 15));
        }
    }
    out
}
fn unescape(text: &str) -> Result<String, ToolProtocolError> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(ToolProtocolError::Decode(
                    "truncated percent escape".into(),
                ));
            }
            out.push((from_hex(bytes[i + 1])? << 4) | from_hex(bytes[i + 2])?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out)
        .map_err(|_| ToolProtocolError::Decode("escaped value is not UTF-8".into()))
}
fn hex(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        _ => (b'A' + v - 10) as char,
    }
}
fn from_hex(v: u8) -> Result<u8, ToolProtocolError> {
    match v {
        b'0'..=b'9' => Ok(v - b'0'),
        b'a'..=b'f' => Ok(v - b'a' + 10),
        b'A'..=b'F' => Ok(v - b'A' + 10),
        _ => Err(ToolProtocolError::Decode(
            "invalid percent escape".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def() -> ToolDefinition {
        ToolDefinition::new(
            "fixture.echo",
            2,
            "fixture.read",
            false,
            1500,
            128,
            vec![
                ToolArgSpec {
                    name: "text".into(),
                    required: true,
                    kind: ToolArgKind::String {
                        min_len: 1,
                        max_len: 32,
                    },
                },
                ToolArgSpec {
                    name: "count".into(),
                    required: false,
                    kind: ToolArgKind::Integer { min: 1, max: 4 },
                },
                ToolArgSpec {
                    name: "mode".into(),
                    required: false,
                    kind: ToolArgKind::Enum {
                        variants: vec!["plain".into(), "upper".into()],
                    },
                },
            ],
        )
        .unwrap()
    }

    fn req() -> ToolRequest {
        ToolRequest::new(
            "session-1:call-7",
            "fixture.echo",
            2,
            vec![
                ToolArgument {
                    name: "text".into(),
                    value: ToolValue::String("hello=world\n✓".into()),
                },
                ToolArgument {
                    name: "count".into(),
                    value: ToolValue::Integer(2),
                },
                ToolArgument {
                    name: "mode".into(),
                    value: ToolValue::String("plain".into()),
                },
            ],
        )
    }

    #[test]
    fn definitions_requests_results_and_errors_roundtrip() {
        let d = def();
        assert_eq!(ToolDefinition::decode(&d.encode()).unwrap(), d);
        let r = req();
        let decoded = ToolRequest::decode(&r.encode()).unwrap();
        decoded.validate_against(&d).unwrap();
        assert_eq!(decoded, r);

        let result = ToolResponse::result("call-1", "line 1\nline=2");
        assert_eq!(ToolResponse::decode(&result.encode()).unwrap(), result);
        let error = ToolResponse::error(
            "call-2",
            ToolErrorSeverity::Recoverable,
            "fixture-timeout",
            "try again",
        );
        assert_eq!(ToolResponse::decode(&error.encode()).unwrap(), error);
    }

    #[test]
    fn invalid_calls_fail_before_executor() {
        let d = def();
        let mut runtime =
            DeterministicMockExecutor::new(MockMode::FixedResult("ok".into()));
        let mut missing = req();
        missing.arguments.retain(|a| a.name != "text");
        let response = invoke_validated(&d, &missing, &mut runtime);
        assert_eq!(runtime.invocations, 0);
        assert!(matches!(response, ToolResponse::Error(_)));

        let mut unknown = req();
        unknown.arguments.push(ToolArgument {
            name: "surprise".into(),
            value: ToolValue::Boolean(true),
        });
        let _ = invoke_validated(&d, &unknown, &mut runtime);
        assert_eq!(runtime.invocations, 0);
    }

    #[test]
    fn types_ranges_enums_duplicates_and_unknown_fields_fail_closed() {
        let d = def();

        let mut wrong = req();
        wrong.arguments[0].value = ToolValue::Integer(1);
        assert!(wrong.validate_against(&d).is_err());

        let mut range = req();
        range.arguments[1].value = ToolValue::Integer(99);
        assert!(range.validate_against(&d).is_err());

        let mut en = req();
        en.arguments[2].value = ToolValue::String("other".into());
        assert!(en.validate_against(&d).is_err());

        let mut dup = req();
        dup.arguments.push(dup.arguments[0].clone());
        assert!(dup.validate_against(&d).is_err());

        let encoded = d.encode();
        assert!(ToolDefinition::decode(&(encoded.clone() + "secret=never\n")).is_err());
        assert!(ToolDefinition::decode(&(encoded + "name=again\n")).is_err());
    }

    #[test]
    fn future_schema_and_malformed_escape_fail_closed() {
        let future = def()
            .encode()
            .replacen("auralis_tool_definition=1", "auralis_tool_definition=999", 1);
        assert!(ToolDefinition::decode(&future).is_err());

        let malformed = req().encode().replace("hello%3Dworld", "hello%QZworld");
        assert!(ToolRequest::decode(&malformed).is_err());
    }

    #[test]
    fn deterministic_mock_is_replaceable_and_correlated() {
        let d = def();
        let r = req();
        let mut a =
            DeterministicMockExecutor::new(MockMode::EchoArgument("text".into()));
        let mut b =
            DeterministicMockExecutor::new(MockMode::EchoArgument("text".into()));
        let ra = invoke_validated(&d, &r, &mut a);
        let rb = invoke_validated(&d, &r, &mut b);
        assert_eq!(ra, rb);
        assert_eq!(a.invocations, 1);
        assert_eq!(b.invocations, 1);
        assert_eq!(ra.call_id(), r.call_id);
    }

    #[test]
    fn invalid_executor_response_is_converted_to_fatal_protocol_error() {
        struct WrongCall;
        impl ToolExecutor for WrongCall {
            fn execute(&mut self, _call: ValidatedToolCall<'_>) -> ToolResponse {
                ToolResponse::result("wrong", "ok")
            }
        }
        let response = invoke_validated(&def(), &req(), &mut WrongCall);
        match response {
            ToolResponse::Error(e) => {
                assert_eq!(e.code, "invalid-tool-response");
                assert_eq!(e.severity, ToolErrorSeverity::Fatal);
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn result_size_limit_is_enforced() {
        let mut d = def();
        d.max_result_bytes = 4;
        let mut runtime =
            DeterministicMockExecutor::new(MockMode::FixedResult("too-long".into()));
        let response = invoke_validated(&d, &req(), &mut runtime);
        assert_eq!(runtime.invocations, 1);
        match response {
            ToolResponse::Error(e) => assert_eq!(e.code, "invalid-tool-response"),
            other => panic!("expected size error, got {other:?}"),
        }
    }
}
