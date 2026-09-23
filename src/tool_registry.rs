//! #49 tool registry and local reference tools.
//!
//! The registry is decoupled from the model. Unknown tools fail before any
//! executor runs. Reference tools are in-process, deterministic and have no
//! shell, network, credential or arbitrary filesystem access.

use crate::tool_protocol::{
    invoke_validated, ToolArgKind, ToolArgSpec, ToolArgument, ToolDefinition, ToolErrorSeverity,
    ToolExecutor, ToolProtocolError, ToolRequest, ToolResponse, ToolValue, ValidatedToolCall,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const TOOL_REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const MAX_REGISTRY_TOOLS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolRegistryError {
    Protocol(String),
    Duplicate(String),
    Unknown(String),
    Capacity,
    Forbidden(String),
}

impl fmt::Display for ToolRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(msg) => write!(f, "protocol: {msg}"),
            Self::Duplicate(id) => write!(f, "duplicate tool {id}"),
            Self::Unknown(id) => write!(f, "unknown tool {id}"),
            Self::Capacity => write!(f, "registry capacity exceeded"),
            Self::Forbidden(msg) => write!(f, "forbidden: {msg}"),
        }
    }
}

impl Error for ToolRegistryError {}

impl From<ToolProtocolError> for ToolRegistryError {
    fn from(err: ToolProtocolError) -> Self {
        Self::Protocol(err.to_string())
    }
}

fn identity(name: &str, version: u32) -> String {
    format!("{name}@{version}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolListing {
    pub name: String,
    pub version: u32,
    pub capability: String,
    pub mutative: bool,
}

#[derive(Clone, Debug)]
struct RegistryEntry {
    definition: ToolDefinition,
    kind: ReferenceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceKind {
    Echo,
    Add,
    FixtureRead,
    Error,
    Latency,
}

#[derive(Clone, Debug)]
pub struct ToolRegistry {
    entries: BTreeMap<String, RegistryEntry>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn with_reference_tools() -> Result<Self, ToolRegistryError> {
        let mut registry = Self::new();
        registry.register(echo_tool()?, ReferenceKind::Echo)?;
        registry.register(add_tool()?, ReferenceKind::Add)?;
        registry.register(fixture_read_tool()?, ReferenceKind::FixtureRead)?;
        registry.register(error_tool()?, ReferenceKind::Error)?;
        registry.register(latency_tool()?, ReferenceKind::Latency)?;
        Ok(registry)
    }

    fn register(
        &mut self,
        definition: ToolDefinition,
        kind: ReferenceKind,
    ) -> Result<(), ToolRegistryError> {
        definition.validate()?;
        if self.entries.len() >= MAX_REGISTRY_TOOLS {
            return Err(ToolRegistryError::Capacity);
        }
        let id = identity(&definition.name, definition.version);
        if self.entries.contains_key(&id) {
            return Err(ToolRegistryError::Duplicate(id));
        }
        self.entries.insert(id, RegistryEntry { definition, kind });
        Ok(())
    }

    pub fn register_definition(
        &mut self,
        definition: ToolDefinition,
    ) -> Result<(), ToolRegistryError> {
        if is_forbidden_name(&definition.name) {
            return Err(ToolRegistryError::Forbidden(definition.name));
        }
        self.register(definition, ReferenceKind::Echo)
    }

    pub fn unregister(&mut self, name: &str, version: u32) -> Result<(), ToolRegistryError> {
        let id = identity(name, version);
        self.entries
            .remove(&id)
            .map(|_| ())
            .ok_or(ToolRegistryError::Unknown(id))
    }

    pub fn get(&self, name: &str, version: u32) -> Option<&ToolDefinition> {
        self.entries
            .get(&identity(name, version))
            .map(|entry| &entry.definition)
    }

    pub fn list(&self) -> Vec<ToolListing> {
        self.entries
            .values()
            .map(|entry| ToolListing {
                name: entry.definition.name.clone(),
                version: entry.definition.version,
                capability: entry.definition.capability.clone(),
                mutative: entry.definition.mutative,
            })
            .collect()
    }

    pub fn capabilities(&self) -> BTreeSet<String> {
        self.entries
            .values()
            .map(|entry| entry.definition.capability.clone())
            .collect()
    }

    pub fn catalog(&self) -> Vec<ToolDefinition> {
        self.entries
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    pub fn invoke(&self, request: &ToolRequest) -> Result<ToolResponse, ToolRegistryError> {
        request.validate_envelope()?;
        let id = identity(&request.tool_name, request.tool_version);
        let entry = self
            .entries
            .get(&id)
            .ok_or_else(|| ToolRegistryError::Unknown(id))?;
        let mut executor = ReferenceExecutor {
            kind: entry.kind,
            invocations: 0,
        };
        Ok(invoke_validated(&entry.definition, request, &mut executor))
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn is_forbidden_name(name: &str) -> bool {
    name.starts_with("shell.")
        || name.starts_with("net.")
        || name.starts_with("browser.")
        || name.starts_with("fs.")
        || name.contains("exec")
}

fn echo_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "ref.echo",
        1,
        "fixture.read",
        false,
        100,
        256,
        vec![ToolArgSpec {
            name: "text".into(),
            required: true,
            kind: ToolArgKind::String {
                min_len: 1,
                max_len: 64,
            },
        }],
    )?)
}

fn add_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "ref.add",
        1,
        "math.read",
        false,
        100,
        64,
        vec![
            ToolArgSpec {
                name: "left".into(),
                required: true,
                kind: ToolArgKind::Integer { min: -1000, max: 1000 },
            },
            ToolArgSpec {
                name: "right".into(),
                required: true,
                kind: ToolArgKind::Integer { min: -1000, max: 1000 },
            },
        ],
    )?)
}

fn fixture_read_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "ref.fixture",
        1,
        "fixture.read",
        false,
        100,
        256,
        vec![ToolArgSpec {
            name: "key".into(),
            required: true,
            kind: ToolArgKind::Enum {
                variants: vec!["alpha".into(), "beta".into()],
            },
        }],
    )?)
}

fn error_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "ref.error",
        1,
        "fixture.read",
        false,
        100,
        64,
        vec![ToolArgSpec {
            name: "code".into(),
            required: true,
            kind: ToolArgKind::Enum {
                variants: vec!["missing".into(), "busy".into()],
            },
        }],
    )?)
}

fn latency_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "ref.latency",
        1,
        "fixture.read",
        false,
        50,
        64,
        vec![ToolArgSpec {
            name: "ticks".into(),
            required: true,
            kind: ToolArgKind::Integer { min: 0, max: 3 },
        }],
    )?)
}

struct ReferenceExecutor {
    kind: ReferenceKind,
    invocations: usize,
}

impl ToolExecutor for ReferenceExecutor {
    fn execute(&mut self, call: ValidatedToolCall<'_>) -> ToolResponse {
        self.invocations += 1;
        let request = call.request();
        match self.kind {
            ReferenceKind::Echo => {
                let text = arg_string(request, "text").unwrap_or_default();
                ok(request, text)
            }
            ReferenceKind::Add => {
                let left = arg_int(request, "left").unwrap_or(0);
                let right = arg_int(request, "right").unwrap_or(0);
                ok(request, (left + right).to_string())
            }
            ReferenceKind::FixtureRead => {
                let key = arg_string(request, "key").unwrap_or_default();
                let value = match key.as_str() {
                    "alpha" => "fixture-alpha",
                    "beta" => "fixture-beta",
                    _ => "fixture-unknown",
                };
                ok(request, value.to_string())
            }
            ReferenceKind::Error => {
                let code = arg_string(request, "code").unwrap_or_else(|| "missing".into());
                ToolResponse::Error(crate::tool_protocol::ToolError {
                    schema_version: crate::tool_protocol::TOOL_PROTOCOL_SCHEMA_VERSION,
                    call_id: request.call_id.clone(),
                    severity: ToolErrorSeverity::Recoverable,
                    code,
                    message: "injected reference error".into(),
                })
            }
            ReferenceKind::Latency => {
                let ticks = arg_int(request, "ticks").unwrap_or(0);
                ok(request, format!("ticks={ticks}"))
            }
        }
    }
}

fn arg_string(request: &ToolRequest, name: &str) -> Option<String> {
    request.arguments.iter().find_map(|arg| {
        if arg.name == name {
            match &arg.value {
                ToolValue::String(value) => Some(value.clone()),
                other => Some(format!("{other:?}")),
            }
        } else {
            None
        }
    })
}

fn arg_int(request: &ToolRequest, name: &str) -> Option<i64> {
    request.arguments.iter().find_map(|arg| {
        if arg.name == name {
            match arg.value {
                ToolValue::Integer(value) => Some(value),
                _ => None,
            }
        } else {
            None
        }
    })
}

fn ok(request: &ToolRequest, content: String) -> ToolResponse {
    ToolResponse::Result(crate::tool_protocol::ToolResult {
        schema_version: crate::tool_protocol::TOOL_PROTOCOL_SCHEMA_VERSION,
        call_id: request.call_id.clone(),
        content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_req(text: &str) -> ToolRequest {
        ToolRequest::new(
            "session-1:call-1",
            "ref.echo",
            1,
            vec![ToolArgument {
                name: "text".into(),
                value: ToolValue::String(text.into()),
            }],
        )
    }

    #[test]
    fn reference_catalog_lists_and_capabilities() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let listed = registry.list();
        assert_eq!(listed.len(), 5);
        let names: BTreeSet<_> = listed.iter().map(|item| item.name.as_str()).collect();
        assert!(names.contains("ref.echo"));
        assert!(names.contains("ref.add"));
        assert!(names.contains("ref.fixture"));
        let caps = registry.capabilities();
        assert!(caps.contains("fixture.read"));
        assert!(caps.contains("math.read"));
        assert!(listed.iter().all(|item| !item.mutative));
    }

    #[test]
    fn unknown_tool_fails_before_execute() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let request = ToolRequest::new("session-1:call-2", "ref.missing", 1, vec![]);
        let err = registry.invoke(&request).unwrap_err();
        assert!(matches!(err, ToolRegistryError::Unknown(_)));
    }

    #[test]
    fn echo_and_add_are_deterministic() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let echo = registry.invoke(&echo_req("hello")).unwrap();
        match echo {
            ToolResponse::Result(result) => assert_eq!(result.content, "hello"),
            ToolResponse::Error(err) => panic!("{err:?}"),
        }
        let add = registry
            .invoke(&ToolRequest::new(
                "session-1:call-3",
                "ref.add",
                1,
                vec![
                    ToolArgument {
                        name: "left".into(),
                        value: ToolValue::Integer(2),
                    },
                    ToolArgument {
                        name: "right".into(),
                        value: ToolValue::Integer(5),
                    },
                ],
            ))
            .unwrap();
        match add {
            ToolResponse::Result(result) => assert_eq!(result.content, "7"),
            ToolResponse::Error(err) => panic!("{err:?}"),
        }
    }

    #[test]
    fn invalid_arguments_are_rejected_by_protocol() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let request = ToolRequest::new("session-1:call-4", "ref.add", 1, vec![]);
        let response = registry.invoke(&request).unwrap();
        assert!(matches!(response, ToolResponse::Error(_)));
    }

    #[test]
    fn register_unregister_roundtrip() {
        let mut registry = ToolRegistry::new();
        registry.register_definition(echo_tool().unwrap()).unwrap();
        assert_eq!(registry.list().len(), 1);
        registry.unregister("ref.echo", 1).unwrap();
        assert!(registry.list().is_empty());
        let err = registry.unregister("ref.echo", 1).unwrap_err();
        assert!(matches!(err, ToolRegistryError::Unknown(_)));
    }

    #[test]
    fn forbidden_names_are_rejected() {
        let mut registry = ToolRegistry::new();
        let def = ToolDefinition::new(
            "shell.rm",
            1,
            "fixture.read",
            true,
            100,
            32,
            vec![],
        )
        .unwrap();
        let err = registry.register_definition(def).unwrap_err();
        assert!(matches!(err, ToolRegistryError::Forbidden(_)));
    }

    #[test]
    fn catalog_is_usable_by_permission_runtime() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let catalog = registry.catalog();
        assert_eq!(catalog.len(), 5);
        for definition in &catalog {
            definition.validate().unwrap();
        }
    }
}
