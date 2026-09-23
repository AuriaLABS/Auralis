//! #49 versioned tool registry and deterministic local reference tools.
//!
//! Discovery is decoupled from execution. Definitions are validated through
//! #44 and the registry grants no authority: reference execution is exposed
//! only as a ToolExecutor that is intended to sit behind #45 authorization.
//! No shell, network, browser, credentials, arbitrary filesystem access or
//! generated-code execution lives here.

use crate::tool_protocol::{
    ToolArgKind, ToolArgSpec, ToolDefinition, ToolErrorSeverity, ToolExecutor, ToolProtocolError,
    ToolRequest, ToolResponse, ToolValue, ValidatedToolCall,
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
    reference_kind: Option<ReferenceKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceKind {
    Echo,
    Add,
    FixtureRead,
    Error,
    Latency,
    AgentLookup,
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
        registry.register_reference(echo_tool()?, ReferenceKind::Echo)?;
        registry.register_reference(add_tool()?, ReferenceKind::Add)?;
        registry.register_reference(fixture_read_tool()?, ReferenceKind::FixtureRead)?;
        registry.register_reference(error_tool()?, ReferenceKind::Error)?;
        registry.register_reference(latency_tool()?, ReferenceKind::Latency)?;
        registry.register_reference(agent_lookup_tool()?, ReferenceKind::AgentLookup)?;
        Ok(registry)
    }

    fn insert(
        &mut self,
        definition: ToolDefinition,
        reference_kind: Option<ReferenceKind>,
    ) -> Result<(), ToolRegistryError> {
        definition.validate()?;
        if self.entries.len() >= MAX_REGISTRY_TOOLS {
            return Err(ToolRegistryError::Capacity);
        }
        let id = identity(&definition.name, definition.version);
        if self.entries.contains_key(&id) {
            return Err(ToolRegistryError::Duplicate(id));
        }
        self.entries.insert(
            id,
            RegistryEntry {
                definition,
                reference_kind,
            },
        );
        Ok(())
    }

    fn register_reference(
        &mut self,
        definition: ToolDefinition,
        kind: ReferenceKind,
    ) -> Result<(), ToolRegistryError> {
        self.insert(definition, Some(kind))
    }

    pub fn register_definition(
        &mut self,
        definition: ToolDefinition,
    ) -> Result<(), ToolRegistryError> {
        if is_forbidden_name(&definition.name) {
            return Err(ToolRegistryError::Forbidden(definition.name));
        }
        self.insert(definition, None)
    }

    pub fn unregister(&mut self, name: &str, version: u32) -> Result<(), ToolRegistryError> {
        let id = identity(name, version);
        self.entries
            .remove(&id)
            .map(|_| ())
            .ok_or(ToolRegistryError::Unknown(id))
    }

    pub fn resolve(&self, name: &str, version: u32) -> Option<&ToolDefinition> {
        self.entries
            .get(&identity(name, version))
            .map(|entry| &entry.definition)
    }

    pub fn get(&self, name: &str, version: u32) -> Option<&ToolDefinition> {
        self.resolve(name, version)
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

    pub fn reference_executor(&self) -> ReferenceToolExecutor {
        let kinds = self
            .entries
            .iter()
            .filter_map(|(id, entry)| entry.reference_kind.map(|kind| (id.clone(), kind)))
            .collect();
        ReferenceToolExecutor {
            kinds,
            invocations: 0,
        }
    }

    pub fn canonical(&self) -> String {
        let mut out = format!("auralis_tool_registry={}\n", TOOL_REGISTRY_SCHEMA_VERSION);
        for entry in self.entries.values() {
            out.push_str(&format!(
                "tool={}@{}|capability={}|mutative={}|reference={}\n",
                entry.definition.name,
                entry.definition.version,
                entry.definition.capability,
                entry.definition.mutative,
                entry.reference_kind.is_some()
            ));
        }
        out
    }

    pub fn fingerprint(&self) -> u64 {
        fingerprint(self.canonical().as_bytes())
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceToolExecutor {
    kinds: BTreeMap<String, ReferenceKind>,
    invocations: usize,
}

impl ReferenceToolExecutor {
    pub fn invocations(&self) -> usize {
        self.invocations
    }
}

impl ToolExecutor for ReferenceToolExecutor {
    fn execute(&mut self, call: ValidatedToolCall<'_>) -> ToolResponse {
        self.invocations += 1;
        let definition = call.definition();
        let request = call.request();
        let id = identity(&definition.name, definition.version);
        let Some(kind) = self.kinds.get(&id).copied() else {
            return ToolResponse::error(
                request.call_id.clone(),
                ToolErrorSeverity::Fatal,
                "no-reference-executor",
                "registered definition has no local reference implementation",
            );
        };

        match kind {
            ReferenceKind::Echo => {
                ToolResponse::result(request.call_id.clone(), string_arg(request, "text"))
            }
            ReferenceKind::Add => {
                let left = integer_arg(request, "left");
                let right = integer_arg(request, "right");
                match left.checked_add(right) {
                    Some(sum) => ToolResponse::result(request.call_id.clone(), sum.to_string()),
                    None => ToolResponse::error(
                        request.call_id.clone(),
                        ToolErrorSeverity::Fatal,
                        "integer-overflow",
                        "reference addition overflowed",
                    ),
                }
            }
            ReferenceKind::FixtureRead => {
                let value = match string_arg(request, "key") {
                    "alpha" => "fixture-alpha",
                    "beta" => "fixture-beta",
                    _ => "fixture-unknown",
                };
                ToolResponse::result(request.call_id.clone(), value)
            }
            ReferenceKind::Error => ToolResponse::error(
                request.call_id.clone(),
                ToolErrorSeverity::Recoverable,
                string_arg(request, "code"),
                "injected reference error",
            ),
            ReferenceKind::Latency => ToolResponse::result(
                request.call_id.clone(),
                format!("ticks={}", integer_arg(request, "ticks")),
            ),
            ReferenceKind::AgentLookup => {
                let key = string_arg(request, "key");
                match agent_lookup(key) {
                    Some(value) => ToolResponse::result(request.call_id.clone(), value),
                    None => ToolResponse::error(
                        request.call_id.clone(),
                        ToolErrorSeverity::Recoverable,
                        "not-found",
                        "controlled lookup key does not exist",
                    ),
                }
            }
        }
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
                kind: ToolArgKind::Integer {
                    min: -1000,
                    max: 1000,
                },
            },
            ToolArgSpec {
                name: "right".into(),
                required: true,
                kind: ToolArgKind::Integer {
                    min: -1000,
                    max: 1000,
                },
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

fn agent_lookup_tool() -> Result<ToolDefinition, ToolRegistryError> {
    Ok(ToolDefinition::new(
        "lookup",
        1,
        "agent.lookup",
        false,
        100,
        128,
        vec![ToolArgSpec {
            name: "key".into(),
            required: true,
            kind: ToolArgKind::String {
                min_len: 1,
                max_len: 64,
            },
        }],
    )?)
}

fn agent_lookup(key: &str) -> Option<String> {
    if key.starts_with("missing-") {
        return None;
    }
    key.strip_prefix("city-")
        .and_then(|value| value.parse::<u64>().ok())
        .map(|index| (1000 + index).to_string())
}

fn string_arg<'a>(request: &'a ToolRequest, name: &str) -> &'a str {
    request
        .arguments
        .iter()
        .find(|argument| argument.name == name)
        .and_then(|argument| match &argument.value {
            ToolValue::String(value) => Some(value.as_str()),
            _ => None,
        })
        .expect("validated string argument")
}

fn integer_arg(request: &ToolRequest, name: &str) -> i64 {
    request
        .arguments
        .iter()
        .find(|argument| argument.name == name)
        .and_then(|argument| match &argument.value {
            ToolValue::Integer(value) => Some(*value),
            _ => None,
        })
        .expect("validated integer argument")
}

fn fingerprint(bytes: &[u8]) -> u64 {
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
    use crate::agent_eval::{generate_suite, AgentEvalProfile, AGENT_EVAL_SEED};
    use crate::agent_permissions::{
        AuthorizationReason, AuthorizedToolRuntime, PermissionPolicy,
    };
    use crate::tool_protocol::{ToolArgument, ToolResponse};

    fn string_request(call: &str, tool: &str, arg: &str, value: &str) -> ToolRequest {
        ToolRequest::new(
            call,
            tool,
            1,
            vec![ToolArgument {
                name: arg.into(),
                value: ToolValue::String(value.into()),
            }],
        )
    }

    fn echo_request(text: &str) -> ToolRequest {
        string_request("session-1:call-1", "ref.echo", "text", text)
    }

    #[test]
    fn reference_catalog_lists_capabilities_and_is_deterministic() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let listed = registry.list();
        assert_eq!(listed.len(), 6);
        let names: BTreeSet<_> = listed.iter().map(|item| item.name.as_str()).collect();
        for name in [
            "lookup",
            "ref.add",
            "ref.echo",
            "ref.error",
            "ref.fixture",
            "ref.latency",
        ] {
            assert!(names.contains(name), "reference registry missing {name}");
        }
        let caps = registry.capabilities();
        assert!(caps.contains("fixture.read"));
        assert!(caps.contains("math.read"));
        assert!(caps.contains("agent.lookup"));
        assert!(listed.iter().all(|item| !item.mutative));
        assert_eq!(
            registry.fingerprint(),
            ToolRegistry::with_reference_tools().unwrap().fingerprint()
        );
    }

    #[test]
    fn register_unregister_resolve_preserves_mutation_metadata() {
        let mut registry = ToolRegistry::new();
        let custom = ToolDefinition::new(
            "custom.note",
            1,
            "custom.write",
            true,
            100,
            64,
            vec![],
        )
        .unwrap();
        registry.register_definition(custom).unwrap();
        assert!(registry.resolve("custom.note", 1).unwrap().mutative);
        assert!(registry.list()[0].mutative);
        assert!(matches!(
            registry.register_definition(
                ToolDefinition::new("custom.note", 1, "custom.write", true, 100, 64, vec![])
                    .unwrap()
            ),
            Err(ToolRegistryError::Duplicate(_))
        ));
        registry.unregister("custom.note", 1).unwrap();
        assert!(registry.resolve("custom.note", 1).is_none());
        assert!(matches!(
            registry.unregister("custom.note", 1),
            Err(ToolRegistryError::Unknown(_))
        ));
    }

    #[test]
    fn forbidden_external_effect_names_are_rejected() {
        let mut registry = ToolRegistry::new();
        for name in ["shell.rm", "net.fetch", "browser.open", "fs.read", "code.exec"] {
            let definition =
                ToolDefinition::new(name, 1, "unsafe.effect", true, 100, 64, vec![]).unwrap();
            assert!(matches!(
                registry.register_definition(definition),
                Err(ToolRegistryError::Forbidden(_))
            ));
        }
    }

    #[test]
    fn permission_runtime_is_the_execution_gate() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let mut runtime =
            AuthorizedToolRuntime::new(registry.catalog(), PermissionPolicy::default()).unwrap();
        let mut executor = registry.reference_executor();

        let denied = runtime.invoke(&echo_request("hello"), false, &mut executor);
        assert!(matches!(denied, ToolResponse::Error(_)));
        assert_eq!(executor.invocations(), 0);
        assert_eq!(
            runtime.audit().last().unwrap().reason,
            AuthorizationReason::NotAllowlisted
        );

        runtime
            .set_policy(PermissionPolicy::default().allow_tool("ref.echo", 1))
            .unwrap();
        assert_eq!(
            runtime.invoke(&echo_request("hello"), false, &mut executor),
            ToolResponse::result("session-1:call-1", "hello")
        );
        assert_eq!(executor.invocations(), 1);
    }

    #[test]
    fn unknown_tool_fails_before_reference_executor() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let mut runtime = AuthorizedToolRuntime::new(
            registry.catalog(),
            PermissionPolicy::default().allow_capability("fixture.read"),
        )
        .unwrap();
        let mut executor = registry.reference_executor();
        let request = ToolRequest::new("session-1:call-2", "ref.missing", 1, vec![]);

        let response = runtime.invoke(&request, false, &mut executor);
        assert!(matches!(response, ToolResponse::Error(_)));
        assert_eq!(executor.invocations(), 0);
        assert_eq!(
            runtime.audit().last().unwrap().reason,
            AuthorizationReason::UnknownTool
        );
    }

    #[test]
    fn dynamically_registered_definition_is_not_misrouted_to_reference_logic() {
        let mut registry = ToolRegistry::with_reference_tools().unwrap();
        registry
            .register_definition(
                ToolDefinition::new(
                    "custom.echo",
                    1,
                    "custom.read",
                    false,
                    100,
                    64,
                    vec![ToolArgSpec {
                        name: "text".into(),
                        required: true,
                        kind: ToolArgKind::String {
                            min_len: 1,
                            max_len: 32,
                        },
                    }],
                )
                .unwrap(),
            )
            .unwrap();

        let mut runtime = AuthorizedToolRuntime::new(
            registry.catalog(),
            PermissionPolicy::default().allow_tool("custom.echo", 1),
        )
        .unwrap();
        let mut executor = registry.reference_executor();
        let response = runtime.invoke(
            &string_request("call:custom", "custom.echo", "text", "do-not-echo"),
            false,
            &mut executor,
        );
        match response {
            ToolResponse::Error(error) => assert_eq!(error.code, "no-reference-executor"),
            other => panic!("expected no-reference-executor, got {other:?}"),
        }
    }

    #[test]
    fn agent_eval_lookup_is_discoverable_and_matches_mock_contract() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        for profile in [AgentEvalProfile::Smoke, AgentEvalProfile::Full] {
            for case in generate_suite(profile, AGENT_EVAL_SEED) {
                for tool in case.allowed_tools {
                    assert!(
                        registry.resolve(tool, 1).is_some(),
                        "agent eval tool {tool} missing from registry"
                    );
                }
            }
        }

        let mut runtime = AuthorizedToolRuntime::new(
            registry.catalog(),
            PermissionPolicy::default().allow_tool("lookup", 1),
        )
        .unwrap();
        let mut executor = registry.reference_executor();

        assert_eq!(
            runtime.invoke(
                &string_request("call:city", "lookup", "key", "city-2"),
                false,
                &mut executor,
            ),
            ToolResponse::result("call:city", "1002")
        );

        match runtime.invoke(
            &string_request("call:missing", "lookup", "key", "missing-2"),
            false,
            &mut executor,
        ) {
            ToolResponse::Error(error) => {
                assert_eq!(error.severity, ToolErrorSeverity::Recoverable);
                assert_eq!(error.code, "not-found");
            }
            other => panic!("expected controlled lookup error, got {other:?}"),
        }
    }

    #[test]
    fn add_error_and_latency_reference_tools_are_deterministic() {
        let registry = ToolRegistry::with_reference_tools().unwrap();
        let mut runtime = AuthorizedToolRuntime::new(
            registry.catalog(),
            PermissionPolicy::default()
                .allow_tool("ref.add", 1)
                .allow_tool("ref.error", 1)
                .allow_tool("ref.latency", 1),
        )
        .unwrap();
        let mut executor = registry.reference_executor();

        let add = ToolRequest::new(
            "call:add",
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
        );
        assert_eq!(
            runtime.invoke(&add, false, &mut executor),
            ToolResponse::result("call:add", "7")
        );

        let latency = ToolRequest::new(
            "call:latency",
            "ref.latency",
            1,
            vec![ToolArgument {
                name: "ticks".into(),
                value: ToolValue::Integer(3),
            }],
        );
        assert_eq!(
            runtime.invoke(&latency, false, &mut executor),
            ToolResponse::result("call:latency", "ticks=3")
        );

        match runtime.invoke(
            &string_request("call:error", "ref.error", "code", "busy"),
            false,
            &mut executor,
        ) {
            ToolResponse::Error(error) => {
                assert_eq!(error.severity, ToolErrorSeverity::Recoverable);
                assert_eq!(error.code, "busy");
            }
            other => panic!("expected injected error, got {other:?}"),
        }
    }
}
