//! #45 deny-by-default authorization and deterministic sandbox runtime.
//!
//! Validation is inherited from #44. This layer decides whether an already
//! typed request may reach an executor and records that decision before any
//! executor invocation. It provides no shell/network implementation.

use crate::tool_protocol::{
    invoke_validated, ToolDefinition, ToolErrorSeverity, ToolExecutor, ToolRequest, ToolResponse,
};
use std::collections::BTreeSet;

pub const PERMISSION_POLICY_SCHEMA_VERSION: u32 = 1;
pub const MAX_POLICY_ITEMS: usize = 256;
pub const MAX_AUDIT_EVENTS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationPolicy {
    Deny,
    RequireConfirmation,
    Allow,
}

impl MutationPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deny => "deny",
            Self::RequireConfirmation => "require-confirmation",
            Self::Allow => "allow",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermissionPolicy {
    pub schema_version: u32,
    pub allowed_tools: BTreeSet<String>,
    pub allowed_capabilities: BTreeSet<String>,
    pub mutation: MutationPolicy,
    pub max_timeout_ms: u64,
    pub max_result_bytes: usize,
    pub max_argument_bytes: usize,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            schema_version: PERMISSION_POLICY_SCHEMA_VERSION,
            allowed_tools: BTreeSet::new(),
            allowed_capabilities: BTreeSet::new(),
            mutation: MutationPolicy::Deny,
            max_timeout_ms: 1_000,
            max_result_bytes: 64 * 1024,
            max_argument_bytes: 16 * 1024,
        }
    }
}

impl PermissionPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PERMISSION_POLICY_SCHEMA_VERSION {
            return Err(format!(
                "permission policy schema {} is unsupported (expected {})",
                self.schema_version, PERMISSION_POLICY_SCHEMA_VERSION
            ));
        }
        if self.allowed_tools.len() > MAX_POLICY_ITEMS
            || self.allowed_capabilities.len() > MAX_POLICY_ITEMS
        {
            return Err(format!(
                "permission policy allowlist exceeds {MAX_POLICY_ITEMS} items"
            ));
        }
        if self.max_timeout_ms == 0 {
            return Err("permission policy max_timeout_ms must be positive".into());
        }
        if self.max_result_bytes == 0 {
            return Err("permission policy max_result_bytes must be positive".into());
        }
        if self.max_argument_bytes == 0 {
            return Err("permission policy max_argument_bytes must be positive".into());
        }
        for tool in &self.allowed_tools {
            if !valid_policy_id(tool) || !tool.contains('@') {
                return Err(format!("invalid allowed tool identity {tool}"));
            }
        }
        for capability in &self.allowed_capabilities {
            if !valid_policy_id(capability) {
                return Err(format!("invalid allowed capability {capability}"));
            }
        }
        Ok(())
    }

    pub fn allow_tool(mut self, name: &str, version: u32) -> Self {
        self.allowed_tools.insert(tool_identity(name, version));
        self
    }

    pub fn allow_capability(mut self, capability: &str) -> Self {
        self.allowed_capabilities.insert(capability.to_string());
        self
    }

    pub fn canonical(&self) -> String {
        let tools = self.allowed_tools.iter().cloned().collect::<Vec<_>>().join(",");
        let caps = self
            .allowed_capabilities
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "permission_policy={}
allowed_tools={}
allowed_capabilities={}
mutation={}
max_timeout_ms={}
max_result_bytes={}
max_argument_bytes={}
",
            self.schema_version,
            tools,
            caps,
            self.mutation.as_str(),
            self.max_timeout_ms,
            self.max_result_bytes,
            self.max_argument_bytes
        )
    }

    pub fn fingerprint(&self) -> u64 {
        fingerprint(self.canonical().as_bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorizationOutcome {
    Allow,
    Deny,
}

impl AuthorizationOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorizationReason {
    AllowedTool,
    AllowedCapability,
    ProtocolValidation,
    UnknownTool,
    NotAllowlisted,
    MutationDenied,
    ConfirmationRequired,
    TimeoutBudget,
    ResultBudget,
    ArgumentBudget,
}

impl AuthorizationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllowedTool => "allowed-tool",
            Self::AllowedCapability => "allowed-capability",
            Self::ProtocolValidation => "protocol-validation",
            Self::UnknownTool => "unknown-tool",
            Self::NotAllowlisted => "not-allowlisted",
            Self::MutationDenied => "mutation-denied",
            Self::ConfirmationRequired => "confirmation-required",
            Self::TimeoutBudget => "timeout-budget",
            Self::ResultBudget => "result-budget",
            Self::ArgumentBudget => "argument-budget",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationDecision {
    pub sequence: u64,
    pub call_id: String,
    pub tool_name: String,
    pub tool_version: u32,
    pub capability: String,
    pub mutative: bool,
    pub mutation_confirmed: bool,
    pub outcome: AuthorizationOutcome,
    pub reason: AuthorizationReason,
    pub policy_fingerprint: u64,
    pub declared_timeout_ms: u64,
    pub declared_max_result_bytes: usize,
    pub argument_bytes: usize,
}

impl AuthorizationDecision {
    pub fn canonical(&self) -> String {
        format!(
            "authorization | seq={} call_id={} tool={}@{} capability={} mutative={} confirmed={} outcome={} reason={} policy={:016x} timeout_ms={} max_result_bytes={} argument_bytes={}",
            self.sequence,
            self.call_id,
            self.tool_name,
            self.tool_version,
            self.capability,
            self.mutative,
            self.mutation_confirmed,
            self.outcome.as_str(),
            self.reason.as_str(),
            self.policy_fingerprint,
            self.declared_timeout_ms,
            self.declared_max_result_bytes,
            self.argument_bytes,
        )
    }
}

#[derive(Clone, Debug)]
pub struct AuthorizedToolRuntime {
    catalog: Vec<ToolDefinition>,
    policy: PermissionPolicy,
    audit: Vec<AuthorizationDecision>,
    next_sequence: u64,
}

impl AuthorizedToolRuntime {
    pub fn new(
        catalog: Vec<ToolDefinition>,
        policy: PermissionPolicy,
    ) -> Result<Self, String> {
        policy.validate()?;
        if catalog.len() > MAX_POLICY_ITEMS {
            return Err(format!("tool catalog exceeds {MAX_POLICY_ITEMS} definitions"));
        }
        let mut identities = BTreeSet::new();
        for definition in &catalog {
            definition.validate().map_err(|e| e.to_string())?;
            let identity = tool_identity(&definition.name, definition.version);
            if !identities.insert(identity.clone()) {
                return Err(format!("duplicate tool definition {identity}"));
            }
        }
        Ok(Self {
            catalog,
            policy,
            audit: Vec::new(),
            next_sequence: 1,
        })
    }

    pub fn policy(&self) -> &PermissionPolicy {
        &self.policy
    }

    pub fn set_policy(&mut self, policy: PermissionPolicy) -> Result<(), String> {
        policy.validate()?;
        self.policy = policy;
        Ok(())
    }

    pub fn audit(&self) -> &[AuthorizationDecision] {
        &self.audit
    }

    pub fn decision_for_call(&self, call_id: &str) -> Option<&AuthorizationDecision> {
        self.audit.iter().rev().find(|decision| decision.call_id == call_id)
    }

    pub fn explain_call(&self, call_id: &str) -> Option<String> {
        self.decision_for_call(call_id)
            .map(AuthorizationDecision::canonical)
    }

    pub fn clear_audit(&mut self) {
        self.audit.clear();
    }

    pub fn invoke<E: ToolExecutor>(
        &mut self,
        request: &ToolRequest,
        mutation_confirmed: bool,
        executor: &mut E,
    ) -> ToolResponse {
        if let Err(error) = request.validate_envelope() {
            return ToolResponse::error(
                "invalid-call",
                ToolErrorSeverity::Fatal,
                "protocol-validation",
                error.to_string(),
            );
        }

        let definition = match self.definition_for(request) {
            Some(definition) => definition.clone(),
            None => {
                self.record(
                    request,
                    None,
                    mutation_confirmed,
                    AuthorizationOutcome::Deny,
                    AuthorizationReason::UnknownTool,
                    request_argument_bytes(request),
                );
                return denied(request, "unknown-tool", "tool is not registered in runtime catalog");
            }
        };

        let argument_bytes = request_argument_bytes(request);
        if let Err(error) = request.validate_against(&definition) {
            self.record(
                request,
                Some(&definition),
                mutation_confirmed,
                AuthorizationOutcome::Deny,
                AuthorizationReason::ProtocolValidation,
                argument_bytes,
            );
            return ToolResponse::error(
                request.call_id.clone(),
                ToolErrorSeverity::Fatal,
                "protocol-validation",
                error.to_string(),
            );
        }

        let (outcome, reason) = self.authorize(&definition, mutation_confirmed, argument_bytes);
        self.record(
            request,
            Some(&definition),
            mutation_confirmed,
            outcome,
            reason,
            argument_bytes,
        );
        if outcome == AuthorizationOutcome::Deny {
            return denied(
                request,
                "authorization-denied",
                &format!("permission policy denied call: {}", reason.as_str()),
            );
        }

        invoke_validated(&definition, request, executor)
    }

    fn definition_for(&self, request: &ToolRequest) -> Option<&ToolDefinition> {
        self.catalog.iter().find(|definition| {
            definition.name == request.tool_name && definition.version == request.tool_version
        })
    }

    fn authorize(
        &self,
        definition: &ToolDefinition,
        mutation_confirmed: bool,
        argument_bytes: usize,
    ) -> (AuthorizationOutcome, AuthorizationReason) {
        let exact_allowed = self
            .policy
            .allowed_tools
            .contains(&tool_identity(&definition.name, definition.version));
        let capability_allowed = self
            .policy
            .allowed_capabilities
            .contains(&definition.capability);
        if !exact_allowed && !capability_allowed {
            return (AuthorizationOutcome::Deny, AuthorizationReason::NotAllowlisted);
        }
        if definition.timeout_ms > self.policy.max_timeout_ms {
            return (AuthorizationOutcome::Deny, AuthorizationReason::TimeoutBudget);
        }
        if definition.max_result_bytes > self.policy.max_result_bytes {
            return (AuthorizationOutcome::Deny, AuthorizationReason::ResultBudget);
        }
        if argument_bytes > self.policy.max_argument_bytes {
            return (AuthorizationOutcome::Deny, AuthorizationReason::ArgumentBudget);
        }
        if definition.mutative {
            match self.policy.mutation {
                MutationPolicy::Deny => {
                    return (AuthorizationOutcome::Deny, AuthorizationReason::MutationDenied)
                }
                MutationPolicy::RequireConfirmation if !mutation_confirmed => {
                    return (
                        AuthorizationOutcome::Deny,
                        AuthorizationReason::ConfirmationRequired,
                    )
                }
                MutationPolicy::RequireConfirmation | MutationPolicy::Allow => {}
            }
        }
        (
            AuthorizationOutcome::Allow,
            if exact_allowed {
                AuthorizationReason::AllowedTool
            } else {
                AuthorizationReason::AllowedCapability
            },
        )
    }

    fn record(
        &mut self,
        request: &ToolRequest,
        definition: Option<&ToolDefinition>,
        mutation_confirmed: bool,
        outcome: AuthorizationOutcome,
        reason: AuthorizationReason,
        argument_bytes: usize,
    ) {
        if self.audit.len() >= MAX_AUDIT_EVENTS {
            self.audit.remove(0);
        }
        let decision = AuthorizationDecision {
            sequence: self.next_sequence,
            call_id: request.call_id.clone(),
            tool_name: request.tool_name.clone(),
            tool_version: request.tool_version,
            capability: definition
                .map(|definition| definition.capability.clone())
                .unwrap_or_default(),
            mutative: definition.map(|definition| definition.mutative).unwrap_or(false),
            mutation_confirmed,
            outcome,
            reason,
            policy_fingerprint: self.policy.fingerprint(),
            declared_timeout_ms: definition.map(|definition| definition.timeout_ms).unwrap_or(0),
            declared_max_result_bytes: definition
                .map(|definition| definition.max_result_bytes)
                .unwrap_or(0),
            argument_bytes,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.audit.push(decision);
    }
}

fn request_argument_bytes(request: &ToolRequest) -> usize {
    request.encoded_argument_bytes()
}

fn tool_identity(name: &str, version: u32) -> String {
    format!("{name}@{version}")
}

fn denied(request: &ToolRequest, code: &str, message: &str) -> ToolResponse {
    ToolResponse::error(
        request.call_id.clone(),
        ToolErrorSeverity::Fatal,
        code,
        message,
    )
}

fn valid_policy_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 160
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'_' | b'-' | b'.' | b':' | b'@' | b'/')
        })
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
    use crate::tool_protocol::{
        DeterministicMockExecutor, MockMode, ToolArgKind, ToolArgSpec, ToolArgument, ToolValue,
    };

    fn read_definition() -> ToolDefinition {
        ToolDefinition::new(
            "fixture.read",
            1,
            "fixture.read",
            false,
            500,
            128,
            vec![ToolArgSpec {
                name: "key".into(),
                required: true,
                kind: ToolArgKind::String {
                    min_len: 1,
                    max_len: 32,
                },
            }],
        )
        .unwrap()
    }

    fn write_definition() -> ToolDefinition {
        ToolDefinition::new(
            "fixture.write",
            1,
            "fixture.write",
            true,
            500,
            128,
            vec![ToolArgSpec {
                name: "value".into(),
                required: true,
                kind: ToolArgKind::String {
                    min_len: 1,
                    max_len: 32,
                },
            }],
        )
        .unwrap()
    }

    fn request(tool: &str, arg: &str, value: &str) -> ToolRequest {
        ToolRequest::new(
            format!("session:call:{tool}"),
            tool,
            1,
            vec![ToolArgument {
                name: arg.into(),
                value: ToolValue::String(value.into()),
            }],
        )
    }

    #[test]
    fn default_policy_denies_registered_tool_without_execution() {
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], PermissionPolicy::default())
                .unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("value".into()));
        let response = runtime.invoke(
            &request("fixture.read", "key", "x"),
            false,
            &mut executor,
        );
        assert_eq!(executor.invocations, 0);
        assert!(matches!(response, ToolResponse::Error(_)));
        assert_eq!(runtime.audit()[0].reason, AuthorizationReason::NotAllowlisted);
    }

    #[test]
    fn exact_tool_and_capability_allowlists_are_independent() {
        for policy in [
            PermissionPolicy::default().allow_tool("fixture.read", 1),
            PermissionPolicy::default().allow_capability("fixture.read"),
        ] {
            let mut runtime =
                AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
            let mut executor = DeterministicMockExecutor::new(
                MockMode::EchoArgument("key".into()),
            );
            let response = runtime.invoke(
                &request("fixture.read", "key", "abc"),
                false,
                &mut executor,
            );
            assert_eq!(executor.invocations, 1);
            assert_eq!(response, ToolResponse::result("session:call:fixture.read", "abc"));
            assert_eq!(runtime.audit()[0].outcome, AuthorizationOutcome::Allow);
        }
    }

    #[test]
    fn malformed_envelope_is_rejected_before_lookup_or_audit() {
        let policy = PermissionPolicy::default().allow_capability("fixture.read");
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("never".into()));
        let mut malformed = request("fixture.read", "key", "x");
        malformed.call_id = "bad\ncall".into();

        let response = runtime.invoke(&malformed, false, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert!(runtime.audit().is_empty());
        match response {
            ToolResponse::Error(error) => {
                assert_eq!(error.call_id, "invalid-call");
                assert_eq!(error.code, "protocol-validation");
            }
            other => panic!("expected protocol error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_and_invalid_calls_fail_before_execution_and_are_audited() {
        let policy = PermissionPolicy::default().allow_capability("fixture.read");
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("never".into()));

        let unknown = request("fixture.unknown", "key", "x");
        let _ = runtime.invoke(&unknown, false, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert_eq!(runtime.audit()[0].reason, AuthorizationReason::UnknownTool);

        let invalid = ToolRequest::new(
            "call:bad",
            "fixture.read",
            1,
            vec![ToolArgument {
                name: "wrong".into(),
                value: ToolValue::String("x".into()),
            }],
        );
        let _ = runtime.invoke(&invalid, false, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert_eq!(
            runtime.audit()[1].reason,
            AuthorizationReason::ProtocolValidation
        );
    }

    #[test]
    fn mutative_calls_are_denied_or_require_explicit_confirmation() {
        let base = PermissionPolicy::default().allow_tool("fixture.write", 1);
        let mut denied_runtime =
            AuthorizedToolRuntime::new(vec![write_definition()], base.clone()).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("written".into()));
        let write = request("fixture.write", "value", "x");
        let _ = denied_runtime.invoke(&write, true, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert_eq!(
            denied_runtime.audit()[0].reason,
            AuthorizationReason::MutationDenied
        );

        let mut confirmed_policy = base;
        confirmed_policy.mutation = MutationPolicy::RequireConfirmation;
        let mut runtime =
            AuthorizedToolRuntime::new(vec![write_definition()], confirmed_policy).unwrap();
        let _ = runtime.invoke(&write, false, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert_eq!(
            runtime.audit()[0].reason,
            AuthorizationReason::ConfirmationRequired
        );
        let result = runtime.invoke(&write, true, &mut executor);
        assert_eq!(executor.invocations, 1);
        assert!(matches!(result, ToolResponse::Result(_)));
        assert_eq!(runtime.audit()[1].outcome, AuthorizationOutcome::Allow);
        assert!(runtime.audit()[1].mutation_confirmed);
    }

    #[test]
    fn resource_budgets_block_calls_before_executor() {
        let mut policy = PermissionPolicy::default().allow_tool("fixture.read", 1);
        policy.max_timeout_ms = 100;
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("never".into()));
        let _ = runtime.invoke(
            &request("fixture.read", "key", "x"),
            false,
            &mut executor,
        );
        assert_eq!(executor.invocations, 0);
        assert_eq!(runtime.audit()[0].reason, AuthorizationReason::TimeoutBudget);

        let mut policy = PermissionPolicy::default().allow_tool("fixture.read", 1);
        policy.max_result_bytes = 64;
        runtime.set_policy(policy).unwrap();
        let _ = runtime.invoke(
            &request("fixture.read", "key", "x"),
            false,
            &mut executor,
        );
        assert_eq!(runtime.audit()[1].reason, AuthorizationReason::ResultBudget);

        let mut policy = PermissionPolicy::default().allow_tool("fixture.read", 1);
        policy.max_argument_bytes = 3;
        runtime.set_policy(policy).unwrap();
        let _ = runtime.invoke(
            &request("fixture.read", "key", "abcd"),
            false,
            &mut executor,
        );
        assert_eq!(runtime.audit()[2].reason, AuthorizationReason::ArgumentBudget);
        assert_eq!(executor.invocations, 0);
    }

    #[test]
    fn argument_budget_uses_canonical_encoded_bytes_for_non_string_values() {
        let definition = ToolDefinition::new(
            "fixture.typed",
            1,
            "fixture.typed",
            false,
            500,
            128,
            vec![
                ToolArgSpec {
                    name: "integer".into(),
                    required: true,
                    kind: ToolArgKind::Integer {
                        min: i64::MIN,
                        max: i64::MAX,
                    },
                },
                ToolArgSpec {
                    name: "flag".into(),
                    required: true,
                    kind: ToolArgKind::Boolean,
                },
            ],
        )
        .unwrap();
        let request = ToolRequest::new(
            "call:typed-budget",
            "fixture.typed",
            1,
            vec![
                ToolArgument {
                    name: "integer".into(),
                    value: ToolValue::Integer(i64::MIN),
                },
                ToolArgument {
                    name: "flag".into(),
                    value: ToolValue::Boolean(false),
                },
            ],
        );
        let canonical_bytes = request.encoded_argument_bytes();

        let mut policy = PermissionPolicy::default().allow_tool("fixture.typed", 1);
        policy.max_argument_bytes = canonical_bytes - 1;
        let mut runtime = AuthorizedToolRuntime::new(vec![definition], policy).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("never".into()));

        let _ = runtime.invoke(&request, false, &mut executor);
        assert_eq!(executor.invocations, 0);
        assert_eq!(
            runtime.audit()[0].reason,
            AuthorizationReason::ArgumentBudget
        );
        assert_eq!(runtime.audit()[0].argument_bytes, canonical_bytes);
    }

    #[test]
    fn policy_can_change_at_runtime_without_model_or_executor_changes() {
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], PermissionPolicy::default())
                .unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("ok".into()));
        let req = request("fixture.read", "key", "x");
        let _ = runtime.invoke(&req, false, &mut executor);
        assert_eq!(executor.invocations, 0);

        runtime
            .set_policy(PermissionPolicy::default().allow_tool("fixture.read", 1))
            .unwrap();
        let response = runtime.invoke(&req, false, &mut executor);
        assert_eq!(executor.invocations, 1);
        assert!(matches!(response, ToolResponse::Result(_)));
        assert_ne!(
            runtime.audit()[0].policy_fingerprint,
            runtime.audit()[1].policy_fingerprint
        );
    }

    #[test]
    fn authorization_can_be_explained_by_stable_call_id() {
        let policy = PermissionPolicy::default().allow_tool("fixture.read", 1);
        let mut runtime =
            AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
        let mut executor =
            DeterministicMockExecutor::new(MockMode::FixedResult("ok".into()));
        let req = request("fixture.read", "key", "x");
        let _ = runtime.invoke(&req, false, &mut executor);
        let explanation = runtime.explain_call(&req.call_id).unwrap();
        assert!(explanation.contains("outcome=allow"));
        assert!(explanation.contains("reason=allowed-tool"));
        assert!(explanation.contains(&req.call_id));
        assert!(runtime.explain_call("missing-call").is_none());
    }

    #[test]
    fn audit_is_deterministic_and_never_contains_argument_values() {
        let policy = PermissionPolicy::default().allow_tool("fixture.read", 1);
        let req = request("fixture.read", "key", "SECRET-VALUE");
        let mut a =
            AuthorizedToolRuntime::new(vec![read_definition()], policy.clone()).unwrap();
        let mut b =
            AuthorizedToolRuntime::new(vec![read_definition()], policy).unwrap();
        let mut ea =
            DeterministicMockExecutor::new(MockMode::FixedResult("ok".into()));
        let mut eb =
            DeterministicMockExecutor::new(MockMode::FixedResult("ok".into()));
        let _ = a.invoke(&req, false, &mut ea);
        let _ = b.invoke(&req, false, &mut eb);
        assert_eq!(a.audit()[0].canonical(), b.audit()[0].canonical());
        assert!(!a.audit()[0].canonical().contains("SECRET-VALUE"));
    }
}
