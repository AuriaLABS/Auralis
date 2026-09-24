//! #113 tool provider boundary. Planner/model do not import providers.
//!
//! A provider can list, describe and invoke tools. It does **not** grant
//! #45 permissions. Authorization stays in AuthorizedToolRuntime.

use crate::tool_protocol::{
    ToolArgKind, ToolArgSpec, ToolDefinition, ToolProtocolError, TOOL_PROTOCOL_SCHEMA_VERSION,
};
use crate::tool_registry::ToolListing;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const TOOL_PROVIDER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderError {
    UnknownTool(String),
    Incompatible(u32),
    Timeout,
    Cancelled,
    Protocol(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTool(name) => write!(f, "unknown tool {name}"),
            Self::Incompatible(v) => write!(f, "incompatible provider schema {v}"),
            Self::Timeout => write!(f, "provider timeout"),
            Self::Cancelled => write!(f, "provider cancelled"),
            Self::Protocol(msg) => write!(f, "protocol: {msg}"),
        }
    }
}

impl Error for ProviderError {}

impl From<ToolProtocolError> for ProviderError {
    fn from(err: ToolProtocolError) -> Self {
        Self::Protocol(err.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderCall {
    pub call_id: String,
    pub tool_name: String,
    pub tool_version: u32,
    pub payload: String,
    pub timeout_ms: u64,
    pub cancelled: bool,
}

pub trait ToolProvider {
    fn id(&self) -> &str;
    fn schema_version(&self) -> u32;
    fn list(&self) -> Vec<ToolListing>;
    fn describe(&self, name: &str, version: u32) -> Option<ToolDefinition>;
    fn negotiate(&self, requested: u32) -> Result<u32, ProviderError>;
    fn invoke(&mut self, call: &ProviderCall) -> Result<String, ProviderError>;
}

#[derive(Clone, Debug)]
pub struct MockProvider {
    id: String,
    tools: BTreeMap<(String, u32), ToolDefinition>,
    invocations: u64,
}

impl MockProvider {
    pub fn echo(id: impl Into<String>) -> Result<Self, ProviderError> {
        let definition = ToolDefinition::new(
            "provider.echo",
            1,
            "echo",
            false,
            50,
            256,
            vec![ToolArgSpec {
                name: "text".into(),
                kind: ToolArgKind::String {
                    min_len: 0,
                    max_len: 64,
                },
                required: true,
            }],
        )?;
        let mut tools = BTreeMap::new();
        tools.insert((definition.name.clone(), definition.version), definition);
        Ok(Self {
            id: id.into(),
            tools,
            invocations: 0,
        })
    }

    pub fn invocations(&self) -> u64 {
        self.invocations
    }
}

impl ToolProvider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn schema_version(&self) -> u32 {
        TOOL_PROVIDER_SCHEMA_VERSION
    }

    fn list(&self) -> Vec<ToolListing> {
        self.tools
            .values()
            .map(|definition| ToolListing {
                name: definition.name.clone(),
                version: definition.version,
                capability: definition.capability.clone(),
                mutative: definition.mutative,
            })
            .collect()
    }

    fn describe(&self, name: &str, version: u32) -> Option<ToolDefinition> {
        self.tools.get(&(name.to_string(), version)).cloned()
    }

    fn negotiate(&self, requested: u32) -> Result<u32, ProviderError> {
        if requested != TOOL_PROVIDER_SCHEMA_VERSION {
            return Err(ProviderError::Incompatible(requested));
        }
        Ok(TOOL_PROVIDER_SCHEMA_VERSION)
    }

    fn invoke(&mut self, call: &ProviderCall) -> Result<String, ProviderError> {
        if call.cancelled {
            return Err(ProviderError::Cancelled);
        }
        let Some(definition) = self.describe(&call.tool_name, call.tool_version) else {
            return Err(ProviderError::UnknownTool(call.tool_name.clone()));
        };
        if call.timeout_ms == 0 || call.timeout_ms > definition.timeout_ms {
            return Err(ProviderError::Timeout);
        }
        self.invocations += 1;
        Ok(format!("{}:{}", self.id, call.payload))
    }
}

pub fn run_provider_contract(provider: &mut impl ToolProvider) -> Result<(), ProviderError> {
    provider.negotiate(TOOL_PROVIDER_SCHEMA_VERSION)?;
    if let Err(err) = provider.negotiate(TOOL_PROVIDER_SCHEMA_VERSION + 1) {
        match err {
            ProviderError::Incompatible(_) => {}
            other => return Err(other),
        }
    } else {
        return Err(ProviderError::Protocol(
            "future schema must fail negotiation".into(),
        ));
    }
    let listed = provider.list();
    if listed.is_empty() {
        return Err(ProviderError::Protocol("provider listed no tools".into()));
    }
    for listing in &listed {
        let Some(definition) = provider.describe(&listing.name, listing.version) else {
            return Err(ProviderError::UnknownTool(listing.name.clone()));
        };
        if definition.name != listing.name || definition.version != listing.version {
            return Err(ProviderError::Protocol("describe drifted from list".into()));
        }
        if definition.schema_version != TOOL_PROTOCOL_SCHEMA_VERSION {
            return Err(ProviderError::Protocol("tool schema drifted".into()));
        }
    }
    let first = &listed[0];
    let accepted = provider.invoke(&ProviderCall {
        call_id: "contract-1".into(),
        tool_name: first.name.clone(),
        tool_version: first.version,
        payload: "ok".into(),
        timeout_ms: 1,
        cancelled: false,
    })?;
    if accepted.is_empty() {
        return Err(ProviderError::Protocol("empty invoke result".into()));
    }
    let cancelled = provider.invoke(&ProviderCall {
        call_id: "contract-cancel".into(),
        tool_name: first.name.clone(),
        tool_version: first.version,
        payload: "late".into(),
        timeout_ms: 1,
        cancelled: true,
    });
    if cancelled != Err(ProviderError::Cancelled) {
        return Err(ProviderError::Protocol("cancel was not propagated".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_permissions::{AuthorizedToolRuntime, PermissionPolicy};
    use crate::tool_protocol::{ToolArgument, ToolRequest, ToolValue};
    use crate::tool_registry::{ReferenceToolExecutor, ToolRegistry};

    #[test]
    fn mock_provider_passes_reusable_contract() {
        let mut provider = MockProvider::echo("mock-a").unwrap();
        run_provider_contract(&mut provider).unwrap();
        assert!(provider.invocations() >= 1);
    }

    #[test]
    fn providers_are_isolated_and_do_not_grant_permissions() {
        let mut a = MockProvider::echo("a").unwrap();
        let mut b = MockProvider::echo("b").unwrap();
        a.invoke(&ProviderCall {
            call_id: "1".into(),
            tool_name: "provider.echo".into(),
            tool_version: 1,
            payload: "x".into(),
            timeout_ms: 1,
            cancelled: false,
        })
        .unwrap();
        assert_eq!(a.invocations(), 1);
        assert_eq!(b.invocations(), 0);

        let registry = ToolRegistry::with_reference_tools().unwrap();
        let policy = PermissionPolicy::default();
        let mut runtime = AuthorizedToolRuntime::new(registry.catalog(), policy).unwrap();
        let mut executor = registry.reference_executor();
        let request = ToolRequest {
            schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
            call_id: "no-grant".into(),
            tool_name: "ref.echo".into(),
            tool_version: 1,
            arguments: vec![ToolArgument {
                name: "text".into(),
                value: ToolValue::String("hi".into()),
            }],
        };
        let response = runtime.invoke(&request, false, &mut executor);
        assert!(matches!(response, crate::tool_protocol::ToolResponse::Error(_)));
    }
}
