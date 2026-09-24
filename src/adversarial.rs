//! #144 adversarial fixtures for tools, permissions and malformed results.
//!
//! Every case must deny or fail closed. A deny never increments the
//! reference executor. Audit text explains the reason.

use crate::agent_permissions::{
    AuthorizationOutcome, AuthorizationReason, AuthorizedToolRuntime, MutationPolicy,
    PermissionPolicy,
};
use crate::event_stream::{EventStream, PushResult, StreamEventKind};
use crate::tool_protocol::{
    ToolArgKind, ToolArgSpec, ToolArgument, ToolDefinition, ToolErrorSeverity, ToolRequest,
    ToolResponse, ToolValue, TOOL_PROTOCOL_SCHEMA_VERSION,
};
use crate::tool_registry::{ReferenceToolExecutor, ToolRegistry};

pub const ADVERSARIAL_SUITE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdversarialClass {
    UnknownTool,
    Lookalike,
    MalformedArgs,
    MalformedResult,
    Oversized,
    Escalation,
    ReplayMutation,
    StaleCorrelation,
    ConfusedDeputy,
    CancelMutation,
}

impl AdversarialClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownTool => "unknown-tool",
            Self::Lookalike => "lookalike",
            Self::MalformedArgs => "malformed-args",
            Self::MalformedResult => "malformed-result",
            Self::Oversized => "oversized",
            Self::Escalation => "escalation",
            Self::ReplayMutation => "replay-mutation",
            Self::StaleCorrelation => "stale-correlation",
            Self::ConfusedDeputy => "confused-deputy",
            Self::CancelMutation => "cancel-mutation",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdversarialCase {
    pub class: AdversarialClass,
    pub blocked: bool,
    pub reason: String,
    pub executed: bool,
}

fn request(call_id: &str, tool: &str, args: Vec<ToolArgument>) -> ToolRequest {
    ToolRequest {
        schema_version: TOOL_PROTOCOL_SCHEMA_VERSION,
        call_id: call_id.into(),
        tool_name: tool.into(),
        tool_version: 1,
        arguments: args,
    }
}

fn text_arg(value: &str) -> ToolArgument {
    ToolArgument {
        name: "text".into(),
        value: ToolValue::String(value.into()),
    }
}

fn write_tool() -> ToolDefinition {
    ToolDefinition::new(
        "ref.write",
        1,
        "fs.write",
        true,
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
    .unwrap()
}

fn lookalike_tool() -> ToolDefinition {
    ToolDefinition::new(
        "ref.echoo",
        1,
        "fixture.read",
        false,
        100,
        64,
        vec![ToolArgSpec {
            name: "text".into(),
            required: true,
            kind: ToolArgKind::String {
                min_len: 0,
                max_len: 32,
            },
        }],
    )
    .unwrap()
}

fn harness() -> (AuthorizedToolRuntime, ReferenceToolExecutor) {
    let registry = ToolRegistry::with_reference_tools().unwrap();
    let mut catalog = registry.catalog();
    catalog.push(write_tool());
    catalog.push(lookalike_tool());
    let mut policy = PermissionPolicy::default()
        .allow_tool("ref.echo", 1)
        .allow_tool("ref.write", 1);
    policy.mutation = MutationPolicy::RequireConfirmation;
    policy.max_argument_bytes = 64;
    let runtime = AuthorizedToolRuntime::new(catalog, policy).unwrap();
    (runtime, registry.reference_executor())
}

fn invoke_case(
    runtime: &mut AuthorizedToolRuntime,
    executor: &mut ReferenceToolExecutor,
    class: AdversarialClass,
    request: &ToolRequest,
    confirmed: bool,
) -> AdversarialCase {
    let before = executor.invocations();
    let response = runtime.invoke(request, confirmed, executor);
    let executed = executor.invocations() > before;
    let decision = runtime.decision_for_call(&request.call_id);
    let blocked = decision
        .map(|d| d.outcome == AuthorizationOutcome::Deny)
        .unwrap_or(matches!(response, ToolResponse::Error(_)));
    let reason = runtime
        .explain_call(&request.call_id)
        .unwrap_or_else(|| format!("{response:?}"));
    AdversarialCase {
        class,
        blocked,
        reason,
        executed,
    }
}

pub fn run_suite() -> Vec<AdversarialCase> {
    let mut cases = Vec::new();
    let (mut runtime, mut executor) = harness();

    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::UnknownTool,
        &request("adv-unknown", "ref.missing", vec![text_arg("x")]),
        false,
    ));

    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::Lookalike,
        &request("adv-lookalike", "ref.echoo", vec![text_arg("x")]),
        false,
    ));

    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::MalformedArgs,
        &request(
            "adv-malformed",
            "ref.echo",
            vec![ToolArgument {
                name: "text".into(),
                value: ToolValue::Integer(1),
            }],
        ),
        false,
    ));

    let oversized = "x".repeat(512);
    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::Oversized,
        &request("adv-oversize", "ref.echo", vec![text_arg(&oversized)]),
        false,
    ));

    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::Escalation,
        &request("adv-escalation", "ref.write", vec![text_arg("drop")]),
        false,
    ));

    let write = request("adv-replay", "ref.write", vec![text_arg("once")]);
    let first = invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::ReplayMutation,
        &write,
        true,
    );
    let replay = invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::ReplayMutation,
        &write,
        false,
    );
    cases.push(AdversarialCase {
        class: AdversarialClass::ReplayMutation,
        blocked: replay.blocked,
        reason: format!("first_executed={} replay={}", first.executed, replay.reason),
        executed: replay.executed,
    });

    cases.push(invoke_case(
        &mut runtime,
        &mut executor,
        AdversarialClass::ConfusedDeputy,
        &request("adv-deputy", "ref.add", vec![
            ToolArgument {
                name: "a".into(),
                value: ToolValue::Integer(1),
            },
            ToolArgument {
                name: "b".into(),
                value: ToolValue::Integer(2),
            },
        ]),
        false,
    ));

    let catalog_echo = ToolRegistry::with_reference_tools()
        .unwrap()
        .resolve("ref.echo", 1)
        .cloned()
        .unwrap();
    let stale = ToolResponse::result("other-call", "leaked");
    let stale_err = stale.validate_for(&catalog_echo, "adv-stale");
    cases.push(AdversarialCase {
        class: AdversarialClass::StaleCorrelation,
        blocked: stale_err.is_err(),
        reason: stale_err.err().map(|e| e.to_string()).unwrap_or_default(),
        executed: false,
    });

    let malformed = ToolResponse::error(
        "adv-malformed-result",
        ToolErrorSeverity::Fatal,
        "boom",
        "token=abc123",
    );
    let malformed_err = malformed.validate_for(&catalog_echo, "expected-id");
    cases.push(AdversarialCase {
        class: AdversarialClass::MalformedResult,
        blocked: malformed_err.is_err(),
        reason: malformed_err
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "accepted".into()),
        executed: false,
    });

    let mut stream = EventStream::new("adv-cancel", 8);
    stream.cancel("client-disconnect");
    let cancel_push = stream.push(StreamEventKind::Tool, "ref.write", true);
    cases.push(AdversarialCase {
        class: AdversarialClass::CancelMutation,
        blocked: cancel_push == PushResult::Cancelled,
        reason: format!("{cancel_push:?}"),
        executed: false,
    });

    cases
}

pub fn suite_holds() -> Result<(), String> {
    let cases = run_suite();
    if cases.len() < 10 {
        return Err("adversarial suite missing cases".into());
    }
    for case in &cases {
        if !case.blocked {
            return Err(format!("{} was not blocked", case.class.as_str()));
        }
        if case.executed {
            return Err(format!("{} executed after deny", case.class.as_str()));
        }
        if case.reason.trim().is_empty() {
            return Err(format!("{} left no trace", case.class.as_str()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversarial_cases_fail_closed_with_traces() {
        suite_holds().unwrap();
        let cases = run_suite();
        assert!(cases
            .iter()
            .any(|c| c.class == AdversarialClass::UnknownTool
                && c.reason.contains(AuthorizationReason::UnknownTool.as_str())
                    || c.reason.contains("unknown")));
        assert!(cases.iter().any(|c| c.class == AdversarialClass::Lookalike));
        assert!(cases
            .iter()
            .any(|c| c.class == AdversarialClass::Escalation));
    }

    #[test]
    fn deny_does_not_run_the_executor() {
        let (mut runtime, mut executor) = harness();
        let before = executor.invocations();
        runtime.invoke(
            &request("no-exec", "ref.missing", vec![text_arg("x")]),
            false,
            &mut executor,
        );
        assert_eq!(executor.invocations(), before);
        assert_eq!(
            runtime.audit()[0].outcome,
            AuthorizationOutcome::Deny
        );
    }
}
