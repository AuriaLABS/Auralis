use auralis::agent_permissions::{AuthorizedToolRuntime, PermissionPolicy};
use auralis::planner::{
    run_direct, run_plan, PlanBudget, Planner, PlannerTask, ReferencePlanner,
};
use auralis::tool_registry::ToolRegistry;

fn runtime(
    registry: &ToolRegistry,
) -> (
    AuthorizedToolRuntime,
    auralis::tool_registry::ReferenceToolExecutor,
) {
    let policy = PermissionPolicy::default()
        .allow_tool("lookup", 1)
        .allow_tool("ref.error", 1)
        .allow_tool("ref.echo", 1);
    (
        AuthorizedToolRuntime::new(registry.catalog(), policy).unwrap(),
        registry.reference_executor(),
    )
}

fn main() {
    let registry = ToolRegistry::with_reference_tools().expect("reference registry");
    let planner = ReferencePlanner;
    let cases = vec![
        (
            "lookup",
            PlannerTask::Lookup { key: "city-2".into() },
            "1002",
        ),
        (
            "recover",
            PlannerTask::RecoverThenEcho { text: "recovered".into() },
            "recovered",
        ),
    ];

    let mut direct_success = 0usize;
    let mut planner_success = 0usize;
    let mut direct_steps = 0usize;
    let mut planner_steps = 0usize;
    let mut direct_tools = 0usize;
    let mut planner_tools = 0usize;
    let mut direct_recovered = 0usize;
    let mut planner_recovered = 0usize;
    let mut direct_cost = 0usize;
    let mut planner_cost = 0usize;

    println!(
        "planner_protocol | schema=1 cases={} budget_steps=12 budget_tools=8 budget_timeout_ms=5000 registry_fingerprint={:016x}",
        cases.len(),
        registry.fingerprint()
    );

    for (id, task, expected) in cases {
        let (mut direct_runtime, mut direct_executor) = runtime(&registry);
        let direct = run_direct(&task, &registry, &mut direct_runtime, &mut direct_executor);

        let plan = planner.plan(&task, PlanBudget::default()).expect("deterministic plan");
        let initial_fingerprint = plan.fingerprint().expect("plan fingerprint");
        let canonical = plan.canonical().expect("plan canonical");
        let (mut planner_runtime, mut planner_executor) = runtime(&registry);
        let planned = run_plan(
            &task,
            &planner,
            plan,
            &registry,
            &mut planner_runtime,
            &mut planner_executor,
        );

        let direct_exact = direct.answer.as_deref() == Some(expected);
        let planner_exact = planned.answer.as_deref() == Some(expected);
        direct_success += usize::from(direct_exact);
        planner_success += usize::from(planner_exact);
        direct_steps += direct.steps_executed;
        planner_steps += planned.steps_executed;
        direct_tools += direct.tool_calls;
        planner_tools += planned.tool_calls;
        direct_recovered += direct.recoverable_errors;
        planner_recovered += planned.recoverable_errors;
        direct_cost += direct.cost_units();
        planner_cost += planned.cost_units();

        println!(
            "planner_case | id={} strategy=direct success={} steps={} tool_calls={} recoverable_errors={} replans={} cost_units={} answer={}",
            id, direct_exact, direct.steps_executed, direct.tool_calls,
            direct.recoverable_errors, direct.replans, direct.cost_units(),
            direct.answer.as_deref().unwrap_or("null")
        );
        println!(
            "planner_case | id={} strategy=planner success={} steps={} tool_calls={} recoverable_errors={} replans={} cost_units={} initial_plan={:016x} final_plan={:016x} answer={}",
            id, planner_exact, planned.steps_executed, planned.tool_calls,
            planned.recoverable_errors, planned.replans, planned.cost_units(),
            initial_fingerprint, planned.plan_fingerprint,
            planned.answer.as_deref().unwrap_or("null")
        );
        println!(
            "planner_plan | id={} canonical_bytes={} audit_events={} executor_invocations={}",
            id, canonical.len(), planner_runtime.audit().len(), planner_executor.invocations()
        );
    }

    println!(
        "planner_summary | direct_success={}/2 planner_success={}/2 direct_steps={} planner_steps={} direct_tool_calls={} planner_tool_calls={} direct_recoverable_errors={} planner_recoverable_errors={} direct_cost_units={} planner_cost_units={} no_auto_promotion=true",
        direct_success, planner_success, direct_steps, planner_steps, direct_tools, planner_tools,
        direct_recovered, planner_recovered, direct_cost, planner_cost
    );
}
