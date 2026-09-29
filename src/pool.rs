//! #85 dynamic agent pool. Roles belong to the work.

pub const POOL_SCHEMA_VERSION: u32 = 1;
pub const MAX_WIP: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Architect,
    Implementer,
    Verifier,
    Reviewer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Agent {
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub id: String,
    pub blocked: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub agent: String,
    pub task: String,
    pub role: Role,
    pub independent_review: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolError {
    Blocked,
    WipFull,
    Empty,
}

pub fn assign(
    agents: &[Agent],
    open: &[Task],
    active: usize,
    solo: bool,
) -> Result<Assignment, PoolError> {
    if agents.is_empty() {
        return Err(PoolError::Empty);
    }
    if active >= MAX_WIP {
        return Err(PoolError::WipFull);
    }
    let task = open
        .iter()
        .find(|t| !t.blocked)
        .ok_or(PoolError::Blocked)?;
    Ok(Assignment {
        agent: agents[0].id.clone(),
        task: task.id.clone(),
        role: Role::Implementer,
        independent_review: !solo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fungible_pick_and_solo_is_not_independent() {
        let agents = [Agent { id: "grok".into() }, Agent { id: "chatgpt".into() }];
        let tasks = [
            Task { id: "blocked".into(), blocked: true },
            Task { id: "next".into(), blocked: false },
        ];
        let a = assign(&agents, &tasks, 0, true).unwrap();
        assert_eq!(a.task, "next");
        assert!(!a.independent_review);
        assert_eq!(assign(&agents, &tasks, 2, false).unwrap_err(), PoolError::WipFull);
        assert_eq!(
            assign(&agents, &[Task { id: "x".into(), blocked: true }], 0, false).unwrap_err(),
            PoolError::Blocked
        );
    }
}
