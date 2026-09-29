//! #299 supported v1 profile. Experimental is not a release requirement.

pub const PROFILE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Core,
    Experimental,
    OutOfScope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub name: &'static str,
    pub status: Status,
    pub backend: &'static str,
}

pub fn profile() -> &'static [Capability] {
    &[
        Capability { name: "cpu-train-eval", status: Status::Core, backend: "cpu" },
        Capability { name: "checkpoint-v1", status: Status::Core, backend: "cpu" },
        Capability { name: "gpu-kernels", status: Status::Experimental, backend: "gpu" },
        Capability { name: "multi-gpu", status: Status::OutOfScope, backend: "gpu" },
    ]
}

pub fn announce(name: &str) -> Result<Status, ()> {
    profile()
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.status)
        .ok_or(())
}

pub fn release_requires(name: &str) -> bool {
    announce(name) == Ok(Status::Core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_is_not_a_release_requirement() {
        assert_eq!(announce("cpu-train-eval").unwrap(), Status::Core);
        assert!(release_requires("cpu-train-eval"));
        assert_eq!(announce("gpu-kernels").unwrap(), Status::Experimental);
        assert!(!release_requires("gpu-kernels"));
        assert_eq!(announce("multi-gpu").unwrap(), Status::OutOfScope);
        assert!(announce("sota-chat").is_err());
    }
}
