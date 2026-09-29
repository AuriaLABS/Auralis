//! #84 RC go/no-go. Never creates a git tag.

pub const RC_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RcError {
    Blocked,
    TagForbidden,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checklist {
    pub formats_frozen: bool,
    pub provenance: bool,
    pub soak: bool,
    pub model_card: bool,
    pub human_approval: bool,
}

impl Checklist {
    pub fn automated() -> Self {
        Self {
            formats_frozen: true,
            provenance: true,
            soak: true,
            model_card: true,
            human_approval: false,
        }
    }

    pub fn ready(&self) -> bool {
        self.formats_frozen && self.provenance && self.soak && self.model_card && self.human_approval
    }
}

pub fn go_no_go(check: &Checklist) -> Result<(), RcError> {
    if check.ready() {
        Ok(())
    } else {
        Err(RcError::Blocked)
    }
}

pub fn tag_v1(check: &Checklist) -> Result<(), RcError> {
    go_no_go(check)?;
    Err(RcError::TagForbidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automated_rc_is_blocked_and_never_tags() {
        let auto = Checklist::automated();
        assert_eq!(go_no_go(&auto), Err(RcError::Blocked));
        assert_eq!(tag_v1(&auto), Err(RcError::Blocked));
        let mut approved = auto;
        approved.human_approval = true;
        assert_eq!(go_no_go(&approved), Ok(()));
        assert_eq!(tag_v1(&approved), Err(RcError::TagForbidden));
    }
}
