//! #301 CLI v1 exit-code contract.

pub const CLI_V1_SCHEMA_VERSION: u32 = 1;
pub const EXIT_OK: i32 = 0;
pub const EXIT_INPUT: i32 = 2;
pub const EXIT_RUNTIME: i32 = 3;
pub const EXIT_CORRUPT: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Supported,
    Experimental,
}

pub fn classify(verb: &str) -> Option<Kind> {
    match verb {
        "check" | "train" | "train-fresh" | "eval" | "chat" | "inspect" | "bench" => {
            Some(Kind::Supported)
        }
        "numeric" => Some(Kind::Experimental),
        _ => None,
    }
}

pub fn exit_code(ok: bool, input: bool, corrupt: bool) -> i32 {
    if corrupt {
        EXIT_CORRUPT
    } else if input {
        EXIT_INPUT
    } else if !ok {
        EXIT_RUNTIME
    } else {
        EXIT_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_never_partial_and_numeric_is_experimental() {
        assert_eq!(exit_code(true, false, false), 0);
        assert_eq!(exit_code(false, true, false), EXIT_INPUT);
        assert_eq!(exit_code(false, false, false), EXIT_RUNTIME);
        assert_eq!(exit_code(true, false, true), EXIT_CORRUPT);
        assert_eq!(classify("train").unwrap(), Kind::Supported);
        assert_eq!(classify("numeric").unwrap(), Kind::Experimental);
        assert!(classify("sota").is_none());
    }
}
