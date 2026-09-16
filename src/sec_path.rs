//! Path confinement for untrusted file arguments.
//! Lexical only: no symlink resolution, no filesystem IO.

use std::path::{Component, Path, PathBuf};

pub const MAX_PATH_COMPONENTS: usize = 32;

/// Join `candidate` under `root` and reject escapes.
/// Absolute candidates and `..` that leave `root` fail closed.
pub fn confine(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    if candidate.is_absolute() {
        return Err(format!("absolute path rejected: {}", candidate.display()));
    }
    let mut components = 0usize;
    let mut depth = 0i32;
    for comp in candidate.components() {
        match comp {
            Component::CurDir => {}
            Component::Normal(_) => {
                components += 1;
                depth += 1;
            }
            Component::ParentDir => {
                components += 1;
                depth -= 1;
                if depth < 0 {
                    return Err(format!("path escapes root: {}", candidate.display()));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("absolute path rejected: {}", candidate.display()));
            }
        }
        if components > MAX_PATH_COMPONENTS {
            return Err(format!(
                "path too deep (max {MAX_PATH_COMPONENTS}): {}",
                candidate.display()
            ));
        }
    }
    Ok(root.join(candidate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn accepts_relative_child() {
        let p = confine(Path::new("/data"), Path::new("ckpt/auralis.bin")).unwrap();
        assert_eq!(p, PathBuf::from("/data/ckpt/auralis.bin"));
    }

    #[test]
    fn rejects_parent_escape() {
        assert!(confine(Path::new("/data"), Path::new("../etc/passwd")).is_err());
        assert!(confine(Path::new("/data"), Path::new("ok/../../etc")).is_err());
    }

    #[test]
    fn rejects_absolute() {
        assert!(confine(Path::new("/data"), Path::new("/etc/passwd")).is_err());
    }

    #[test]
    fn rejects_too_many_components() {
        let deep = (0..=MAX_PATH_COMPONENTS).map(|_| "a").collect::<Vec<_>>().join("/");
        assert!(confine(Path::new("/data"), Path::new(&deep)).is_err());
    }
}
