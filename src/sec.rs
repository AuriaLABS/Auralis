//! Security inventory for unsafe Rust and untrusted-input limits.
//!
//! Foundation currently admits zero `unsafe` blocks. New `unsafe` must update
//! [`EXPECTED_UNSAFE_BLOCKS`] and document invariants in `docs/sec-unsafe.md`.

use std::fs;
use std::path::Path;

/// Contract: Auralis engine/tooling code is safe Rust only.
pub const EXPECTED_UNSAFE_BLOCKS: usize = 0;

/// Checkpoint string payload cap (bytes), mirrored from checkpoint loader.
pub const MAX_CHECKPOINT_STRING: u32 = 1_048_576;
/// BPE table cap mirrored from checkpoint loader.
pub const MAX_BPE_TABLE: u32 = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsafeHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecReport {
    pub files_scanned: usize,
    pub hits: Vec<UnsafeHit>,
}

impl SecReport {
    pub fn human(&self) -> String {
        let mut out = format!(
            "sec-audit | files={} unsafe_hits={} expected={}\n",
            self.files_scanned,
            self.hits.len(),
            EXPECTED_UNSAFE_BLOCKS
        );
        for hit in &self.hits {
            out.push_str(&format!(
                "unsafe | {}:{} {}\n",
                hit.path, hit.line, hit.text
            ));
        }
        out.push_str(&format!(
            "limits | MAX_CHECKPOINT_STRING={} MAX_BPE_TABLE={}\n",
            MAX_CHECKPOINT_STRING, MAX_BPE_TABLE
        ));
        out
    }

    pub fn ok(&self) -> bool {
        self.hits.len() == EXPECTED_UNSAFE_BLOCKS
    }
}

fn is_code_unsafe_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.starts_with("//") || trimmed.starts_with('*') || trimmed.starts_with("///") {
        return false;
    }
    if let Some(idx) = trimmed.find("unsafe") {
        let before = trimmed[..idx].chars().last();
        let after = trimmed[idx + 6..].chars().next();
        let start_ok = before.map(|c| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(true);
        let end_ok = after.map(|c| !c.is_ascii_alphanumeric() && c != '_').unwrap_or(true);
        return start_ok && end_ok;
    }
    false
}

pub fn scan_source(text: &str, path: &str) -> Vec<UnsafeHit> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| is_code_unsafe_line(line))
        .map(|(i, line)| UnsafeHit {
            path: path.to_string(),
            line: i + 1,
            text: line.trim().to_string(),
        })
        .collect()
}

pub fn scan_tree(root: impl AsRef<Path>) -> Result<SecReport, String> {
    let root = root.as_ref();
    let mut files_scanned = 0;
    let mut hits = Vec::new();
    walk_rs(root, root, &mut files_scanned, &mut hits)?;
    Ok(SecReport {
        files_scanned,
        hits,
    })
}

fn walk_rs(
    root: &Path,
    dir: &Path,
    files: &mut usize,
    hits: &mut Vec<UnsafeHit>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read dirent: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            walk_rs(root, &path, files, hits)?;
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        *files += 1;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        hits.extend(scan_source(&text, &rel));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_and_identifiers_are_not_hits() {
        let src = "// unsafe comment\nfn unsafe_name() {}\nlet x = 1;\n";
        assert!(scan_source(src, "t.rs").is_empty());
    }

    #[test]
    fn real_unsafe_block_is_a_hit() {
        let src = "fn f() { unsafe { *p } }\n";
        let hits = scan_source(src, "t.rs");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 1);
    }

    #[test]
    fn crate_src_matches_expected_unsafe_contract() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        if !root.is_dir() {
            return;
        }
        let report = scan_tree(&root).unwrap();
        assert!(
            report.ok(),
            "unexpected unsafe Rust:\n{}",
            report.human()
        );
    }
}
