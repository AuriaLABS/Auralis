//! Read-only inspection of checkpoints, manifests and effective run config.
//!
//! Does not train. Fail-closed on corrupt or incompatible artifacts.

use crate::checkpoint;
use crate::manifest::{self, ExperimentManifest, MANIFEST_VERSION};
use crate::run_config::{RunConfig, RUN_CONFIG_SCHEMA_VERSION};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct InspectReport {
    pub path: String,
    pub bytes: u64,
    pub tokenizer_kind: String,
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block: usize,
    pub n_ff: usize,
    pub parameter_count: usize,
    pub parameter_fingerprint: u64,
    pub has_adam: bool,
    pub adam_lr: Option<f32>,
    pub adam_t: Option<i32>,
    pub adam_state_len: Option<usize>,
    pub manifest_present: bool,
    pub manifest_ok: bool,
    pub manifest_error: Option<String>,
    pub manifest_version: Option<u32>,
    pub run_config_schema: Option<u32>,
    pub run_config_fingerprint: Option<u64>,
    pub global_step: Option<u64>,
    pub weights_match_manifest: Option<bool>,
}

impl InspectReport {
    pub fn human(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "inspect | path={} bytes={}\n",
            self.path, self.bytes
        ));
        out.push_str(&format!(
            "model | tok={} vocab={} embd={} head={} layer={} block={} ff={} params={} fingerprint={:016x}\n",
            self.tokenizer_kind,
            self.vocab,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block,
            self.n_ff,
            self.parameter_count,
            self.parameter_fingerprint
        ));
        match (self.has_adam, self.adam_lr, self.adam_t, self.adam_state_len) {
            (true, Some(lr), Some(t), Some(n)) => {
                out.push_str(&format!("optimizer | present=true lr={lr} t={t} state_len={n}\n"));
            }
            _ => out.push_str("optimizer | present=false\n"),
        }
        if !self.manifest_present {
            out.push_str("manifest | present=false\n");
        } else if let Some(err) = &self.manifest_error {
            out.push_str(&format!("manifest | present=true ok=false error={err}\n"));
        } else {
            out.push_str(&format!(
                "manifest | present=true ok=true version={} run_config_schema={} run_config_fingerprint={:016x} global_step={} weights_match={}\n",
                self.manifest_version.unwrap_or(0),
                self.run_config_schema.unwrap_or(0),
                self.run_config_fingerprint.unwrap_or(0),
                self.global_step.unwrap_or(0),
                self.weights_match_manifest.unwrap_or(false)
            ));
        }
        out.push_str(&format!(
            "compat | manifest_schema_expected={} run_config_schema_expected={}\n",
            MANIFEST_VERSION, RUN_CONFIG_SCHEMA_VERSION
        ));
        out
    }

    pub fn json(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        format!(
            concat!(
                "{{\"path\":\"{}\",\"bytes\":{},\"tokenizer_kind\":\"{}\",",
                "\"vocab\":{},\"n_embd\":{},\"n_head\":{},\"n_layer\":{},",
                "\"block\":{},\"n_ff\":{},\"parameter_count\":{},",
                "\"parameter_fingerprint\":\"{:016x}\",\"has_adam\":{},",
                "\"adam_t\":{},\"manifest_present\":{},\"manifest_ok\":{},",
                "\"manifest_version\":{},\"run_config_fingerprint\":\"{}\",",
                "\"global_step\":{},\"weights_match_manifest\":{}}}"
            ),
            esc(&self.path),
            self.bytes,
            esc(&self.tokenizer_kind),
            self.vocab,
            self.n_embd,
            self.n_head,
            self.n_layer,
            self.block,
            self.n_ff,
            self.parameter_count,
            self.parameter_fingerprint,
            self.has_adam,
            opt_i32(self.adam_t),
            self.manifest_present,
            self.manifest_ok,
            opt_u32(self.manifest_version),
            self.run_config_fingerprint
                .map(|v| format!("{v:016x}"))
                .unwrap_or_else(|| "null".into()),
            opt_u64(self.global_step),
            opt_bool(self.weights_match_manifest),
        )
    }
}

fn opt_i32(v: Option<i32>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
}
fn opt_u32(v: Option<u32>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
}
fn opt_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
}
fn opt_bool(v: Option<bool>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
}

pub fn inspect_checkpoint(path: impl AsRef<Path>) -> Result<InspectReport, String> {
    let path = path.as_ref();
    let bytes = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    let (gpt, tok, adam) = checkpoint::load_full(path).map_err(|e| {
        format!("corrupt or incompatible checkpoint {}: {e}", path.display())
    })?;
    let params = gpt.collect_params();
    let parameter_fingerprint = manifest::fingerprint_params(&params);

    let mut report = InspectReport {
        path: path.display().to_string(),
        bytes,
        tokenizer_kind: tok.kind().to_string(),
        vocab: gpt.cfg.vocab,
        n_embd: gpt.cfg.n_embd,
        n_head: gpt.cfg.n_head,
        n_layer: gpt.cfg.n_layer,
        block: gpt.cfg.block,
        n_ff: gpt.cfg.n_ff,
        parameter_count: params.len(),
        parameter_fingerprint,
        has_adam: adam.is_some(),
        adam_lr: adam.as_ref().map(|a| a.export().0),
        adam_t: adam.as_ref().map(|a| a.export().1),
        adam_state_len: adam.as_ref().map(|a| a.export().2.len()),
        manifest_present: manifest::manifest_path(path).exists(),
        manifest_ok: false,
        manifest_error: None,
        manifest_version: None,
        run_config_schema: None,
        run_config_fingerprint: None,
        global_step: None,
        weights_match_manifest: None,
    };

    if report.manifest_present {
        match manifest::load_manifest(path) {
            Ok(m) => apply_manifest(&mut report, &m, parameter_fingerprint),
            Err(e) => {
                report.manifest_ok = false;
                report.manifest_error = Some(e);
            }
        }
    }
    Ok(report)
}

fn apply_manifest(report: &mut InspectReport, m: &ExperimentManifest, fp: u64) {
    report.manifest_ok = true;
    report.manifest_version = Some(m.version);
    report.run_config_schema = Some(m.run_config_schema);
    report.run_config_fingerprint = Some(m.run_config_fingerprint);
    report.global_step = Some(m.global_step);
    report.weights_match_manifest = Some(m.parameter_fingerprint == fp && m.parameter_count == report.parameter_count);
}

pub fn inspect_run_config(path: Option<&Path>) -> Result<RunConfig, String> {
    let cfg = match path {
        Some(p) => RunConfig::load(p)?,
        None => RunConfig::default(),
    };
    cfg.validate()
        .map_err(|e| format!("configuración inválida: {e}"))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BpeTokenizer;
    use crate::model::{Config, Gpt};
    use crate::optim::Adam;
    use crate::run_config::RunConfig;
    use crate::tokenizer::AnyTok;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique(tag: &str) -> std::path::PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("auralis-inspect-{tag}-{}-{n}.bin", std::process::id()))
    }

    fn fixture() -> (Gpt, AnyTok, Adam, RunConfig) {
        let tok = AnyTok::Bpe(BpeTokenizer::fit("auralis inspect fixture", 4));
        let cfg = Config {
            vocab: tok.vocab_size(),
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block: 4,
            n_ff: 16,
        };
        let mut rng = StdRng::seed_from_u64(11);
        let gpt = Gpt::new(cfg, &mut rng);
        let adam = Adam::new(gpt.collect_params().len(), 3e-3);
        (gpt, tok, adam, RunConfig::default())
    }

    #[test]
    fn inspect_reports_checkpoint_and_matching_manifest() {
        let (gpt, tok, adam, run) = fixture();
        let path = unique("ok");
        checkpoint::save_full(&path, &gpt, &tok, Some(&adam)).unwrap();
        let manifest = ExperimentManifest::capture(&gpt, &tok, &run, 0xabc, 7);
        manifest::save_manifest(&path, &manifest).unwrap();

        let report = inspect_checkpoint(&path).unwrap();
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(manifest::manifest_path(&path));

        assert!(report.has_adam);
        assert_eq!(report.adam_t, Some(adam.t));
        assert!(report.manifest_ok);
        assert_eq!(report.weights_match_manifest, Some(true));
        assert!(report.human().contains("inspect |"));
        assert!(report.json().contains("\"has_adam\":true"));
    }

    #[test]
    fn inspect_rejects_corrupt_checkpoint() {
        let path = unique("bad");
        fs::write(&path, b"not-a-checkpoint").unwrap();
        let err = inspect_checkpoint(&path).unwrap_err();
        let _ = fs::remove_file(&path);
        assert!(err.contains("corrupt or incompatible"));
    }

    #[test]
    fn inspect_flags_unreadable_manifest() {
        let (gpt, tok, adam, _) = fixture();
        let path = unique("badman");
        checkpoint::save_full(&path, &gpt, &tok, Some(&adam)).unwrap();
        fs::write(manifest::manifest_path(&path), "auralis_manifest=3\n").unwrap();
        let report = inspect_checkpoint(&path).unwrap();
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(manifest::manifest_path(&path));
        assert!(report.manifest_present);
        assert!(!report.manifest_ok);
        assert!(report
            .manifest_error
            .unwrap()
            .contains("explicit migration required"));
    }
}
