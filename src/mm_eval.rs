//! #65 unimodal and cross-modal eval suite.
//!
//! Scores stay separate. A single aggregate is descriptive only and never
//! a pass/fail gate.

pub const MM_EVAL_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Text,
    Vision,
    Audio,
    CrossModal,
}

impl Track {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::CrossModal => "cross-modal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackScore {
    pub track: Track,
    pub hits: u32,
    pub total: u32,
}

impl TrackScore {
    pub fn pass(&self) -> bool {
        self.hits == self.total
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalCost {
    pub tokens: u32,
    pub patches: u32,
    pub frames: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmEvalReport {
    pub schema_version: u32,
    pub commit: String,
    pub config: String,
    pub scores: Vec<TrackScore>,
    pub cost: EvalCost,
}

impl MmEvalReport {
    pub fn score(&self, track: Track) -> Option<&TrackScore> {
        self.scores.iter().find(|item| item.track == track)
    }

    pub fn descriptive_mean(&self) -> u32 {
        let hits: u32 = self.scores.iter().map(|s| s.hits).sum();
        let total: u32 = self.scores.iter().map(|s| s.total).sum();
        if total == 0 {
            0
        } else {
            hits * 100 / total
        }
    }

    pub fn regressions_against(&self, baseline: &MmEvalReport) -> Vec<Track> {
        let mut out = Vec::new();
        for score in &self.scores {
            if let Some(prev) = baseline.score(score.track) {
                if score.hits < prev.hits {
                    out.push(score.track);
                }
            }
        }
        out
    }
}

pub fn smoke_suite(commit: &str, config: &str, vision_hits: u32) -> MmEvalReport {
    MmEvalReport {
        schema_version: MM_EVAL_SCHEMA_VERSION,
        commit: commit.into(),
        config: config.into(),
        scores: vec![
            TrackScore {
                track: Track::Text,
                hits: 3,
                total: 3,
            },
            TrackScore {
                track: Track::Vision,
                hits: vision_hits.min(3),
                total: 3,
            },
            TrackScore {
                track: Track::Audio,
                hits: 2,
                total: 2,
            },
            TrackScore {
                track: Track::CrossModal,
                hits: 1,
                total: 1,
            },
        ],
        cost: EvalCost {
            tokens: 3,
            patches: 4,
            frames: 2,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unimodal_and_cross_modal_are_separate() {
        let report = smoke_suite("abc", "tiny", 3);
        assert!(report.score(Track::Text).unwrap().pass());
        assert!(report.score(Track::Vision).unwrap().pass());
        assert!(report.score(Track::Audio).unwrap().pass());
        assert!(report.score(Track::CrossModal).unwrap().pass());
        assert_eq!(report.commit, "abc");
        assert_eq!(report.config, "tiny");
    }

    #[test]
    fn vision_regression_is_detected_even_if_mean_looks_fine() {
        let baseline = smoke_suite("aaa", "tiny", 3);
        let worse = smoke_suite("bbb", "tiny", 1);
        assert_eq!(worse.regressions_against(&baseline), vec![Track::Vision]);
        assert!(worse.descriptive_mean() >= 50);
        assert!(!worse.score(Track::Vision).unwrap().pass());
        assert!(worse.score(Track::Text).unwrap().pass());
    }
}
