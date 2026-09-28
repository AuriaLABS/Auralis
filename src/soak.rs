//! #82 release soak and resume stability.

pub const SOAK_SCHEMA_VERSION: u32 = 1;
pub const MAX_RSS: u32 = 64;
pub const MAX_RETRIES: u32 = 0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SoakError {
    NonFinite,
    RssGrowth,
    Retry,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoakStep {
    pub loss: u32,
    pub rss: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoakReport {
    pub steps: Vec<SoakStep>,
    pub fingerprint: u32,
}

pub fn run(steps: u32, resume_from: Option<&SoakReport>) -> Result<SoakReport, SoakError> {
    if MAX_RETRIES != 0 {
        return Err(SoakError::Retry);
    }
    if let Some(prev) = resume_from {
        if prev.steps.iter().any(|s| s.rss > MAX_RSS) {
            return Err(SoakError::RssGrowth);
        }
    }
    let start = resume_from.map(|r| r.steps.len() as u32).unwrap_or(0);
    let mut report = resume_from.cloned().unwrap_or(SoakReport {
        steps: Vec::new(),
        fingerprint: 0,
    });
    for i in start..steps {
        let loss = 100u32.saturating_sub(i);
        let rss = 8 + (i % 3);
        if rss > MAX_RSS {
            return Err(SoakError::RssGrowth);
        }
        report.steps.push(SoakStep { loss, rss });
        report.fingerprint = report
            .fingerprint
            .wrapping_mul(33)
            .wrapping_add(loss)
            .wrapping_add(rss);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soak_and_resume_keep_fingerprint() {
        let full = run(8, None).unwrap();
        let mid = run(4, None).unwrap();
        let resumed = run(8, Some(&mid)).unwrap();
        assert_eq!(full.fingerprint, resumed.fingerprint);
        assert_eq!(full.steps.len(), 8);
        assert!(full.steps.iter().all(|s| s.rss <= MAX_RSS));
        assert_eq!(MAX_RETRIES, 0);
    }

    #[test]
    fn rss_growth_fails_closed() {
        let err = run(8, Some(&SoakReport {
            steps: vec![SoakStep { loss: 1, rss: MAX_RSS + 1 }; 1],
            fingerprint: 0,
        }));
        assert_eq!(err.unwrap_err(), SoakError::RssGrowth);
        assert!(run(0, None).unwrap().steps.is_empty());
    }
}
