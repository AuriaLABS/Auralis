//! #77 multi-objective candidate comparison.
//!
//! No single score is a pass/fail gate.

pub const PARETO_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Quality,
    Cost,
    Memory,
    Latency,
}

impl Axis {
    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Quality)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultVector {
    pub id: String,
    pub quality: Option<u32>,
    pub cost: Option<u32>,
    pub memory: Option<u32>,
    pub latency: Option<u32>,
}

impl ResultVector {
    pub fn value(&self, axis: Axis) -> Option<u32> {
        match axis {
            Axis::Quality => self.quality,
            Axis::Cost => self.cost,
            Axis::Memory => self.memory,
            Axis::Latency => self.latency,
        }
    }

    pub fn complete(&self) -> bool {
        [Axis::Quality, Axis::Cost, Axis::Memory, Axis::Latency]
            .into_iter()
            .all(|axis| self.value(axis).is_some())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Dominates,
    Dominated,
    TradeOff,
    Incomparable,
}

pub fn compare(a: &ResultVector, b: &ResultVector) -> Compare {
    if !a.complete() || !b.complete() {
        return Compare::Incomparable;
    }
    let axes = [Axis::Quality, Axis::Cost, Axis::Memory, Axis::Latency];
    let mut better = 0;
    let mut worse = 0;
    for axis in axes {
        let left = a.value(axis).unwrap();
        let right = b.value(axis).unwrap();
        let (win, lose) = if axis.higher_is_better() {
            (left > right, left < right)
        } else {
            (left < right, left > right)
        };
        if win {
            better += 1;
        }
        if lose {
            worse += 1;
        }
    }
    match (better, worse) {
        (0, 0) => Compare::TradeOff,
        (_, 0) => Compare::Dominates,
        (0, _) => Compare::Dominated,
        _ => Compare::TradeOff,
    }
}

pub fn front<'a>(rows: &'a [ResultVector]) -> Vec<&'a ResultVector> {
    rows.iter()
        .filter(|cand| {
            cand.complete()
                && !rows.iter().any(|other| {
                    other.id != cand.id && compare(other, cand) == Compare::Dominates
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, q: Option<u32>, c: Option<u32>, m: Option<u32>, l: Option<u32>) -> ResultVector {
        ResultVector {
            id: id.into(),
            quality: q,
            cost: c,
            memory: m,
            latency: l,
        }
    }

    #[test]
    fn dominance_and_tradeoff_are_separate() {
        let good = row("a", Some(9), Some(1), Some(1), Some(1));
        let worse = row("b", Some(8), Some(2), Some(2), Some(2));
        let trade = row("c", Some(10), Some(5), Some(1), Some(1));
        assert_eq!(compare(&good, &worse), Compare::Dominates);
        assert_eq!(compare(&worse, &good), Compare::Dominated);
        assert_eq!(compare(&good, &trade), Compare::TradeOff);
        assert_eq!(front(&[good.clone(), worse, trade.clone()]).len(), 2);
    }

    #[test]
    fn incomplete_is_incomparable() {
        let full = row("a", Some(1), Some(1), Some(1), Some(1));
        let hole = row("b", Some(9), None, Some(1), Some(1));
        assert_eq!(compare(&full, &hole), Compare::Incomparable);
        assert!(front(&[full.clone(), hole]).iter().all(|r| r.id == "a"));
    }
}
