//! Reusable f32 scratch arena for Engine temporaries (#87).
//!
//! This module does not wire into `model.rs` yet (#27 owns forward/caches).
//! Callers that later adopt it must `reset()` once per step/phase and treat
//! returned slots as exclusive until the next reset.

#[derive(Clone, Debug)]
pub struct Arena {
    buf: Vec<f32>,
    offset: usize,
    resets: u64,
    growths: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    start: usize,
    len: usize,
}

impl Slot {
    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

impl Arena {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            buf: Vec::with_capacity(n),
            offset: 0,
            resets: 0,
            growths: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    pub fn used(&self) -> usize {
        self.offset
    }

    pub fn resets(&self) -> u64 {
        self.resets
    }

    pub fn growths(&self) -> u64 {
        self.growths
    }

    /// Rewind the bump pointer. Does **not** zero memory; that is the reuse win.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.resets += 1;
    }

    /// Rewind and write NaN over the used region so stale reads are obvious in tests.
    pub fn reset_poison(&mut self) {
        let end = self.offset;
        for slot in &mut self.buf[..end] {
            *slot = f32::NAN;
        }
        self.reset();
    }

    /// Reserve `n` floats from the bump pointer. Grows the backing store if needed.
    /// Memory beyond the previous length is zeroed so first-time reads are defined.
    pub fn alloc(&mut self, n: usize) -> Slot {
        let start = self.offset;
        let end = start.checked_add(n).expect("arena allocation overflow");
        if end > self.buf.capacity() {
            self.growths += 1;
            self.buf.reserve(end - self.buf.capacity());
        }
        if end > self.buf.len() {
            self.buf.resize(end, 0.0);
        }
        self.offset = end;
        Slot { start, len: n }
    }

    pub fn get(&self, slot: Slot) -> &[f32] {
        &self.buf[slot.start..slot.start + slot.len]
    }

    pub fn get_mut(&mut self, slot: Slot) -> &mut [f32] {
        &mut self.buf[slot.start..slot.start + slot.len]
    }

    pub fn fill(&mut self, slot: Slot, value: f32) {
        self.get_mut(slot).fill(value);
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_across_steps_does_not_grow() {
        let mut arena = Arena::with_capacity(16);
        let a = arena.alloc(8);
        arena.fill(a, 1.0);
        let first_cap = arena.capacity();
        arena.reset();
        let b = arena.alloc(8);
        assert_eq!(arena.get(b), &[1.0; 8], "reset keeps backing bytes dirty");
        arena.fill(b, 2.0);
        assert_eq!(arena.capacity(), first_cap);
        assert_eq!(arena.growths(), 0);
        assert_eq!(arena.resets(), 1);
    }

    #[test]
    fn variable_shapes_grow_once_then_reuse() {
        let mut arena = Arena::new();
        let _ = arena.alloc(4);
        assert!(arena.growths() >= 1);
        let growths_after_first = arena.growths();
        arena.reset();
        let _ = arena.alloc(4);
        let wide = arena.alloc(12);
        assert_eq!(wide.len(), 12);
        arena.reset();
        let _ = arena.alloc(16);
        assert_eq!(arena.growths(), growths_after_first + 1);
        let cap = arena.capacity();
        arena.reset();
        let _ = arena.alloc(16);
        assert_eq!(arena.capacity(), cap);
    }

    #[test]
    fn slots_do_not_alias_within_a_phase() {
        let mut arena = Arena::new();
        let a = arena.alloc(3);
        let b = arena.alloc(3);
        arena.fill(a, 1.0);
        arena.fill(b, 2.0);
        assert_eq!(arena.get(a), &[1.0, 1.0, 1.0]);
        assert_eq!(arena.get(b), &[2.0, 2.0, 2.0]);
        assert_eq!(a.start + a.len, b.start);
    }

    #[test]
    fn poison_makes_stale_reads_nan() {
        let mut arena = Arena::new();
        let a = arena.alloc(2);
        arena.fill(a, 3.0);
        arena.reset_poison();
        let b = arena.alloc(2);
        assert!(arena.get(b).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn empty_alloc_is_valid() {
        let mut arena = Arena::new();
        let z = arena.alloc(0);
        assert!(z.is_empty());
        assert_eq!(arena.used(), 0);
    }
}
