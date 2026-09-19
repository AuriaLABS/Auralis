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

    /// Borrow two ordered, non-overlapping slots mutably at the same time.
    ///
    /// The caller must pass slots in allocation order. This keeps the
    /// implementation entirely safe: the backing slice is split at the start
    /// of the second slot, so the returned regions cannot alias.
    pub fn get2_mut(&mut self, a: Slot, b: Slot) -> (&mut [f32], &mut [f32]) {
        assert_ordered_non_overlapping(a, b);
        assert!(slot_end(b) <= self.buf.len(), "arena slot out of bounds");

        let (before_b, from_b) = self.buf.split_at_mut(b.start);
        let a_slice = &mut before_b[a.start..slot_end(a)];
        let b_slice = &mut from_b[..b.len];
        (a_slice, b_slice)
    }

    /// Borrow three ordered, non-overlapping slots mutably at the same time.
    pub fn get3_mut(
        &mut self,
        a: Slot,
        b: Slot,
        c: Slot,
    ) -> (&mut [f32], &mut [f32], &mut [f32]) {
        assert_ordered_non_overlapping(a, b);
        assert_ordered_non_overlapping(b, c);
        assert!(slot_end(c) <= self.buf.len(), "arena slot out of bounds");

        let (before_b, from_b) = self.buf.split_at_mut(b.start);
        let a_slice = &mut before_b[a.start..slot_end(a)];

        let c_from_b = c.start - b.start;
        let (before_c, from_c) = from_b.split_at_mut(c_from_b);
        let b_slice = &mut before_c[..b.len];
        let c_slice = &mut from_c[..c.len];
        (a_slice, b_slice, c_slice)
    }

    /// Borrow four ordered, non-overlapping slots mutably at the same time.
    pub fn get4_mut(
        &mut self,
        a: Slot,
        b: Slot,
        c: Slot,
        d: Slot,
    ) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        assert_ordered_non_overlapping(a, b);
        assert_ordered_non_overlapping(b, c);
        assert_ordered_non_overlapping(c, d);
        assert!(slot_end(d) <= self.buf.len(), "arena slot out of bounds");

        let (before_b, from_b) = self.buf.split_at_mut(b.start);
        let a_slice = &mut before_b[a.start..slot_end(a)];

        let c_from_b = c.start - b.start;
        let (before_c, from_c) = from_b.split_at_mut(c_from_b);
        let b_slice = &mut before_c[..b.len];

        let d_from_c = d.start - c.start;
        let (before_d, from_d) = from_c.split_at_mut(d_from_c);
        let c_slice = &mut before_d[..c.len];
        let d_slice = &mut from_d[..d.len];

        (a_slice, b_slice, c_slice, d_slice)
    }
}

fn slot_end(slot: Slot) -> usize {
    slot.start
        .checked_add(slot.len)
        .expect("arena slot end overflow")
}

fn assert_ordered_non_overlapping(left: Slot, right: Slot) {
    assert!(
        slot_end(left) <= right.start,
        "arena slots must be ordered and non-overlapping"
    );
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
    fn multi_slot_mut_access_keeps_regions_disjoint() {
        let mut arena = Arena::new();
        let a = arena.alloc(2);
        let b = arena.alloc(3);
        let c = arena.alloc(1);
        let d = arena.alloc(4);

        {
            let (aa, bb) = arena.get2_mut(a, b);
            aa.fill(1.0);
            bb.fill(2.0);
        }
        {
            let (aa, bb, cc) = arena.get3_mut(a, b, c);
            aa[0] = 3.0;
            bb[0] = 4.0;
            cc[0] = 5.0;
        }
        {
            let (aa, bb, cc, dd) = arena.get4_mut(a, b, c, d);
            aa[1] = 6.0;
            bb[2] = 7.0;
            cc[0] = 8.0;
            dd.fill(9.0);
        }

        assert_eq!(arena.get(a), &[3.0, 6.0]);
        assert_eq!(arena.get(b), &[4.0, 2.0, 7.0]);
        assert_eq!(arena.get(c), &[8.0]);
        assert_eq!(arena.get(d), &[9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "ordered and non-overlapping")]
    fn multi_slot_mut_rejects_reversed_slots() {
        let mut arena = Arena::new();
        let a = arena.alloc(2);
        let b = arena.alloc(2);
        let _ = arena.get2_mut(b, a);
    }

    #[test]
    #[should_panic(expected = "ordered and non-overlapping")]
    fn multi_slot_mut_rejects_overlapping_slots() {
        let mut arena = Arena::new();
        let _ = arena.alloc(6);
        let a = Slot { start: 1, len: 3 };
        let b = Slot { start: 3, len: 2 };
        let _ = arena.get2_mut(a, b);
    }

    #[test]
    fn multi_slot_mut_supports_gaps_and_empty_slots() {
        let mut arena = Arena::new();
        let a = arena.alloc(2);
        let _gap = arena.alloc(3);
        let b = arena.alloc(0);
        let c = arena.alloc(2);
        let d = arena.alloc(1);

        let (aa, bb, cc, dd) = arena.get4_mut(a, b, c, d);
        aa.fill(1.0);
        assert!(bb.is_empty());
        cc.fill(2.0);
        dd.fill(3.0);

        assert_eq!(arena.get(a), &[1.0, 1.0]);
        assert_eq!(arena.get(c), &[2.0, 2.0]);
        assert_eq!(arena.get(d), &[3.0]);
    }

    #[test]
    fn get4_mut_exhaustive_small_layouts_preserve_disjoint_regions() {
        for a_len in 0..=3 {
            for gap_ab in 0..=2 {
                for b_len in 0..=3 {
                    for gap_bc in 0..=2 {
                        for c_len in 0..=3 {
                            for gap_cd in 0..=2 {
                                for d_len in 0..=3 {
                                    let mut arena = Arena::new();
                                    let a = arena.alloc(a_len);
                                    let gap1 = arena.alloc(gap_ab);
                                    let b = arena.alloc(b_len);
                                    let gap2 = arena.alloc(gap_bc);
                                    let c_slot = arena.alloc(c_len);
                                    let gap3 = arena.alloc(gap_cd);
                                    let d = arena.alloc(d_len);

                                    arena.fill(gap1, -1.0);
                                    arena.fill(gap2, -2.0);
                                    arena.fill(gap3, -3.0);

                                    {
                                        let (aa, bb, cc, dd) =
                                            arena.get4_mut(a, b, c_slot, d);
                                        aa.fill(11.0);
                                        bb.fill(22.0);
                                        cc.fill(33.0);
                                        dd.fill(44.0);
                                    }

                                    assert!(arena.get(a).iter().all(|&x| x == 11.0));
                                    assert!(arena.get(b).iter().all(|&x| x == 22.0));
                                    assert!(arena.get(c_slot).iter().all(|&x| x == 33.0));
                                    assert!(arena.get(d).iter().all(|&x| x == 44.0));
                                    assert!(arena.get(gap1).iter().all(|&x| x == -1.0));
                                    assert!(arena.get(gap2).iter().all(|&x| x == -2.0));
                                    assert!(arena.get(gap3).iter().all(|&x| x == -3.0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "arena slot out of bounds")]
    fn multi_slot_mut_rejects_out_of_bounds_last_slot() {
        let mut arena = Arena::new();
        let a = arena.alloc(2);
        let b = arena.alloc(2);
        let c_slot = arena.alloc(2);
        let _ = arena.alloc(2);
        let forged_d = Slot { start: 6, len: 3 };
        let _ = arena.get4_mut(a, b, c_slot, forged_d);
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
