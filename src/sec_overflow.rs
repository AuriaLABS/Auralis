//! Fail-closed arithmetic for untrusted sizes.

pub fn checked_add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("overflow: {a} + {b}"))
}

pub fn checked_mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("overflow: {a} * {b}"))
}

/// Product of dimensions. Empty slice is 0.
pub fn checked_product(dims: &[usize]) -> Result<usize, String> {
    let mut acc = 1usize;
    if dims.is_empty() {
        return Ok(0);
    }
    for (i, d) in dims.iter().enumerate() {
        acc = acc
            .checked_mul(*d)
            .ok_or_else(|| format!("overflow at dim {i}: product * {d}"))?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_mul_ok() {
        assert_eq!(checked_add(2, 3).unwrap(), 5);
        assert_eq!(checked_mul(4, 5).unwrap(), 20);
    }

    #[test]
    fn add_overflows() {
        assert!(checked_add(usize::MAX, 1).is_err());
    }

    #[test]
    fn mul_overflows() {
        assert!(checked_mul(usize::MAX, 2).is_err());
    }

    #[test]
    fn product_tracks_index() {
        assert_eq!(checked_product(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_product(&[]).unwrap(), 0);
        let err = checked_product(&[usize::MAX, 2]).unwrap_err();
        assert!(err.contains("dim 1"));
    }
}
