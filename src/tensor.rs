//! Algebra minima. Matrices densas f32.

#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 { self.data[r * self.cols + c] }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) { self.data[r * self.cols + c] = v; }
    #[inline]
    pub fn add_at(&mut self, r: usize, c: usize, v: f32) { self.data[r * self.cols + c] += v; }
    pub fn fill(&mut self, v: f32) { for x in &mut self.data { *x = v; } }
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) { *a += *b; }
    }
    pub fn scale(&mut self, s: f32) { for x in &mut self.data { *x *= s; } }
    pub fn matmul(a: &Mat, b: &Mat) -> Mat {
        assert_eq!(a.cols, b.rows);
        let mut out = Mat::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for k in 0..a.cols {
                let aik = a.get(i, k);
                if aik == 0.0 { continue; }
                for j in 0..b.cols {
                    out.data[i * out.cols + j] += aik * b.get(k, j);
                }
            }
        }
        out
    }
    pub fn matmul_bt(a: &Mat, b: &Mat) -> Mat {
        assert_eq!(a.cols, b.cols);
        let mut out = Mat::zeros(a.rows, b.rows);
        for i in 0..a.rows {
            for j in 0..b.rows {
                let mut s = 0.0;
                for k in 0..a.cols { s += a.get(i, k) * b.get(j, k); }
                out.set(i, j, s);
            }
        }
        out
    }
    pub fn matmul_at(a: &Mat, b: &Mat) -> Mat {
        assert_eq!(a.rows, b.rows);
        let mut out = Mat::zeros(a.cols, b.cols);
        for k in 0..a.rows {
            for i in 0..a.cols {
                let aki = a.get(k, i);
                for j in 0..b.cols {
                    out.data[i * out.cols + j] += aki * b.get(k, j);
                }
            }
        }
        out
    }
    pub fn add(a: &Mat, b: &Mat) -> Mat {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        let data = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
        Mat::from_vec(a.rows, a.cols, data)
    }
    pub fn softmax_rows(m: &Mat) -> Mat {
        let mut out = Mat::zeros(m.rows, m.cols);
        for i in 0..m.rows {
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..m.cols { maxv = maxv.max(m.get(i, j)); }
            let mut sum = 0.0;
            for j in 0..m.cols {
                let e = (m.get(i, j) - maxv).exp();
                out.set(i, j, e);
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for j in 0..m.cols {
                let v = out.get(i, j) * inv;
                out.set(i, j, v);
            }
        }
        out
    }
    pub fn gelu(x: f32) -> f32 {
        const C: f32 = 0.7978845608;
        0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
    }
    pub fn gelu_deriv(x: f32) -> f32 {
        const C: f32 = 0.7978845608;
        let u = C * (x + 0.044715 * x * x * x);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let du = C * (1.0 + 3.0 * 0.044715 * x * x);
        0.5 * (1.0 + t) + 0.5 * x * sech2 * du
    }
}

#[cfg(test)]
mod tests {
    use super::Mat;
    #[test]
    fn matmul_identity() {
        let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let i = Mat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let c = Mat::matmul(&a, &i);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-6);
    }
    #[test]
    fn softmax_rows_sum_one() {
        let m = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let s = Mat::softmax_rows(&m);
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
