//! Adam implementado a mano sobre vectores de parámetros.

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: i32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(n: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    pub fn from_state(lr: f32, t: i32, m: Vec<f32>, v: Vec<f32>) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t,
            m,
            v,
        }
    }

    pub fn export(&self) -> (f32, i32, &[f32], &[f32]) {
        (self.lr, self.t, &self.m, &self.v)
    }

    pub fn step(&mut self, w: &mut [f32], g: &[f32]) {
        assert_eq!(w.len(), g.len());
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);
        for i in 0..w.len() {
            let gi = g[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gi;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gi * gi;
            let mhat = self.m[i] / b1t;
            let vhat = self.v[i] / b2t;
            w[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}
