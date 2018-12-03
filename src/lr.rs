
extern crate rand;

use std::arch::x86_64::*;
use std::mem;
use std::f64::consts::E;
use self::rand::{XorShiftRng,SeedableRng};
use self::rand::seq::SliceRandom;

#[derive(Clone)]
pub enum ClassWeight {
    None,
    Inverse,
    Fixed(f64, f64)
}

#[derive(Clone)]
pub enum Sigmoid {
    Normal,
    Hard
}

#[derive(Clone)]
pub struct VWRule {
    alpha: f64,
    power_t: f64,
    t: u64
}

impl VWRule {
    pub fn new(alpha: f64, power_t: f64) -> LearningRule {
        LearningRule::VW(VWRule {
            alpha: alpha,
            power_t: power_t,
            t: 0
        })
    }
}

#[derive(Clone)]
pub enum LearningRule {
    Constant(f64),
    Exponential(f64, f64),
    VW(VWRule)
}

impl LearningRule {
    pub fn get_lr(&mut self) -> f64 {
        match self {
            LearningRule::Constant(lr) => *lr,
            LearningRule::Exponential(alpha, decay) => {
                *alpha *= *decay;
                *alpha
            },
            LearningRule::VW(vr) => {
                vr.t += 1;
                let lr = vr.alpha * (1. / (1. + vr.t as f64)).powf(vr.power_t);
                lr
            }
        }
    }
}

#[derive(Clone)]
pub struct SGDOptions {
    pub iters: u32,
    pub batch_size: usize,
    pub seed: u64,
    pub class_weight: ClassWeight,
    pub sigmoid: Sigmoid,
    pub learning_rule: LearningRule
}

impl SGDOptions {
    pub fn new(iters: u32, batch_size: usize) -> Self {
        SGDOptions {
            iters: iters,
            batch_size: batch_size,
            seed: 1234,
            class_weight: ClassWeight::None,
            sigmoid: Sigmoid::Normal,
            learning_rule: LearningRule::Constant(1.)
        }
    }

    pub fn learning_rule(mut self, lr: LearningRule) -> Self {
        self.learning_rule = lr;
        self
    }

    pub fn set_seed(&mut self, seed: u64) -> () {
        self.seed = seed;
    }

    pub fn set_class_weight(&mut self, cw: ClassWeight) -> () {
        self.class_weight = cw;
    }

    pub fn set_sigmoid(&mut self, s: Sigmoid) -> () {
        self.sigmoid = s;
    }

}

pub type Sparse = Vec<(usize, f64)>;

#[allow(non_snake_case)]
pub fn learn(w: &mut Vec<f64>, input: &Vec<(Sparse, bool)>, options: &SGDOptions) -> () {

    let mut lr = options.learning_rule.clone();

    // filter out empty vecs
    let Xy: Vec<_> = input.iter().filter(|x| (x.0).len() > 0).collect();
    let len = Xy.len();
    let mut grads = Vec::new();
    let scale = 1f64 / options.batch_size as f64;

    // Number of iterations to run
    let iters = (len as f64 / options.batch_size as f64) as u32 * options.iters;

    // Importance weights - We use inverse proportion to scale the pos/negative weights
    let (mut p_w, mut n_w) = match options.class_weight {
        ClassWeight::None => (1f64, 1f64),
        ClassWeight::Inverse => compute_balanced_weight(&Xy),
        ClassWeight::Fixed(p_w, n_w) => (p_w, n_w)
    };

    p_w *= scale;
    n_w *= scale;

    let mut idxs: Vec<_> = (0usize..len).collect();
    let mut cur_idx = 0usize;
    let mut rng = XorShiftRng::seed_from_u64(options.seed);
    for _iter in 0..iters {
        grads.clear();
        // Get random batch
        // update weights
        let alpha = lr.get_lr();
        for _ in 0..options.batch_size {
            if cur_idx == 0 {
                idxs.as_mut_slice().shuffle(&mut rng);
            };
            cur_idx = (cur_idx + 1) % len;
            let idx = idxs[cur_idx];
            let (ref Xi, ref y) = Xy[idx];
            let (yi, iw) = if *y { (1f64, p_w) } else { (0f64, n_w) };  

            // forward pass
            let y_hat = match options.sigmoid {
                Sigmoid::Hard => hard_sigmoid(dot(Xi, &w)),
                Sigmoid::Normal => sigmoid(dot(Xi, &w))
            };

            let denom = alpha * iw * (y_hat - yi);

            for &(idx, xi) in Xi.iter() {
                let g = xi * denom;
                grads.push((idx, g));
            }
        }

        for (idx, g) in grads.drain(0..) {
            w[idx as usize] -= g;
        }
    }
}

pub fn test(w: &Vec<f64>, xy: &Vec<(Sparse, bool)>) -> (f64, f64) {
    let mut misclass = 0;
    let mut logloss = 0.;
    for (x, y) in xy.iter() {
        let y_hat = dot(&x, &w);
        let yi = if *y { 1.} else { 0. };
        logloss += log_loss(yi, clip(sigmoid(y_hat)));
        if (y_hat > 0.) != *y {
            misclass += 1;
        }
    }
    let err = (misclass as f64) / (xy.len() as f64);
    let ll = (logloss as f64) / (xy.len() as f64);
    (err, ll)
}

#[inline]
fn hard_sigmoid(x: f64) -> f64 {
    0f64.max(1f64.min(x * 0.2 + 0.5))
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1. / (1. + E.powf(-x))
}


#[inline]
fn clip(v: f64) -> f64 {
    let eps = 1e-15f64;
    if v == 0. {
        eps
    } else if v == 1. {
        1. - eps
    } else {
        v
    }
}

#[inline]
fn log_loss(y: f64, y_hat: f64) -> f64 {
    let out = -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln());
    if out.is_nan() { 0. } else {out}
}

pub fn dot(x: &Sparse, w: &[f64]) -> f64 {
    if is_x86_feature_detected!("avx2") {
        unsafe { dot_avx(x, w) }
    } else {
        dot_basic(x, w)
    }

}

#[inline]
fn dot_basic(x: &Sparse, w: &[f64]) -> f64 {
    let mut sum = 0f64;
    for &(ref idx, ref xi) in x.iter() {
        sum += xi * w[*idx as usize];
    }
    sum
}

/// AVX based dot product
#[target_feature(enable = "avx2")]
unsafe fn dot_avx(x: &[(usize, f64)], w: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    if x.len() >= 4 { 
        let mut sv = _mm256_setzero_pd();
        while i < (x.len() - 4) {
            let l = _mm256_set_pd(x[i].1, x[i+1].1, x[i+2].1, x[i+3].1); 
            let r = _mm256_set_pd(w[x[i].0], w[x[i+1].0], w[x[i+2].0], w[x[i+3].0]); 
            sv = _mm256_fmadd_pd(l, r, sv);
            i += 4;
        }
        let mut extract: [f64; 4] = mem::uninitialized();
        _mm256_store_pd(&extract[0] as *const f64, sv);
        sum = extract[0] + extract[1] + extract[2] + extract[3];
    }

    // Remainder
    for i in i..x.len() {
        sum += x[i].1 * w[x[i].0];
    }
    sum 
}

#[allow(non_snake_case)]
fn compute_balanced_weight<A>(Xy: &[&(A, bool)]) -> (f64, f64) {
    let len = Xy.len();
    let p_count: f64 = Xy.iter().map(|x| if x.1 {1.} else {0.}).sum();
    let p_w = len as f64 / (2. * p_count);
    let n_w = len as f64 / (2. * (len as f64 - p_count));
    (p_w, n_w)
}

#[cfg(test)]
mod def_test {
    use super::*;

    #[test]
    fn test_balanced_weight() {
        let v = vec![(1, true), (2, true), (3, true), (4, false)];
        let v_test = v.iter().collect();
        let (p_w, n_w) = compute_balanced_weight(&v_test);
        assert_eq!(p_w, 2. / 3.);
        assert_eq!(n_w, 2.);
    }
}
