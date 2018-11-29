
extern crate rand;
extern crate core;

use std::collections::HashMap;
use std::f64::consts::E;
use self::rand::prelude::*;
use self::rand::{XorShiftRng,SeedableRng};

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
            sigmoid: Sigmoid::Hard,
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
pub fn learn(w: &mut Vec<f64>, Xy: Vec<(Sparse, bool)>, options: &SGDOptions) -> () {

    let mut lr = options.learning_rule.clone();

    // filter out empty vecs
    let Xy: Vec<_> = Xy.into_iter().filter(|x| (x.0).len() > 0).collect();
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

    let mut logloss = 0f64;
    let mut c = 0f64;
    let mut idxs: Vec<_> = (0usize..len).collect();
    let mut cur_idx = 0usize;
    let mut rng = XorShiftRng::seed_from_u64(options.seed);
    for iter in 0..iters {
        grads.clear();
        logloss = 0.;
        // Get random batch
        // update weights
        let alpha = lr.get_lr();
        for _ in 0..options.batch_size {
            if cur_idx == 0 {
                rng.shuffle(&mut idxs);
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

/*
#[allow(non_snake_case)]
pub fn _learn(w: &mut Vec<f64>, Xy: Vec<(Sparse, bool)>, options: &SGDOptions) -> f64 {

    let mut lr = options.learning_rule.clone();

    // filter out empty vecs
    let Xy: Vec<_> = Xy.into_iter().filter(|x| (x.0).len() > 0).collect();
    let len = Xy.len();

    // Importance weights - We use inverse proportion to scale the pos/negative weights
    let (mut p_w, mut n_w) = match options.class_weight {
        ClassWeight::None => (1f64, 1f64),
        ClassWeight::Inverse => compute_balanced_weight(&Xy),
        ClassWeight::Fixed(p_w, n_w) => (p_w, n_w)
    };

    let mut logloss = 0f64;
    let mut c = 0f64;

    let mut idxs: Vec<_> = (0usize..len).collect();
    for iter in 0..options.iters {
        logloss = 0.;
        c = 0.;
        let alpha = lr.get_lr();
        thread_rng().shuffle(&mut idxs);
        let mut cur_idx = 0usize;
        while (cur_idx + options.batch_size) <= len {
            let mut grads = vec![0f64; w.len()];
            for _ in 0..options.batch_size {
                let idx = idxs[cur_idx];
                cur_idx += 1;
                let (ref Xi, ref y) = Xy[idx];
                let (yi, iw) = if *y { (1f64, p_w) } else { (0f64, n_w) };  

                // forward pass
                let y_hat = match options.sigmoid {
                    Sigmoid::Hard => hard_sigmoid(dot(Xi, &w)),
                    Sigmoid::Normal => sigmoid(dot(Xi, &w))
                };

                if iter + 1 == options.iters {
                    let ll = log_loss(yi, clip(y_hat));
                    logloss += ll;
                    c += 1f64;
                }

                for &(idx, xi) in Xi.iter() {
                    let g = (y_hat - yi) * xi; // * iw;
                    grads[idx] += g;
                }
            }

            for (idx, g) in grads.into_iter().enumerate() {
                w[idx as usize] -= (g / options.batch_size as f64) * alpha;
            }
        }
    }
    logloss / c
}

*/

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

#[inline]
pub fn dot(x: &Sparse, w: &Vec<f64>) -> f64 {
    //x.iter().map(|(ref idx, ref xi)| xi * w[*idx as usize]).sum()
    let mut sum = 0f64;
    for &(ref idx, ref xi) in x.iter() {
        sum += xi * w[*idx as usize];
    }
    sum
}

#[allow(non_snake_case)]
fn compute_balanced_weight<A>(Xy: &[(A, bool)]) -> (f64, f64) {
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
        let (p_w, n_w) = compute_balanced_weight(&v);
        assert_eq!(p_w, 2. / 3.);
        assert_eq!(n_w, 2.);
    }
}
