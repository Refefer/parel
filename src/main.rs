extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate tange;
extern crate tange_collection;
extern crate svmloader;
extern crate env_logger;

mod lr;

use std::env::args;
use std::sync::Arc;

use tange::deferred::{Deferred, tree_reduce};
use tange::scheduler::GreedyScheduler;
use tange::scheduler::LeveledScheduler;
use tange_collection::utils::read_text;
use tange_collection::interfaces::Stream;
use tange_collection::collection::memory::MemoryCollection;
use svmloader::types::SparseData;
use svmloader::*;

use lr::{SGDOptions,Sparse,learn,test,ClassWeight,LearningRule,VWRule};

#[derive(Serialize,Deserialize,Clone)]
pub struct MySparse(pub usize, pub Vec<usize>, pub Vec<f32>);

fn load_data(path: &str, dims: usize) -> MemoryCollection<(Vec<(usize, f64)>, bool)> {
    let col = read_text(path, 1_000_000).expect("File missing!");
    let sd = SparseData(dims);
    let mut training_data = col.emit(move |s, emitter| {
        if let Some(row) = parse_line(&BinaryClassification, &sd, &s) {
            let v: Vec<(usize,f64)> = row.x.1.into_iter()
                .zip(row.x.2.into_iter().map(|x| x as f64)).collect();
            emitter((v, row.y));
        }
    });
    training_data
}

fn main() {
    env_logger::init();
    let path: String = args().skip(1).next().unwrap();
    let dims: usize = args().skip(2).next().unwrap().parse().unwrap();
    let partitions: usize = args().skip(3).next().unwrap().parse().ok().unwrap();
    let test_file: Option<String> = args().skip(4).next();

    let mut training_data = load_data(&path, dims);

    let test_data = if let Some(test_path) = test_file {
        load_data(&test_path, dims)
    } else {
        training_data.clone()
    };
    
    // Create our initial weight vector
    let mut w = Deferred::lift(vec![0f64; dims], None);

    let mut errors = Vec::new();
    training_data = training_data.split(partitions);
    for pass in 0..20 {
        //let alpha = 5e-3 / ((pass + 2) as f64).ln();
        let alpha = 0.5;
        let mut opts = SGDOptions::new(8, 8)
            .learning_rule(LearningRule::Constant(alpha));

        //opts.set_class_weight(ClassWeight::Inverse);
        opts.set_class_weight(ClassWeight::None);
        opts.set_seed(pass + 1);
        let sgd = Arc::new(opts);

        // Shuffle training data
        if pass % 5 == 0 {
            training_data = training_data.split(partitions);
        }

        // Train
        let out: Vec<_> = training_data.to_defs().iter().map(|d| {
            let opts = sgd.clone();
            d.join(&w, move |td, v| {
                let mut my_w = v.clone();
                let training: Vec<_> = td.stream().into_iter().collect();
                let err = learn(&mut my_w, training, &opts);
                (1u32, my_w)
            })
        }).collect();

        // Average Ws together
        let avg_w = tree_reduce(&out, |(c1, left), (c2, right)| {
            let c3 = c1 + c2;
            let mut out: Vec<_> = left.clone();
            for i in 0..left.len() {
                out[i] += right[i];
            }
            (c3, out)
        }).unwrap();

        w = avg_w.apply(|(count, v)| {
            let mut w = v.clone();
            let c = *count as f64;
            if c > 1. {
                for i in 0..w.len() {
                    w[i] /= c;
                }
            }
            w
        });

        // Test
        let misclass: Vec<_> = test_data.to_defs().iter().map(|d| {
            d.join(&w, move |td, w| {
                let training: Vec<_> = td.stream().into_iter().collect();
                let res = test(w, &training);
                res.0
            })
        }).collect();

        let sum = tree_reduce(&misclass, |left, right| left + right).unwrap();
        let n = misclass.len();
        errors.push(sum.apply(move |s| {
            let err = s / (n as f64);
            vec![(pass, err)]
        }));
    }

    let mut gs = GreedyScheduler::new();
    //gs.set_threads(1);
    //let gs = LeveledScheduler;
    if let Some(results) = MemoryCollection::from_defs(errors).run(&gs) {
        for (p, err) in results {
            println!("Pass: {}, Error: {}", p, err);
        }
    }

}
