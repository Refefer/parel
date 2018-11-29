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
use std::fs;

use tange::deferred::{Deferred, tree_reduce};
use tange::scheduler::GreedyScheduler;
use tange::scheduler::LeveledScheduler;
use tange_collection::utils::read_text;
use tange_collection::interfaces::Stream;
use tange_collection::collection::memory::MemoryCollection;
use tange_collection::collection::disk::DiskCollection;
use svmloader::types::SparseData;
use svmloader::*;

use lr::{SGDOptions,Sparse,learn,test,ClassWeight,LearningRule,VWRule};

#[derive(Serialize,Deserialize,Clone)]
pub struct MySparse(pub usize, pub Vec<usize>, pub Vec<f32>);

fn load_data(path: &str, dims: usize, chunk_size: u64) -> DiskCollection<(Vec<(usize, f64)>, bool)> {
    let md = fs::metadata(path).expect("Error reading file!");
    let col = read_text(path, chunk_size).expect("File missing!");
    let sd = SparseData(dims);
    //let mut training_data = col.emit(move |s, emitter| {
    col.emit_to_disk("/tmp/par-lr".into(), move |s, emitter| {
        if let Some(row) = parse_line(&BinaryClassification, &sd, &s) {
            let mut v: Vec<(usize,f64)> = row.x.1.into_iter()
                .zip(row.x.2.into_iter().map(|x| x as f64)).collect();
            v.shrink_to_fit();
            emitter((v, row.y));
        }
    })
}

fn main() {
    env_logger::init();
    let path: String = args().skip(1).next().unwrap();
    let dims: usize = args().skip(2).next().unwrap().parse().unwrap();
    let chunk_size: u64 = args().skip(3).next().unwrap().parse().ok().unwrap();
    let test_file: Option<String> = args().skip(4).next();

    let training_data = load_data(&path, dims, chunk_size);
    println!("Number of parallel partitions: {}", training_data.n_partitions());

    let test_data = if let Some(test_path) = test_file {
        load_data(&test_path, dims, chunk_size)
    } else {
        training_data.clone()
    };
    
    // Create our initial weight vector
    let mut w = Deferred::lift(vec![0f64; dims], None);

    let mut errors = Vec::new();
    let mut best_w = Vec::new();
    for pass in 0..20 {
        //let alpha = 0.5 / ((pass + 2) as f64).ln();
        let alpha = 0.5;
        let mut opts = SGDOptions::new(1, 2)
            .learning_rule(LearningRule::Constant(alpha));

        //opts.set_class_weight(ClassWeight::Inverse);
        opts.set_class_weight(ClassWeight::None);
        opts.set_seed(pass + 1);
        let sgd = Arc::new(opts);

        // Train
        let out: Vec<_> = training_data.to_defs().iter().map(|d| {
            let opts = sgd.clone();
            d.join(&w, move |td, v| {
                let mut my_w = v.clone();
                let training: Vec<_> = td.stream().into_iter().collect();
                learn(&mut my_w, training, &opts);
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
        let cur_error = sum.apply(move |s| {
            let err = s / (n as f64);
            vec![(pass, err)]
        });
        best_w.push(cur_error.join(&w, |err, cur_w| {
            (err[0].1, cur_w.clone())
        }));
        errors.push(cur_error);
    }

    // Select best W
    let best = tree_reduce(&best_w, |left, right| {
        if left.0 < right.0 {
            left.clone()
        } else {
            right.clone()
        }
    }).unwrap();

    let output = MemoryCollection::from_defs(errors)
        .split(1).to_defs()[0].join(&best, |errors, b| {
        (errors.clone(), b.clone())
    });

    let gs = GreedyScheduler::new();
    if let Some((results, (best, w))) = output.run(&gs) {
        for (p, err) in results {
            println!("Pass: {}, Error: {}", p, err);
        }
        println!("Best: {}", best);
    }

}
