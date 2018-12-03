#[macro_use]
extern crate clap;

#[macro_use]
extern crate log;
extern crate tange;
extern crate tange_collection;
extern crate svmloader;
extern crate env_logger;
extern crate rand;

extern crate parlr;

use std::sync::Arc;
use std::fs;
use std::io::{self};
use std::io::prelude::*;

use clap::{Arg, App, ArgMatches};
use tange::deferred::{Deferred, tree_reduce};
use tange::scheduler::GreedyScheduler;
use tange_collection::utils::read_text;
use tange_collection::interfaces::Stream;
use tange_collection::collection::memory::MemoryCollection;
use tange_collection::collection::disk::DiskCollection;
use svmloader::types::SparseData;
use svmloader::*;
use rand::{XorShiftRng,SeedableRng};
use rand::seq::SliceRandom;

use parlr::lr::{SGDOptions,learn,test,ClassWeight,LearningRule,dot,Sigmoid};

fn parse<'a>() -> ArgMatches<'a> {
  App::new("par-lr")
    .version("0.0.1")
    .author("Andrew S. <refefer@gmail.com>")
    .about("Parallel Logistic Regression")
    .arg(Arg::with_name("train")
        .index(1)
        .required(true)
        .help("Training File"))
    .arg(Arg::with_name("partition-size")
        .long("partition-size")
        .short("p")
        .takes_value(true)
        .help("Partition size to use for dataset"))
    .arg(Arg::with_name("parts-per-update")
        .long("parts-per-update")
        .short("u")
        .takes_value(true)
        .help("Number of partitions to process before updating.  If omitted, uses all"))
    .arg(Arg::with_name("dims")
        .long("dims")
        .short("d")
        .takes_value(true)
        .required(true)
        .help("Number of dimensions of the data"))
    .arg(Arg::with_name("valid")
        .long("validation")
        .short("v")
        .takes_value(true)
        .help("Validation file"))
    .arg(Arg::with_name("valid-partition-size")
        .long("validation-partition-size")
        .takes_value(true)
        .help("Size of each partition for validation.  If omitted, uses same as train"))
    .arg(Arg::with_name("test")
        .long("test")
        .short("t")
        .takes_value(true)
        .multiple(true)
        .min_values(2)
        .max_values(2)
        .help("Requires two arguments: Test dataset and directory to output scores"))
    .arg(Arg::with_name("learning_rate")
        .long("learning-rate")
        .short("lr")
        .takes_value(true)
        .help("Set the learning rate"))
    .arg(Arg::with_name("passes")
        .long("passes")
        .short("N")
        .takes_value(true)
        .help("Number of passes to perform"))
    .arg(Arg::with_name("batch_size")
        .long("batch-size")
        .short("b")
        .takes_value(true)
        .help("Batch size"))
    .arg(Arg::with_name("train-iters")
        .long("train-iters")
        .short("i")
        .takes_value(true)
        .help("Number of epochs for an inner pass"))
    .arg(Arg::with_name("balanced-weighting")
        .long("balanced")
        .short("b")
        .help("Scales loss inverse to the number of classes"))
    .arg(Arg::with_name("logloss")
        .long("logloss")
        .help("Selects based on logloss rather than accuracy"))
    .arg(Arg::with_name("lr-decay")
        .long("decay")
        .takes_value(true)
        .help("Learning rate decay for each pass through the data"))
    .arg(Arg::with_name("hard-sigmoid")
        .long("hard-sigmoid")
        .short("s")
        .help("Uses a sigmoid approximation for speed up"))
    .arg(Arg::with_name("model-out-file")
        .long("model-out")
        .takes_value(true)
        .help("Write out model to the given file"))
    .arg(Arg::with_name("model-in-file")
        .long("model-in")
        .takes_value(true)
        .help("If provided, reads a model from disk as starting point"))
    .arg(Arg::with_name("seed")
        .long("set-seed")
        .takes_value(true)
        .help("Random seed to use"))

    .get_matches()
}

fn load_data(path: &str, dims: usize, part_size: u64) -> DiskCollection<(Vec<(usize, f64)>, bool)> {
    let col = read_text(path, part_size).expect("File missing!");
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

fn write_model(path: &str, w: &[f64]) -> io::Result<()> {
    let f = fs::File::create(path)?;
    let mut bw = io::BufWriter::new(f);
    let nl = [10u8];
    for (i, v) in w.iter().enumerate() {
        if *v != 0. {
            bw.write(format!("{}:{:.5}", i, v).as_bytes())?;
            if i + 1 != w.len() {
                bw.write(&nl)?;
            }
        }
    }
    Ok(())
}

fn read_model(path: &str, w: &mut [f64]) -> io::Result<()> {
    let f = fs::File::open(path)?;
    let bw = io::BufReader::new(f);
    for line in bw.lines().map(|l| l.unwrap()) {
        let mut it = line.split(":");
        let idx: usize = it.next()
            .expect("Bad model format!")
            .parse()
            .expect("Index is not a number!");

        let v: f64 = it.next()
            .expect("Bad model format!")
            .parse()
            .expect("value is not a number!");

        w[idx] = v;
    }
    Ok(())
}

fn average_ws(ws: &[Deferred<(u32, Vec<f64>)>]) -> Deferred<Vec<f64>> {
    // Average Ws together
    let avg_w = tree_reduce(&ws, |(c1, left), (c2, right)| {
        let c3 = c1 + c2;
        let mut out: Vec<_> = left.clone();
        for i in 0..left.len() {
            out[i] += right[i];
        }
        (c3, out)
    }).unwrap();

    avg_w.apply(|(count, v)| {
        let mut w = v.clone();
        let c = *count as f64;
        if c > 1. {
            for i in 0..w.len() {
                w[i] /= c;
            }
        }
        w
    })
}

fn main() {
    env_logger::init();
    let args = parse();
    let path = args.value_of("train").unwrap();
    let dims = value_t!(args, "dims", usize)
        .expect("Required number of dims missing");
    let part_size = value_t!(args, "partition-size", u64)
        .unwrap_or(64_000_000);
    assert!(part_size > 0, "`partition-size` needs to be greater than 0!");
    let ppu = value_t!(args, "parts-per-update", usize).ok();
    assert!(ppu.is_none() || ppu > Some(0), 
            "`parts-per-update` needs to be greater than 0!");

    let lr = value_t!(args, "learning_rate", f64)
        .unwrap_or(10.);
    let passes = value_t!(args, "passes", u64)
        .unwrap_or(10);
    let validation = value_t!(args, "valid", String).ok();
    let batch_size = value_t!(args, "batch_size", usize).unwrap_or(4);
    let iterations = value_t!(args, "train-iters", u32).unwrap_or(1);
    let logloss    = args.is_present("logloss");
    let decay      = value_t!(args, "lr-decay", f64).unwrap_or(.6);
    let hard_sigmoid = args.is_present("hard-sigmoid");
    let model_out  = value_t!(args, "model-out-file", String).ok();
    let model_in  = value_t!(args, "model-in-file", String).ok();
    let seed = value_t!(args, "seed", u64).unwrap_or(2019);

    /*
    Load training and validation datasets
    */
    let training_data = load_data(&path, dims, part_size);
    info!("Number of training partitions: {}", training_data.n_partitions());

    let valid_data = if let Some(valid_path) = validation {
        let v_part_size = value_t!(args, "validation-partition-size", u64)
            .unwrap_or(part_size);
        load_data(&valid_path, dims, v_part_size)
    } else {
        training_data.clone()
    };
    info!("Number of valid partitions: {}", valid_data.n_partitions());

    // How many partitions are we using?
    let part_batches = ppu.unwrap_or(training_data.n_partitions());
    let n_parts = ((training_data.to_defs().len() as f64) 
        / (part_batches as f64)).ceil() as usize;

    info!("Global updates per pass: {}", n_parts);
    
    // Create our initial weight vector
    let mut v = vec![0f64; dims];
    if let Some(model_in_path) = model_in {
        read_model(&model_in_path, &mut v)
            .expect("Unable to read model!");
    }
    let mut w = Deferred::lift(v, None);

    let mut errors = Vec::new();
    let mut best_w = Vec::new();
    let mut rng = XorShiftRng::seed_from_u64(seed);
    for pass in 0..passes {
        // Build our options
        let alpha = lr * decay.powi(pass as i32);
        let mut opts = SGDOptions::new(iterations, batch_size)
            .learning_rule(LearningRule::Constant(alpha));

        if hard_sigmoid {
            opts.set_sigmoid(Sigmoid::Hard);
        }

        if args.is_present("balanced-weighting") {
            opts.set_class_weight(ClassWeight::Inverse);
        } else {
            opts.set_class_weight(ClassWeight::None);
        }

        opts.set_seed(pass + seed);
        let sgd = Arc::new(opts);

        // *
        // Train
        // *
        // Shuffle our training partitions
        // Group our training data into sub-batches
        let mut train_defs = training_data.to_defs().clone();
        train_defs.as_mut_slice().shuffle(&mut rng);

        for batch in train_defs.chunks(part_batches) {
            let out: Vec<_> = batch.iter().map(|d| {
                let opts = sgd.clone();
                d.join(&w, move |td, v| {
                    let mut my_w = v.clone();
                    let training: Vec<_> = td.stream().into_iter().collect();
                    learn(&mut my_w, &training, &opts);
                    (1u32, my_w)
                })
            }).collect();

            w = average_ws(&out);
        }

        // Test
        let misclass: Vec<_> = valid_data.to_defs().iter().map(|d| {
            d.join(&w, move |td, w| {
                let training: Vec<_> = td.stream().into_iter().collect();
                let res = test(w, &training);
                if logloss { res.1 } else { res.0 }
            })
        }).collect();

        let sum = tree_reduce(&misclass, |left, right| left + right).unwrap();
        let n = misclass.len();
        let cur_error = sum.apply(move |s| {
            let err = s / (n as f64);
            println!("Pass: {}, Error: {}", pass, err);
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

        // Write out model
        if let Some(model_path) = model_out {
            write_model(&model_path, &w)
                .expect("Unable to write model to disk!");
        }

        // predict files into directory
        if let Some(tester) = values_t!(args, "test", String).ok() {
            let test_path = &tester[0];
            let output_dir = tester[1].to_owned();
            let w = Deferred::lift(w, None);
            let td = load_data(test_path, dims, part_size);
            let dfs = td.to_defs().iter().map(|test| {
                test.join(&w, |s, v| -> Vec<String> {
                    s.stream().into_iter().map(|(x, y)| {
                        let yi = if y { 1.0 } else { -1.0 };
                        format!("{},{}", yi, dot(&x, v))
                    }).collect()
                })
            }).collect();

            MemoryCollection::from_defs(dfs)
                .sink(&output_dir)
                .run(&gs);
        }
    }
}
