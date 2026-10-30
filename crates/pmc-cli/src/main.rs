// SPDX-License-Identifier: GPL-3.0-only

//! The `pmc` binary.
//!
//! One executable, no runtime dependencies, no Python. It trains, discovers,
//! fuses and exports, which is enough for a local machine with nothing
//! installed.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use pmc_cli::fuse::{fuse, FuseRequest};
use pmc_cli::state_memory::{fuse as fuse_state_memory, StateMemoryRequest};
use pmc_core::{Json, Rng};
use pmc_discovery::StateSpace;
use pmc_train::{export_checkpoint, train, TrainConfig};

fn flag(args: &[String], name: &str, fallback: f32) -> f32 {
    for pair in args.windows(2) {
        if pair[0] == name {
            return pair[1].parse().unwrap_or(fallback);
        }
    }
    fallback
}

fn option(args: &[String], name: &str) -> Option<String> {
    for pair in args.windows(2) {
        if pair[0] == name {
            return Some(pair[1].clone());
        }
    }
    None
}

fn usage() {
    println!(
        "pmc - PMC Framework trainer

Usage:
  pmc train    [--seed N] [--steps N] [--lr F] [--hidden N] [--keys N] [--out DIR]
  pmc discover [--seed N] [--points N] [--dim N] [--k N]
  pmc fuse     --base BASE.gguf --rows ROWS.bin --meta META.json --out OUT.gguf
               [--manifest MANIFEST.json]
  pmc state-memory --base BASE.gguf --weights STATE.pmc --out OUT.gguf
               [--manifest MANIFEST.json]
  pmc version
"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
        return ExitCode::from(1);
    }

    match args[1].as_str() {
        "train" => {
            let config = TrainConfig {
                seed: flag(&args, "--seed", 20261026.0) as u64,
                steps: flag(&args, "--steps", 200.0) as usize,
                learning_rate: flag(&args, "--lr", 0.05),
                hidden: flag(&args, "--hidden", 32.0) as usize,
                keys: flag(&args, "--keys", 16.0) as usize,
            };
            let (model, report) = train(&config);
            let metrics = report.to_json();
            println!("{}", metrics.to_string());

            if let Some(directory) = option(&args, "--out") {
                let path = PathBuf::from(directory);
                if let Err(error) = export_checkpoint(&path, &model, &config, &metrics) {
                    eprintln!("error: export failed: {}", error);
                    return ExitCode::from(1);
                }
                println!("{}", Json::obj(vec![("exported_to", Json::str(path.display().to_string()))]).to_string());
            }
            ExitCode::SUCCESS
        }
        "discover" => {
            let seed = flag(&args, "--seed", 20261026.0) as u64;
            let points = flag(&args, "--points", 500.0) as usize;
            let dim = flag(&args, "--dim", 8.0) as usize;
            let k = flag(&args, "--k", 4.0) as usize;

            let mut rng = Rng::new(seed);
            let mut space = StateSpace::new(dim, k);
            let mut novelty_total = 0.0f32;
            for _ in 0..points {
                let observation: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
                let (_, distance) = space.observe(&observation);
                novelty_total += distance;
            }
            let summary = Json::obj(vec![
                ("points", Json::num(points as f64)),
                (
                    "mean_novelty",
                    Json::num((novelty_total / points.max(1) as f32) as f64),
                ),
                ("space", space.to_json()),
            ]);
            println!("{}", summary.to_string());
            ExitCode::SUCCESS
        }
        "fuse" => {
            let (Some(base), Some(rows), Some(meta), Some(out)) = (
                option(&args, "--base"),
                option(&args, "--rows"),
                option(&args, "--meta"),
                option(&args, "--out"),
            ) else {
                eprintln!("error: fuse requires --base, --rows, --meta and --out");
                usage();
                return ExitCode::from(1);
            };
            let manifest = option(&args, "--manifest");
            if manifest.is_none() && args.iter().any(|arg| arg == "--manifest") {
                eprintln!("error: --manifest requires a path");
                return ExitCode::from(1);
            }
            let request = FuseRequest {
                base: PathBuf::from(base),
                rows: PathBuf::from(rows),
                meta: PathBuf::from(meta),
                out: PathBuf::from(out),
                manifest: manifest.map(PathBuf::from),
            };
            match fuse(&request) {
                Ok(report) => {
                    println!("{}", report.to_json().to_string());
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: fuse failed: {}", error);
                    ExitCode::from(1)
                }
            }
        }
        "state-memory" => {
            let (Some(base), Some(weights), Some(out)) = (
                option(&args, "--base"),
                option(&args, "--weights"),
                option(&args, "--out"),
            ) else {
                eprintln!("error: state-memory requires --base, --weights and --out");
                usage();
                return ExitCode::from(1);
            };
            let manifest = option(&args, "--manifest");
            if manifest.is_none() && args.iter().any(|arg| arg == "--manifest") {
                eprintln!("error: --manifest requires a path");
                usage();
                return ExitCode::from(1);
            }
            let request = StateMemoryRequest {
                base: PathBuf::from(base),
                weights: PathBuf::from(weights),
                out: PathBuf::from(out),
                manifest: manifest.map(PathBuf::from),
            };
            match fuse_state_memory(&request) {
                Ok(manifest) => {
                    println!("{}", manifest.to_string());
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: state-memory failed: {}", error);
                    ExitCode::from(1)
                }
            }
        }
        "version" => {
            println!("pmc {} (artifact format {})", env!("CARGO_PKG_VERSION"), pmc_core::ARTIFACT_FORMAT_VERSION);
            ExitCode::SUCCESS
        }
        _ => {
            usage();
            ExitCode::from(1)
        }
    }
}
