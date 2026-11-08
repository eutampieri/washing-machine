//! Washing machine monitor: reads the panel of a washing machine from an RTSP
//! camera and publishes its state to Home Assistant over MQTT.
//!
//! Usage:
//!   washing-machine-monitor                    run the monitor (configured via env)
//!   washing-machine-monitor recognise IMAGE…   print the reading of each image as JSON
//!   washing-machine-monitor evaluate DIR       accuracy on a labelled dataset

mod align;
mod capture;
mod config;
mod evaluate;
mod model;
mod mqtt;
mod recognise;

use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context, Result};
use log::{info, warn};

use crate::{
    capture::FrameSource,
    config::Config,
    mqtt::{Publisher, State},
    recognise::Recogniser,
};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(e) = run() {
        log::error!("{e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("run") => monitor(),
        Some("recognise") if args.len() > 1 => recognise_files(&args[1..]),
        Some("evaluate") if args.len() == 2 => evaluate::evaluate(Path::new(&args[1])),
        Some("-h" | "--help") => {
            println!("{}", USAGE);
            Ok(())
        }
        _ => bail!("invalid arguments\n{USAGE}"),
    }
}

const USAGE: &str = "usage: washing-machine-monitor [run | recognise IMAGE... | evaluate DIR]";

fn recognise_files(paths: &[String]) -> Result<()> {
    let recogniser = Recogniser::new()?;
    for path in paths {
        let output = match recogniser.recognise_file(Path::new(path)) {
            Ok(reading) => serde_json::json!({ "file": path, "reading": reading }),
            Err(e) => serde_json::json!({ "file": path, "error": format!("{e:#}") }),
        };
        println!("{output}");
    }
    Ok(())
}

fn monitor() -> Result<()> {
    let config = Config::from_env()?;
    let recogniser = Recogniser::new()?;
    let source = FrameSource::new(
        config.stream_url.clone(),
        config.capture_timeout,
        config.warmup_frames,
    );

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).context("signal handler")?;
    }

    info!(
        "monitoring {} → MQTT {}:{} (topic {})",
        capture::redact(&config.stream_url),
        config.mqtt.host,
        config.mqtt.port,
        config.ha.base_topic
    );
    let publisher = Publisher::start(config.mqtt, config.ha);

    let mut failures = 0u32;
    while !stop.load(Ordering::SeqCst) {
        let started = Instant::now();
        let interval = match source.grab().and_then(|frame| recogniser.recognise(&frame)) {
            Ok(reading) => {
                failures = 0;
                info!(
                    "running: {}, phase: {:?}, remaining: {:?} min (digits {:?})",
                    reading.running,
                    reading.phase,
                    reading.remaining_minutes,
                    reading.digits.map(|d| d.value)
                );
                publisher.set_available(true);
                publisher.publish_state(&State::new(&reading, SystemTime::now()));
                if reading.running {
                    config.interval_running
                } else {
                    config.interval_idle
                }
            }
            Err(e) => {
                failures += 1;
                warn!("reading failed ({failures} in a row): {e:#}");
                if failures >= config.failures_before_unavailable {
                    publisher.set_available(false);
                }
                config.interval_running
            }
        };
        sleep_until(started + interval, &stop);
    }

    info!("shutting down");
    publisher.shutdown();
    Ok(())
}

/// Sleeps until `deadline`, waking up early if `stop` is set.
fn sleep_until(deadline: Instant, stop: &AtomicBool) {
    while !stop.load(Ordering::SeqCst) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        std::thread::sleep((deadline - now).min(Duration::from_millis(250)));
    }
}
