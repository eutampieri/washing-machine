//! Accuracy of the recogniser on a labelled dataset.
//!
//! File names follow the labeler's convention: `<10-digit timestamp>.jpg` for
//! a machine that is off, `<timestamp><phase letter><minutes>.jpg` for one
//! that is running (e.g. `1734194469R20.jpg`).
//!
//! Compared with the prototype's evaluation loop, every metric is computed
//! against the *labels* (not the predictions), failed recognitions count as
//! errors instead of being skipped or crashing, and the three metrics are
//! reported separately rather than averaged into one number.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use log::warn;

use crate::recognise::{Phase, Recogniser};

#[derive(Debug, PartialEq)]
pub struct Label {
    pub phase: Option<Phase>,
    pub minutes: Option<u16>,
}

impl Label {
    pub fn running(&self) -> bool {
        self.phase.is_some()
    }

    /// Parses a file stem; `None` if it does not follow the convention.
    pub fn parse(stem: &str) -> Option<Self> {
        let timestamp = stem.get(..10)?;
        if !timestamp.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let rest = &stem[10..];
        let mut chars = rest.chars();
        match chars.next() {
            None => Some(Self {
                phase: None,
                minutes: None,
            }),
            Some(code) => Some(Self {
                phase: Some(Phase::from_code(code)?),
                minutes: Some(chars.as_str().parse().ok()?),
            }),
        }
    }
}

#[derive(Default)]
struct Metrics {
    images: usize,
    failures: usize,
    /// [label][prediction], false = off, true = on.
    on_off: [[usize; 2]; 2],
    labelled_on: usize,
    phase_correct: usize,
    minutes_read: usize,
    minutes_exact: usize,
    minutes_abs_error: u64,
}

pub fn evaluate(dir: &Path) -> Result<()> {
    let recogniser = Recogniser::new()?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png"))
        })
        .collect();
    files.sort();

    let mut m = Metrics::default();
    for path in &files {
        let Some(label) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(Label::parse)
        else {
            warn!("skipping {}: not a labelled file name", path.display());
            continue;
        };
        m.images += 1;
        m.labelled_on += usize::from(label.running());
        let reading = match recogniser.recognise_file(path) {
            Ok(r) => r,
            Err(e) => {
                warn!("{}: {e:#}", path.display());
                m.failures += 1;
                // Counted as a wrong on/off answer.
                m.on_off[usize::from(label.running())][usize::from(!label.running())] += 1;
                continue;
            }
        };
        m.on_off[usize::from(label.running())][usize::from(reading.running)] += 1;
        if !label.running() {
            continue;
        }
        if reading.phase == label.phase {
            m.phase_correct += 1;
        }
        if let (Some(expected), Some(read)) = (label.minutes, reading.remaining_minutes) {
            m.minutes_read += 1;
            m.minutes_exact += usize::from(expected == read);
            m.minutes_abs_error += u64::from(expected.abs_diff(read));
        }
    }

    let pct = |n: usize, d: usize| {
        if d == 0 {
            f64::NAN
        } else {
            100.0 * n as f64 / d as f64
        }
    };
    let [[tn, fp], [fn_, tp]] = m.on_off;
    println!(
        "images:              {} ({} failed to process)",
        m.images, m.failures
    );
    println!(
        "on/off accuracy:     {:.1}% (TP {tp}, TN {tn}, FP {fp}, FN {fn_})",
        pct(tp + tn, m.images)
    );
    println!(
        "phase accuracy:      {:.1}% of {} running images",
        pct(m.phase_correct, m.labelled_on),
        m.labelled_on
    );
    println!(
        "minutes read:        {:.1}% of running images",
        pct(m.minutes_read, m.labelled_on)
    );
    println!(
        "minutes exact:       {:.1}% of those read",
        pct(m.minutes_exact, m.minutes_read)
    );
    println!(
        "minutes mean error:  {:.2} min",
        if m.minutes_read == 0 {
            f64::NAN
        } else {
            m.minutes_abs_error as f64 / m.minutes_read as f64
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_labels() {
        assert_eq!(
            Label::parse("1733738109"),
            Some(Label {
                phase: None,
                minutes: None
            })
        );
        assert_eq!(
            Label::parse("1734194469R20"),
            Some(Label {
                phase: Some(Phase::Rinse),
                minutes: Some(20)
            })
        );
        assert_eq!(Label::parse(".DS_Store"), None);
        assert_eq!(Label::parse("short"), None);
        assert_eq!(Label::parse("1734194469X20"), None);
        assert_eq!(Label::parse("1734194469R"), None);
        assert_eq!(Label::parse("17341944àà9R2"), None);
    }
}
