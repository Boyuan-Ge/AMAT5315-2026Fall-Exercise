use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "T")]
    pub t: f64,
    pub sweep: u64,
    pub m: f64,
    pub spins: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Algorithm {
    Metropolis,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub sizes: Vec<usize>,
    pub t_grid: Vec<f64>,
    pub eq_sweeps: usize,
    pub meas_sweeps: usize,
    pub meas_sweeps_critical: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub algorithm: Algorithm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeriesRow {
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "T")]
    pub t: f64,
    pub sweep: usize,
    #[serde(rename = "M")]
    pub m: f64,
    #[serde(rename = "E")]
    pub e: f64,
}

pub fn write_run_metadata(path: &Path, metadata: &RunMetadata) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create run metadata at {}", path.display()))?;
    serde_json::to_writer_pretty(file, metadata)
        .with_context(|| format!("failed to write run metadata at {}", path.display()))
}

pub struct SeriesWriter {
    writer: BufWriter<File>,
}

impl SeriesWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create series at {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write(&mut self, row: &SeriesRow) -> Result<()> {
        writeln!(
            self.writer,
            "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}",
            row.l, row.t, row.sweep, row.m, row.e
        )
        .context("failed to write measurement row")
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer
            .flush()
            .context("failed to flush measurement rows")
    }
}
