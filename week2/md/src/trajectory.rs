use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    pub force: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_to: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

pub struct TrajectoryWriter {
    frames: BufWriter<File>,
}

impl TrajectoryWriter {
    pub fn create(
        directory: &Path,
        metadata: &RunMetadata,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        fs::create_dir_all(directory)?;
        let metadata_file = File::create(directory.join("run.json"))?;
        serde_json::to_writer_pretty(BufWriter::new(metadata_file), metadata)?;
        let frames = BufWriter::new(File::create(directory.join("traj.jsonl"))?);
        Ok(Self { frames })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer(&mut self.frames, frame)?;
        self.frames.write_all(b"\n")?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.frames.flush()?;
        Ok(())
    }
}
