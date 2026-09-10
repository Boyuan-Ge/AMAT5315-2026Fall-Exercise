use anyhow::{Result, bail};
use rand::Rng;

use crate::lattice::Lattice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WolffSweepStats {
    pub touched: usize,
    pub clusters: usize,
}

pub struct WolffUpdater {
    marks: Vec<u32>,
    generation: u32,
    stack: Vec<usize>,
    cluster: Vec<usize>,
}

impl WolffUpdater {
    pub fn new(lattice_len: usize) -> Self {
        Self {
            marks: vec![0; lattice_len],
            generation: 0,
            stack: Vec::with_capacity(lattice_len),
            cluster: Vec::with_capacity(lattice_len),
        }
    }

    pub fn cluster_flip<R: Rng + ?Sized>(
        &mut self,
        lattice: &mut Lattice,
        t: f64,
        rng: &mut R,
    ) -> Result<usize> {
        validate_temperature(t)?;
        if self.marks.len() != lattice.len() {
            bail!("Wolff workspace size does not match the lattice");
        }
        self.start_generation();
        self.stack.clear();
        self.cluster.clear();

        let seed = rng.random_range(0..lattice.len());
        let cluster_spin = lattice.spin(seed);
        self.marks[seed] = self.generation;
        self.stack.push(seed);
        self.cluster.push(seed);
        let add_probability = 1.0 - (-2.0 / t).exp();

        while let Some(site) = self.stack.pop() {
            for neighbour in lattice.neighbours(site) {
                if self.marks[neighbour] == self.generation
                    || lattice.spin(neighbour) != cluster_spin
                {
                    continue;
                }
                if rng.random::<f64>() < add_probability {
                    self.marks[neighbour] = self.generation;
                    self.stack.push(neighbour);
                    self.cluster.push(neighbour);
                }
            }
        }

        for &site in &self.cluster {
            lattice.flip(site);
        }
        Ok(self.cluster.len())
    }

    pub fn sweep<R: Rng + ?Sized>(
        &mut self,
        lattice: &mut Lattice,
        t: f64,
        rng: &mut R,
    ) -> Result<WolffSweepStats> {
        let target = lattice.len();
        let mut touched = 0;
        let mut clusters = 0;
        while touched < target {
            touched += self.cluster_flip(lattice, t, rng)?;
            clusters += 1;
        }
        Ok(WolffSweepStats { touched, clusters })
    }

    fn start_generation(&mut self) {
        if self.generation == u32::MAX {
            self.marks.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }
}

pub fn wolff_cluster_flip<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    t: f64,
    rng: &mut R,
) -> Result<usize> {
    WolffUpdater::new(lattice.len()).cluster_flip(lattice, t, rng)
}

pub fn wolff_sweep<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    t: f64,
    rng: &mut R,
) -> Result<WolffSweepStats> {
    WolffUpdater::new(lattice.len()).sweep(lattice, t, rng)
}

fn validate_temperature(t: f64) -> Result<()> {
    if !t.is_finite() || t <= 0.0 {
        bail!("temperature must be a positive finite number");
    }
    Ok(())
}
