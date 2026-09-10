use anyhow::{Result, bail};
use rand::Rng;

use crate::lattice::Lattice;

#[derive(Clone, Debug)]
pub struct AcceptanceTable([f64; 5]);

impl AcceptanceTable {
    pub fn new(t: f64) -> Result<Self> {
        if !t.is_finite() || t <= 0.0 {
            bail!("temperature must be a positive finite number");
        }
        Ok(Self([1.0, 1.0, 1.0, (-4.0 / t).exp(), (-8.0 / t).exp()]))
    }

    pub fn probability(&self, delta_energy: i32) -> f64 {
        let index = match delta_energy {
            -8 => 0,
            -4 => 1,
            0 => 2,
            4 => 3,
            8 => 4,
            _ => panic!("invalid Ising flip energy {delta_energy}"),
        };
        self.0[index]
    }
}

pub fn metropolis_sweep<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    table: &AcceptanceTable,
    rng: &mut R,
) -> usize {
    let proposals = lattice.len();
    let mut accepted = 0;
    for _ in 0..proposals {
        let site = rng.random_range(0..proposals);
        let probability = table.probability(lattice.delta_energy(site));
        if probability >= 1.0 || rng.random::<f64>() < probability {
            lattice.flip(site);
            accepted += 1;
        }
    }
    accepted
}
