use anyhow::{Result, bail};
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    l: usize,
    spins: Vec<i8>,
}

impl Lattice {
    pub fn all_up(l: usize) -> Result<Self> {
        Self::validate_side(l)?;
        let len = l
            .checked_mul(l)
            .ok_or_else(|| anyhow::anyhow!("L is too large"))?;
        Ok(Self {
            l,
            spins: vec![1; len],
        })
    }

    pub fn from_spins(l: usize, spins: Vec<i8>) -> Result<Self> {
        Self::validate_side(l)?;
        let expected = l
            .checked_mul(l)
            .ok_or_else(|| anyhow::anyhow!("L is too large"))?;
        if spins.len() != expected {
            bail!(
                "expected {expected} spins for L={l}, received {}",
                spins.len()
            );
        }
        if let Some((index, spin)) = spins
            .iter()
            .copied()
            .enumerate()
            .find(|(_, spin)| !matches!(spin, -1 | 1))
        {
            bail!("spin at index {index} is {spin}; spins must be -1 or +1");
        }
        Ok(Self { l, spins })
    }

    pub fn random<R: Rng + ?Sized>(l: usize, rng: &mut R) -> Result<Self> {
        Self::validate_side(l)?;
        let len = l
            .checked_mul(l)
            .ok_or_else(|| anyhow::anyhow!("L is too large"))?;
        let spins = (0..len)
            .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
            .collect();
        Ok(Self { l, spins })
    }

    pub fn side(&self) -> usize {
        self.l
    }

    pub fn len(&self) -> usize {
        self.spins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spins.is_empty()
    }

    pub fn spin(&self, index: usize) -> i8 {
        self.spins[index]
    }

    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    pub fn flip(&mut self, index: usize) {
        self.spins[index] = -self.spins[index];
    }

    pub fn neighbour_sum(&self, index: usize) -> i32 {
        self.neighbours(index)
            .into_iter()
            .map(|neighbour| i32::from(self.spins[neighbour]))
            .sum()
    }

    pub fn neighbours(&self, index: usize) -> [usize; 4] {
        let row = index / self.l;
        let column = index % self.l;
        let up = (row + self.l - 1) % self.l;
        let down = (row + 1) % self.l;
        let left = (column + self.l - 1) % self.l;
        let right = (column + 1) % self.l;
        [
            up * self.l + column,
            down * self.l + column,
            row * self.l + left,
            row * self.l + right,
        ]
    }

    pub fn delta_energy(&self, index: usize) -> i32 {
        2 * i32::from(self.spins[index]) * self.neighbour_sum(index)
    }

    pub fn total_energy(&self) -> i64 {
        let mut energy = 0_i64;
        for row in 0..self.l {
            for column in 0..self.l {
                let index = row * self.l + column;
                let spin = i64::from(self.spins[index]);
                let right = i64::from(self.spins[row * self.l + (column + 1) % self.l]);
                let down = i64::from(self.spins[((row + 1) % self.l) * self.l + column]);
                energy -= spin * (right + down);
            }
        }
        energy
    }

    pub fn energy_per_site(&self) -> f64 {
        self.total_energy() as f64 / self.len() as f64
    }

    pub fn magnetization(&self) -> f64 {
        let total: i64 = self.spins.iter().map(|&spin| i64::from(spin)).sum();
        total as f64 / self.len() as f64
    }

    pub fn spin_bits(&self) -> String {
        self.spins
            .iter()
            .map(|&spin| if spin == 1 { '1' } else { '0' })
            .collect()
    }

    pub fn character_picture(&self) -> String {
        let mut picture = String::with_capacity(self.len() + self.l.saturating_sub(1));
        for row in 0..self.l {
            if row > 0 {
                picture.push('\n');
            }
            for column in 0..self.l {
                picture.push(if self.spins[row * self.l + column] == 1 {
                    '+'
                } else {
                    '-'
                });
            }
        }
        picture
    }

    fn validate_side(l: usize) -> Result<()> {
        if l < 2 {
            bail!("L must be at least 2");
        }
        Ok(())
    }
}
