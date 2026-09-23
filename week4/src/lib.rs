use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

pub trait Integrator {
    fn step(
        &self,
        state: &[Complex64],
        dt: f64,
        rate: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    ) -> Vec<Complex64>;
}

pub struct Euler;
pub struct Midpoint;
pub struct Rk4;
pub struct EqualWeightRk4;

fn shifted(state: &[Complex64], rate: &[Complex64], dt: f64) -> Vec<Complex64> {
    state.iter().zip(rate).map(|(y, k)| *y + *k * dt).collect()
}

impl Integrator for Euler {
    fn step(
        &self,
        state: &[Complex64],
        dt: f64,
        rate: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    ) -> Vec<Complex64> {
        shifted(state, &rate(state), dt)
    }
}

impl Integrator for Midpoint {
    fn step(
        &self,
        state: &[Complex64],
        dt: f64,
        rate: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    ) -> Vec<Complex64> {
        let k1 = rate(state);
        let k2 = rate(&shifted(state, &k1, dt / 2.0));
        shifted(state, &k2, dt)
    }
}

impl Integrator for Rk4 {
    fn step(
        &self,
        state: &[Complex64],
        dt: f64,
        rate: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    ) -> Vec<Complex64> {
        let k1 = rate(state);
        let k2 = rate(&shifted(state, &k1, dt / 2.0));
        let k3 = rate(&shifted(state, &k2, dt / 2.0));
        let k4 = rate(&shifted(state, &k3, dt));
        state
            .iter()
            .enumerate()
            .map(|(i, y)| *y + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
            .collect()
    }
}

impl Integrator for EqualWeightRk4 {
    fn step(
        &self,
        state: &[Complex64],
        dt: f64,
        rate: &dyn Fn(&[Complex64]) -> Vec<Complex64>,
    ) -> Vec<Complex64> {
        let k1 = rate(state);
        let k2 = rate(&shifted(state, &k1, dt / 2.0));
        let k3 = rate(&shifted(state, &k2, dt / 2.0));
        let k4 = rate(&shifted(state, &k3, dt));
        state
            .iter()
            .enumerate()
            .map(|(i, y)| *y + dt * (k1[i] + k2[i] + k3[i] + k4[i]) / 4.0)
            .collect()
    }
}

pub fn integrator(method: &str) -> Option<Box<dyn Integrator>> {
    match method {
        "euler" => Some(Box::new(Euler)),
        "rk2" => Some(Box::new(Midpoint)),
        "rk4" => Some(Box::new(Rk4)),
        "rk4-equal" => Some(Box::new(EqualWeightRk4)),
        _ => None,
    }
}

pub fn wave_number(i: usize, n: usize) -> i32 {
    if i < (n + 1) / 2 {
        i as i32
    } else {
        i as i32 - n as i32
    }
}

pub fn index(kx: i32, ky: i32, n: usize) -> usize {
    ky.rem_euclid(n as i32) as usize * n + kx.rem_euclid(n as i32) as usize
}

pub struct Fft2 {
    pub n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Fft2 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    fn transform(&self, data: &mut [Complex64], inverse: bool) {
        let fft = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for row in data.chunks_exact_mut(self.n) {
            fft.process(row);
        }
        let mut col = vec![Complex64::new(0.0, 0.0); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                col[y] = data[y * self.n + x];
            }
            fft.process(&mut col);
            for y in 0..self.n {
                data[y * self.n + x] = col[y];
            }
        }
        if inverse {
            let scale = 1.0 / (self.n * self.n) as f64;
            for value in data {
                *value *= scale;
            }
        }
    }

    pub fn forward_real(&self, data: &[f64]) -> Vec<Complex64> {
        let mut result: Vec<_> = data.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        self.transform(&mut result, false);
        result
    }

    pub fn inverse_real(&self, data: &[Complex64]) -> Vec<f64> {
        let mut result = data.to_vec();
        self.transform(&mut result, true);
        result.iter().map(|z| z.re).collect()
    }
}

pub fn line_rate(state: &[Complex64], n: usize, c: f64, nu: f64, centred: bool) -> Vec<Complex64> {
    let dx = 2.0 * PI / n as f64;
    state
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let k = wave_number(i, n) as f64;
            let (first, second) = if centred {
                (
                    (k * dx).sin() / dx,
                    -4.0 * (k * dx / 2.0).sin().powi(2) / dx.powi(2),
                )
            } else {
                (if n % 2 == 0 && i == n / 2 { 0.0 } else { k }, -k * k)
            };
            *value * Complex64::new(nu * second, -c * first)
        })
        .collect()
}

pub fn derivative(fft: &Fft2, field: &[f64], ax: usize, ay: usize) -> Vec<f64> {
    let mut modes = fft.forward_real(field);
    for y in 0..fft.n {
        for x in 0..fft.n {
            let kx = wave_number(x, fft.n) as f64;
            let ky = wave_number(y, fft.n) as f64;
            let odd_nyquist = fft.n % 2 == 0
                && ((ax % 2 == 1 && x == fft.n / 2) || (ay % 2 == 1 && y == fft.n / 2));
            modes[y * fft.n + x] *= if odd_nyquist {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(0.0, kx).powu(ax as u32) * Complex64::new(0.0, ky).powu(ay as u32)
            };
        }
    }
    fft.inverse_real(&modes)
}

pub struct Fluid {
    pub fft: Fft2,
    pub nu: f64,
}

impl Fluid {
    pub fn new(n: usize, nu: f64) -> Self {
        Self {
            fft: Fft2::new(n),
            nu,
        }
    }

    pub fn mask(&self, field: &mut [Complex64]) {
        let n = self.fft.n;
        let cutoff = (n / 3) as i32;
        for y in 0..n {
            for x in 0..n {
                if wave_number(x, n).abs() > cutoff || wave_number(y, n).abs() > cutoff {
                    field[y * n + x] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }

    pub fn vorticity_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<Complex64> {
        let uh = self.fft.forward_real(u);
        let vh = self.fft.forward_real(v);
        let n = self.fft.n;
        let mut wh = vec![Complex64::new(0.0, 0.0); n * n];
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                wh[i] = Complex64::new(0.0, wave_number(x, n) as f64) * vh[i]
                    - Complex64::new(0.0, wave_number(y, n) as f64) * uh[i];
            }
        }
        self.mask(&mut wh);
        wh
    }

    pub fn velocity(&self, w: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.fft.n;
        let mut uh = vec![Complex64::new(0.0, 0.0); n * n];
        let mut vh = uh.clone();
        for y in 0..n {
            for x in 0..n {
                let kx = wave_number(x, n) as f64;
                let ky = wave_number(y, n) as f64;
                let k2 = kx * kx + ky * ky;
                if k2 > 0.0 {
                    let i = y * n + x;
                    let psi = w[i] / k2;
                    uh[i] = Complex64::new(0.0, ky) * psi;
                    vh[i] = Complex64::new(0.0, -kx) * psi;
                }
            }
        }
        (self.fft.inverse_real(&uh), self.fft.inverse_real(&vh))
    }

    pub fn rate(&self, w: &[Complex64]) -> Vec<Complex64> {
        let n = self.fft.n;
        let mut uh = vec![Complex64::new(0.0, 0.0); n * n];
        let mut vh = uh.clone();
        let mut wxh = uh.clone();
        let mut wyh = uh.clone();
        for y in 0..n {
            for x in 0..n {
                let kx = wave_number(x, n) as f64;
                let ky = wave_number(y, n) as f64;
                let k2 = kx * kx + ky * ky;
                let i = y * n + x;
                if k2 > 0.0 {
                    let psi = w[i] / k2;
                    uh[i] = Complex64::new(0.0, ky) * psi;
                    vh[i] = Complex64::new(0.0, -kx) * psi;
                }
                wxh[i] = Complex64::new(0.0, kx) * w[i];
                wyh[i] = Complex64::new(0.0, ky) * w[i];
            }
        }
        let u = self.fft.inverse_real(&uh);
        let v = self.fft.inverse_real(&vh);
        let wx = self.fft.inverse_real(&wxh);
        let wy = self.fft.inverse_real(&wyh);
        let product: Vec<f64> = (0..n * n).map(|i| -u[i] * wx[i] - v[i] * wy[i]).collect();
        let mut result = self.fft.forward_real(&product);
        self.mask(&mut result);
        for y in 0..n {
            for x in 0..n {
                let kx = wave_number(x, n) as f64;
                let ky = wave_number(y, n) as f64;
                let i = y * n + x;
                result[i] -= self.nu * (kx * kx + ky * ky) * w[i];
            }
        }
        result
    }

    pub fn energy_enstrophy(&self, w: &[Complex64]) -> (f64, f64) {
        let n2 = (self.fft.n * self.fft.n) as f64;
        let mut e = 0.0;
        let mut z = 0.0;
        for y in 0..self.fft.n {
            for x in 0..self.fft.n {
                let i = y * self.fft.n + x;
                let kx = wave_number(x, self.fft.n) as f64;
                let ky = wave_number(y, self.fft.n) as f64;
                let k2 = kx * kx + ky * ky;
                z += w[i].norm_sqr();
                if k2 > 0.0 {
                    e += w[i].norm_sqr() / k2;
                }
            }
        }
        (e / (2.0 * n2 * n2), z / (2.0 * n2 * n2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn steppers_match_single_wave() {
        let lambda = Complex64::new(-0.05 * 9.0, -3.0);
        let rate = |y: &[Complex64]| vec![lambda * y[0]];
        let exact = (lambda * 0.01).exp();
        for (method, tolerance) in [("euler", 1e-3), ("rk2", 1e-5), ("rk4", 1e-9)] {
            let got = integrator(method)
                .unwrap()
                .step(&[Complex64::new(1.0, 0.0)], 0.01, &rate)[0];
            assert!((got - exact).norm() < tolerance, "{method}");
        }
    }
    #[test]
    fn nyquist_does_not_advect() {
        let y = vec![Complex64::new(1.0, 0.0); 64];
        let rate = line_rate(&y, 64, 1.0, 0.05, false);
        assert_eq!(rate[32].im, 0.0);
        assert!((rate[32].re + 51.2).abs() < 1e-12);
    }
    #[test]
    fn spectral_derivative_of_represented_wave() {
        let n = 32;
        let fft = Fft2::new(n);
        let field: Vec<f64> = (0..n * n)
            .map(|i| {
                (3.0 * 2.0 * PI * (i % n) as f64 / n as f64).sin()
                    * (2.0 * 2.0 * PI * (i / n) as f64 / n as f64).cos()
            })
            .collect();
        let dx = derivative(&fft, &field, 1, 0);
        let dxx = derivative(&fft, &field, 2, 0);
        assert!((dx[0] - 3.0).abs() < 1e-10);
        assert!(
            dxx.iter()
                .zip(&field)
                .all(|(a, b)| (a + 9.0 * b).abs() < 1e-10)
        );
    }
    #[test]
    fn line_integrators_advance_a_single_exact_wave() {
        let n = 64;
        let mut wave = vec![Complex64::new(0.0, 0.0); n];
        wave[3] = Complex64::new(1.0, 0.0);
        let expected = (Complex64::new(-0.05 * 9.0, -3.0)).exp();
        for (method, limit) in [("euler", 0.04), ("rk2", 0.0005), ("rk4", 2e-8)] {
            let stepper = integrator(method).unwrap();
            let mut state = wave.clone();
            for _ in 0..100 {
                state = stepper.step(&state, 0.01, &|modes| line_rate(modes, n, 1.0, 0.05, false));
            }
            assert!((state[3] - expected).norm() < limit, "{method}");
            assert!(
                state
                    .iter()
                    .enumerate()
                    .all(|(i, z)| i == 3 || z.norm() == 0.0)
            );
        }
    }
    #[test]
    fn taylor_green_rate_is_pure_diffusion_and_high_modes_are_masked() {
        let n = 32;
        let mut u = Vec::new();
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let xx = 2.0 * PI * x as f64 / n as f64;
                let yy = 2.0 * PI * y as f64 / n as f64;
                u.push(xx.cos() * yy.sin());
                v.push(-xx.sin() * yy.cos());
            }
        }
        let model = Fluid::new(n, 0.1);
        let mut w = model.vorticity_from_velocity(&u, &v);
        w[index(12, 12, n)] = Complex64::new(1.0, 0.0);
        model.mask(&mut w);
        assert_eq!(w[index(12, 12, n)], Complex64::new(0.0, 0.0));
        let rate = model.rate(&w);
        let error = rate
            .iter()
            .zip(&w)
            .map(|(r, z)| (*r + 0.2 * (*z)).norm())
            .fold(0.0, f64::max);
        assert!(error < 1e-10, "Taylor-Green nonlinear rate error {error}");
    }
}
