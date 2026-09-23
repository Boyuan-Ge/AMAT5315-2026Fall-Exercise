use std::f64::consts::PI;
use week4_fluid::{Fft2, derivative};

fn main() {
    println!("derivative\tfinite difference n=32\tfinite difference n=64\tratio\tFourier n=32");
    let mut errors: [[f64; 3]; 4] = [[0.0; 3]; 4];
    for (column, n) in [(0, 32usize), (1, 64usize)] {
        let h = 2.0 * PI / n as f64;
        let fft = Fft2::new(n);
        let mut g = vec![0.0; n * n];
        let mut exact = vec![vec![0.0; n * n]; 4];
        for y in 0..n {
            for x in 0..n {
                let xx = x as f64 * h;
                let yy = y as f64 * h;
                let i = y * n + x;
                g[i] = (3.0 * xx).sin() * (2.0 * yy).cos();
                exact[0][i] = 3.0 * (3.0 * xx).cos() * (2.0 * yy).cos();
                exact[1][i] = -9.0 * g[i];
                exact[2][i] = -6.0 * (3.0 * xx).cos() * (2.0 * yy).sin();
                exact[3][i] = -13.0 * g[i];
            }
        }
        let modes = [
            derivative(&fft, &g, 1, 0),
            derivative(&fft, &g, 2, 0),
            derivative(&fft, &g, 1, 1),
            {
                let xx = derivative(&fft, &g, 2, 0);
                let yy = derivative(&fft, &g, 0, 2);
                xx.iter().zip(yy).map(|(a, b)| a + b).collect()
            },
        ];
        for y in 0..n {
            for x in 0..n {
                let i = y * n + x;
                let xp = y * n + (x + 1) % n;
                let xm = y * n + (x + n - 1) % n;
                let yp = ((y + 1) % n) * n + x;
                let ym = ((y + n - 1) % n) * n + x;
                let xpy = ((y + 1) % n) * n + (x + 1) % n;
                let xmy = ((y + 1) % n) * n + (x + n - 1) % n;
                let xpym = ((y + n - 1) % n) * n + (x + 1) % n;
                let xmym = ((y + n - 1) % n) * n + (x + n - 1) % n;
                let fd = [
                    (g[xp] - g[xm]) / (2.0 * h),
                    (g[xp] - 2.0 * g[i] + g[xm]) / (h * h),
                    (g[xpy] - g[xmy] - g[xpym] + g[xmym]) / (4.0 * h * h),
                    (g[xp] + g[xm] + g[yp] + g[ym] - 4.0 * g[i]) / (h * h),
                ];
                for j in 0..4 {
                    errors[j][column] = errors[j][column].max((fd[j] - exact[j][i]).abs());
                    if n == 32 {
                        errors[j][2] = errors[j][2].max((modes[j][i] - exact[j][i]).abs());
                    }
                }
            }
        }
    }
    for (j, label) in ["dx", "dxx", "dxdy", "laplacian"].iter().enumerate() {
        println!(
            "{label}\t{:.8}\t{:.8}\t{:.3}\t{:.3e}",
            errors[j][0],
            errors[j][1],
            errors[j][0] / errors[j][1],
            errors[j][2]
        );
    }
}
