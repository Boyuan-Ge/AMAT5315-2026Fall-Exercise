use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result, bail};
use rustfft::{FftPlanner, num_complex::Complex};

use crate::artifacts::{RunMetadata, SeriesRow};

pub const ONSAGER_TC: f64 = 2.26919;

#[derive(Clone, Debug)]
pub struct GroupAnalysis {
    pub l: usize,
    pub t: f64,
    pub count: usize,
    pub mean_abs_m: f64,
    pub susceptibility: f64,
    pub naive_error: f64,
    pub blocked_error: f64,
    pub error_ratio: f64,
    pub tau_int: f64,
}

#[derive(Clone, Debug)]
pub struct RunAnalysis {
    pub metadata: RunMetadata,
    pub groups: Vec<GroupAnalysis>,
    pub peaks: Vec<(usize, f64)>,
    pub critical_temperature: Option<f64>,
}

pub fn susceptibility(l: usize, t: f64, values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        bail!("susceptibility requires at least one magnetization");
    }
    if !t.is_finite() || t <= 0.0 {
        bail!("susceptibility requires a positive finite temperature");
    }
    let n = values.len() as f64;
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / n;
    let mean_abs = values.iter().map(|value| value.abs()).sum::<f64>() / n;
    Ok((l * l) as f64 * (mean_square - mean_abs * mean_abs) / t)
}

pub fn naive_standard_error(values: &[f64]) -> Result<f64> {
    if values.len() < 2 {
        bail!("a standard error requires at least two values");
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sample_variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    Ok((sample_variance / values.len() as f64).sqrt())
}

pub fn blocked_standard_error(values: &[f64], requested_blocks: usize) -> Result<f64> {
    if requested_blocks < 2 {
        bail!("block analysis requires at least two blocks");
    }
    if values.len() < requested_blocks {
        bail!(
            "block analysis requested {requested_blocks} blocks from only {} values",
            values.len()
        );
    }
    let block_len = values.len() / requested_blocks;
    let block_means: Vec<_> = values
        .chunks_exact(block_len)
        .take(requested_blocks)
        .map(|block| block.iter().sum::<f64>() / block.len() as f64)
        .collect();
    naive_standard_error(&block_means)
}

pub fn autocorrelation_direct(values: &[f64]) -> Result<Vec<f64>> {
    let centered = centered_values(values)?;
    let variance = centered.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    if variance <= f64::EPSILON {
        return Ok(constant_autocorrelation(values.len()));
    }
    Ok((0..values.len())
        .map(|lag| {
            let covariance = centered
                .iter()
                .zip(centered.iter().skip(lag))
                .map(|(left, right)| left * right)
                .sum::<f64>()
                / (values.len() - lag) as f64;
            covariance / variance
        })
        .collect())
}

pub fn autocorrelation_fft(values: &[f64]) -> Result<Vec<f64>> {
    let centered = centered_values(values)?;
    let variance = centered.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    if variance <= f64::EPSILON {
        return Ok(constant_autocorrelation(values.len()));
    }

    let fft_len = (2 * values.len()).next_power_of_two();
    let mut buffer = vec![Complex::new(0.0, 0.0); fft_len];
    for (slot, value) in buffer.iter_mut().zip(centered) {
        slot.re = value;
    }

    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);
    forward.process(&mut buffer);
    for value in &mut buffer {
        *value = Complex::new(value.norm_sqr(), 0.0);
    }
    inverse.process(&mut buffer);

    Ok((0..values.len())
        .map(|lag| {
            let covariance = buffer[lag].re / fft_len as f64 / (values.len() - lag) as f64;
            covariance / variance
        })
        .collect())
}

pub fn integrated_autocorrelation_time(values: &[f64]) -> Result<f64> {
    let rho = autocorrelation_fft(values)?;
    let mut tau = 0.5;
    for (lag, &correlation) in rho.iter().enumerate().skip(1) {
        if !correlation.is_finite() || correlation <= 0.0 {
            break;
        }
        tau += correlation;
        if lag as f64 >= 6.0 * tau {
            break;
        }
    }
    Ok(tau.max(0.5))
}

pub fn parabolic_peak(points: &[(f64, f64)]) -> Result<f64> {
    if points.len() < 3 {
        bail!("parabolic peak fitting requires at least three points");
    }
    let center = points.iter().map(|(x, _)| x).sum::<f64>() / points.len() as f64;
    let mut normal = [[0.0; 4]; 3];
    for &(x, y) in points {
        let u = x - center;
        let row = [u * u, u, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                normal[i][j] += row[i] * row[j];
            }
            normal[i][3] += row[i] * y;
        }
    }
    let [a, b, _c] = solve_three_by_three(normal)?;
    if a >= 0.0 || a.abs() <= f64::EPSILON {
        bail!("fitted susceptibility curve does not have a downward peak");
    }
    Ok(center - b / (2.0 * a))
}

pub fn analyze_group(
    l: usize,
    t: f64,
    magnetizations: &[f64],
    blocks: usize,
) -> Result<GroupAnalysis> {
    if magnetizations.len() < blocks.max(2) {
        bail!("L={l} T={t} has too few rows for {blocks} blocks");
    }
    let absolute: Vec<_> = magnetizations.iter().map(|value| value.abs()).collect();
    let mean_abs_m = absolute.iter().sum::<f64>() / absolute.len() as f64;
    let naive_error = naive_standard_error(&absolute)?;
    let blocked_error = blocked_standard_error(&absolute, blocks)?;
    let error_ratio = if naive_error > 0.0 {
        blocked_error / naive_error
    } else {
        1.0
    };
    Ok(GroupAnalysis {
        l,
        t,
        count: magnetizations.len(),
        mean_abs_m,
        susceptibility: susceptibility(l, t, magnetizations)?,
        naive_error,
        blocked_error,
        error_ratio,
        tau_int: integrated_autocorrelation_time(&absolute)?,
    })
}

pub fn analyze_folder(folder: &Path, blocks: usize) -> Result<RunAnalysis> {
    let metadata_path = folder.join("run.json");
    let metadata: RunMetadata = serde_json::from_reader(
        File::open(&metadata_path)
            .with_context(|| format!("failed to open {}", metadata_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", metadata_path.display()))?;

    let series_path = folder.join("series.jsonl");
    let reader = BufReader::new(
        File::open(&series_path)
            .with_context(|| format!("failed to open {}", series_path.display()))?,
    );
    let mut groups = Vec::new();
    let mut current_key: Option<(usize, f64)> = None;
    let mut current_values = Vec::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} of {}",
                line_index + 1,
                series_path.display()
            )
        })?;
        let row: SeriesRow = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid JSON on line {} of {}",
                line_index + 1,
                series_path.display()
            )
        })?;
        let key = (row.l, row.t);
        if let Some((l, t)) = current_key
            && (l != row.l || t != row.t)
        {
            groups.push(analyze_group(l, t, &current_values, blocks)?);
            current_values.clear();
        }
        current_key = Some(key);
        current_values.push(row.m);
    }
    if let Some((l, t)) = current_key {
        groups.push(analyze_group(l, t, &current_values, blocks)?);
    }
    if groups.is_empty() {
        bail!("{} contains no measurement rows", series_path.display());
    }

    let peaks = fitted_peaks(&groups)?;
    let peak_map: BTreeMap<_, _> = peaks.iter().copied().collect();
    let critical_temperature = peak_map
        .get(&32)
        .zip(peak_map.get(&64))
        .map(|(peak_32, peak_64)| 2.0 * peak_64 - peak_32);

    Ok(RunAnalysis {
        metadata,
        groups,
        peaks,
        critical_temperature,
    })
}

pub fn render_analysis(analysis: &RunAnalysis) -> String {
    let mut output = String::new();
    for group in &analysis.groups {
        output.push_str(&format!(
            "L={} T={:.3} n={} mean_abs_m={:.6} naive_err={:.6} blocked_err={:.6} ratio={:.2} tau_int={:.2} chi={:.6}\n",
            group.l,
            group.t,
            group.count,
            group.mean_abs_m,
            group.naive_error,
            group.blocked_error,
            group.error_ratio,
            group.tau_int,
            group.susceptibility
        ));
    }
    for &(l, peak) in &analysis.peaks {
        output.push_str(&format!("peak L={l} T={peak:.4}\n"));
    }
    if let Some(critical) = analysis.critical_temperature {
        output.push_str(&format!(
            "T_c={critical:.4} exact={ONSAGER_TC:.5} relative_error={:.4}\n",
            (critical - ONSAGER_TC).abs() / ONSAGER_TC
        ));
    }
    output
}

fn fitted_peaks(groups: &[GroupAnalysis]) -> Result<Vec<(usize, f64)>> {
    let mut by_size: BTreeMap<usize, Vec<&GroupAnalysis>> = BTreeMap::new();
    for group in groups {
        by_size.entry(group.l).or_default().push(group);
    }
    let mut peaks = Vec::new();
    for (l, size_groups) in by_size {
        if size_groups.len() < 5 {
            continue;
        }
        let maximum = size_groups
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.susceptibility.total_cmp(&right.susceptibility))
            .map(|(index, _)| index)
            .expect("size group is known nonempty");
        if maximum < 2 || maximum + 2 >= size_groups.len() {
            continue;
        }
        let points: Vec<_> = size_groups[maximum - 2..=maximum + 2]
            .iter()
            .map(|group| (group.t, group.susceptibility))
            .collect();
        peaks.push((l, parabolic_peak(&points)?));
    }
    Ok(peaks)
}

fn centered_values(values: &[f64]) -> Result<Vec<f64>> {
    if values.len() < 2 {
        bail!("autocorrelation requires at least two values");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("autocorrelation values must be finite");
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Ok(values.iter().map(|value| value - mean).collect())
}

fn constant_autocorrelation(len: usize) -> Vec<f64> {
    let mut result = vec![0.0; len];
    result[0] = 1.0;
    result
}

fn solve_three_by_three(mut matrix: [[f64; 4]; 3]) -> Result<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|&left, &right| {
                matrix[left][column]
                    .abs()
                    .total_cmp(&matrix[right][column].abs())
            })
            .expect("the pivot range is nonempty");
        if matrix[pivot][column].abs() <= f64::EPSILON {
            bail!("singular quadratic fit");
        }
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in &mut matrix[column][column..] {
            *value /= divisor;
        }
        let pivot_row = matrix[column];
        for (row_index, row_values) in matrix.iter_mut().enumerate() {
            if row_index == column {
                continue;
            }
            let factor = row_values[column];
            for (value, pivot_value) in row_values[column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    Ok([matrix[0][3], matrix[1][3], matrix[2][3]])
}
