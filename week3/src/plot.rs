use std::{collections::BTreeMap, fmt::Debug, path::PathBuf};

use anyhow::{Result, anyhow};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use crate::analysis::{ONSAGER_TC, RunAnalysis, analyze_folder};

const IMAGE_SIZE: (u32, u32) = (1200, 800);

pub fn onsager_magnetization(t: f64) -> f64 {
    let exact_tc = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
    if !t.is_finite() || t <= 0.0 || t >= exact_tc {
        return 0.0;
    }
    (1.0 - (2.0 / t).sinh().powi(-4)).powf(0.125)
}

pub fn plot_saved_run(folder: &std::path::Path, blocks: usize) -> Result<Vec<PathBuf>> {
    let analysis = analyze_folder(folder, blocks)?;
    let magnetization = folder.join("magnetization.png");
    let susceptibility = folder.join("susceptibility.png");
    let tau = folder.join("tau.png");
    plot_magnetization(&analysis, &magnetization)?;
    plot_susceptibility(&analysis, &susceptibility)?;
    plot_tau(&analysis, &tau)?;
    Ok(vec![magnetization, susceptibility, tau])
}

pub fn plot_magnetization(analysis: &RunAnalysis, output: &std::path::Path) -> Result<()> {
    let (x_min, x_max) = temperature_bounds(analysis)?;
    let root = BitMapBackend::new(output, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Mean absolute magnetization", ("sans-serif", 42))
        .margin(28)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, 0.0..1.05)
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("mean |m|")
        .draw()
        .map_err(plot_error)?;

    let exact = (0..=400).map(|index| {
        let t = x_min + (x_max - x_min) * index as f64 / 400.0;
        (t, onsager_magnetization(t))
    });
    chart
        .draw_series(LineSeries::new(exact, RED.stroke_width(3)))
        .map_err(plot_error)?
        .label("Onsager, infinite lattice")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], RED.stroke_width(3)));

    draw_group_lines(&mut chart, analysis, |group| group.mean_abs_m, "measured")?;
    draw_tc_marker(&mut chart, 0.0, 1.05)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)
}

pub fn plot_susceptibility(analysis: &RunAnalysis, output: &std::path::Path) -> Result<()> {
    let (x_min, x_max) = temperature_bounds(analysis)?;
    let y_max = analysis
        .groups
        .iter()
        .map(|group| group.susceptibility)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1.15;
    let root = BitMapBackend::new(output, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Susceptibility", ("sans-serif", 42))
        .margin(28)
        .x_label_area_size(55)
        .y_label_area_size(75)
        .build_cartesian_2d(x_min..x_max, 0.0..y_max)
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("chi(T)")
        .draw()
        .map_err(plot_error)?;
    draw_group_lines(&mut chart, analysis, |group| group.susceptibility, "L")?;
    draw_tc_marker(&mut chart, 0.0, y_max)?;
    for &(l, peak) in &analysis.peaks {
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(peak, 0.0), (peak, y_max)],
                BLACK.mix(0.35).stroke_width(1),
            )))
            .map_err(plot_error)?
            .label(format!("L={l} peak {peak:.4}"));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)
}

pub fn plot_tau(analysis: &RunAnalysis, output: &std::path::Path) -> Result<()> {
    let (x_min, x_max) = temperature_bounds(analysis)?;
    let y_max = analysis
        .groups
        .iter()
        .map(|group| group.tau_int)
        .fold(0.5_f64, f64::max)
        .max(1.0)
        * 1.5;
    let root = BitMapBackend::new(output, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Integrated autocorrelation time", ("sans-serif", 42))
        .margin(28)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(x_min..x_max, (0.4..y_max).log_scale())
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("tau_int (sweeps)")
        .draw()
        .map_err(plot_error)?;
    draw_group_lines(&mut chart, analysis, |group| group.tau_int, "L")?;
    draw_tc_marker(&mut chart, 0.4, y_max)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)
}

pub fn plot_tau_comparison(
    metropolis: &RunAnalysis,
    wolff: &RunAnalysis,
    l: usize,
    output: &std::path::Path,
) -> Result<()> {
    let metropolis_points: Vec<_> = metropolis
        .groups
        .iter()
        .filter(|group| group.l == l)
        .map(|group| (group.t, group.tau_int))
        .collect();
    let wolff_points: Vec<_> = wolff
        .groups
        .iter()
        .filter(|group| group.l == l)
        .map(|group| (group.t, group.tau_int))
        .collect();
    if metropolis_points.is_empty() || wolff_points.is_empty() {
        return Err(anyhow!("both runs must contain L={l}"));
    }
    let x_min = wolff_points
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let x_max = wolff_points
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let metropolis_points: Vec<_> = metropolis_points
        .into_iter()
        .filter(|(t, _)| *t >= x_min && *t <= x_max)
        .collect();
    let y_max = metropolis_points
        .iter()
        .chain(&wolff_points)
        .map(|point| point.1)
        .fold(1.0_f64, f64::max)
        * 1.5;

    let root = BitMapBackend::new(output, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Critical slowing down at L = {l}"),
            ("sans-serif", 42),
        )
        .margin(28)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(x_min..x_max, (0.4..y_max).log_scale())
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("tau_int (sweeps)")
        .draw()
        .map_err(plot_error)?;
    chart
        .draw_series(LineSeries::new(
            metropolis_points.clone(),
            RED.stroke_width(3),
        ))
        .map_err(plot_error)?
        .label("single-flip Metropolis")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], RED.stroke_width(3)));
    chart
        .draw_series(
            metropolis_points
                .into_iter()
                .map(|point| Circle::new(point, 4, RED.filled())),
        )
        .map_err(plot_error)?;
    chart
        .draw_series(LineSeries::new(wolff_points.clone(), BLUE.stroke_width(3)))
        .map_err(plot_error)?
        .label("Wolff clusters")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BLUE.stroke_width(3)));
    chart
        .draw_series(
            wolff_points
                .into_iter()
                .map(|point| Circle::new(point, 4, BLUE.filled())),
        )
        .map_err(plot_error)?;
    draw_tc_marker(&mut chart, 0.4, y_max)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)
}

fn draw_group_lines<Y: Ranged<ValueType = f64>>(
    chart: &mut ChartContext<'_, BitMapBackend<'_>, Cartesian2d<RangedCoordf64, Y>>,
    analysis: &RunAnalysis,
    value: impl Fn(&crate::analysis::GroupAnalysis) -> f64,
    label_prefix: &str,
) -> Result<()> {
    let mut by_size = BTreeMap::<usize, Vec<(f64, f64)>>::new();
    for group in &analysis.groups {
        by_size
            .entry(group.l)
            .or_default()
            .push((group.t, value(group)));
    }
    let colors = [BLUE, GREEN, MAGENTA, CYAN];
    for (index, (l, points)) in by_size.into_iter().enumerate() {
        let color = colors[index % colors.len()];
        chart
            .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))
            .map_err(plot_error)?
            .label(format!("{label_prefix} = {l}"))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(2))
            });
        chart
            .draw_series(
                points
                    .into_iter()
                    .map(|point| Circle::new(point, 4, color.filled())),
            )
            .map_err(plot_error)?;
    }
    Ok(())
}

fn draw_tc_marker<Y: Ranged<ValueType = f64>>(
    chart: &mut ChartContext<'_, BitMapBackend<'_>, Cartesian2d<RangedCoordf64, Y>>,
    y_min: f64,
    y_max: f64,
) -> Result<()> {
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(ONSAGER_TC, y_min), (ONSAGER_TC, y_max)],
            BLACK.mix(0.55).stroke_width(2),
        )))
        .map_err(plot_error)?
        .label(format!("exact T_c = {ONSAGER_TC:.5}"));
    Ok(())
}

fn temperature_bounds(analysis: &RunAnalysis) -> Result<(f64, f64)> {
    let x_min = analysis
        .groups
        .iter()
        .map(|group| group.t)
        .min_by(f64::total_cmp)
        .ok_or_else(|| anyhow!("cannot plot an empty analysis"))?;
    let x_max = analysis
        .groups
        .iter()
        .map(|group| group.t)
        .max_by(f64::total_cmp)
        .ok_or_else(|| anyhow!("cannot plot an empty analysis"))?;
    if x_min == x_max {
        return Ok((x_min - 0.05, x_max + 0.05));
    }
    Ok((x_min, x_max))
}

fn plot_error(error: impl Debug) -> anyhow::Error {
    anyhow!("plotting failed: {error:?}")
}
