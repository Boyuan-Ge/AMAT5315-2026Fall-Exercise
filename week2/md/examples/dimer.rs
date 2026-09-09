use md::{Euler, System, VelocityVerlet, simulate_steps};
use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dimer.png".to_string());
    let dt = 0.01;
    let euler = simulate_steps(System::dimer(), &Euler, dt, 500);
    let verlet_short = simulate_steps(System::dimer(), &VelocityVerlet, dt, 500);
    let verlet_long = simulate_steps(System::dimer(), &VelocityVerlet, dt, 5_000);

    let root = BitMapBackend::new(&output, (1_100, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    let mut short_chart = ChartBuilder::on(&panels[0])
        .caption("First 500 steps", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..5.0, -0.25..2.25)?;
    short_chart
        .configure_mesh()
        .x_desc("time t")
        .y_desc("(E(t) - E0) / |E0|")
        .draw()?;
    short_chart
        .draw_series(LineSeries::new(
            euler
                .relative_errors
                .iter()
                .enumerate()
                .map(|(i, error)| ((i + 1) as f64 * dt, *error)),
            RED.stroke_width(2),
        ))?
        .label("forward Euler")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], RED.stroke_width(2)));
    short_chart
        .draw_series(LineSeries::new(
            verlet_short
                .relative_errors
                .iter()
                .enumerate()
                .map(|(i, error)| ((i + 1) as f64 * dt, *error)),
            BLUE.stroke_width(2),
        ))?
        .label("velocity-Verlet")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BLUE.stroke_width(2)));
    short_chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    let mut long_chart = ChartBuilder::on(&panels[1])
        .caption("Velocity-Verlet: 5,000 steps", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..50.0, -1.0..1.0)?;
    long_chart
        .configure_mesh()
        .x_desc("time t")
        .y_desc("relative error x 1000")
        .draw()?;
    long_chart.draw_series(LineSeries::new(
        verlet_long
            .relative_errors
            .iter()
            .enumerate()
            .map(|(i, error)| ((i + 1) as f64 * dt, error * 1_000.0)),
        BLUE.stroke_width(2),
    ))?;

    root.present()?;
    println!(
        "Euler final error={:.6}; Verlet max error={:.6}; wrote {output}",
        euler.final_relative_error, verlet_long.max_relative_error
    );
    Ok(())
}
