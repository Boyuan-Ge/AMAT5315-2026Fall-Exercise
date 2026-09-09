use plotters::prelude::*;
use std::error::Error;
use std::io;

const PARTICLE_COUNTS: [f64; 3] = [100.0, 400.0, 1_600.0];
const TIMED_STEPS: f64 = 600.0;

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() != 7 {
        return Err(invalid_input(
            "usage: scaling OUTPUT NAIVE_100 NAIVE_400 NAIVE_1600 CELLS_100 CELLS_400 CELLS_1600",
        )
        .into());
    }

    let output = &arguments[0];
    let seconds: Vec<f64> = arguments[1..]
        .iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| invalid_input(format!("invalid timing value: {value}")))
        })
        .collect::<Result<_, _>>()?;
    if seconds
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid_input("timings must be finite and positive").into());
    }

    let naive: Vec<(f64, f64)> = PARTICLE_COUNTS
        .iter()
        .zip(&seconds[..3])
        .map(|(n, elapsed)| (*n, elapsed / TIMED_STEPS))
        .collect();
    let cells: Vec<(f64, f64)> = PARTICLE_COUNTS
        .iter()
        .zip(&seconds[3..])
        .map(|(n, elapsed)| (*n, elapsed / TIMED_STEPS))
        .collect();

    let root = BitMapBackend::new(output, (960, 620)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Molecular-dynamics scaling", ("sans-serif", 32))
        .margin(22)
        .x_label_area_size(65)
        .y_label_area_size(85)
        .build_cartesian_2d(
            (80.0_f64..2_000.0_f64).log_scale(),
            (3e-5_f64..2e-2_f64).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("particle count N")
        .y_desc("elapsed seconds per step")
        .x_labels(5)
        .y_labels(8)
        .draw()?;

    chart
        .draw_series(LineSeries::new(naive.iter().copied(), RED.stroke_width(3)))?
        .label("naive all pairs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], RED.stroke_width(3)));
    chart.draw_series(
        naive
            .iter()
            .map(|point| Circle::new(*point, 6, RED.filled())),
    )?;

    chart
        .draw_series(LineSeries::new(cells.iter().copied(), BLUE.stroke_width(3)))?
        .label("cell list")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], BLUE.stroke_width(3)));
    chart.draw_series(
        cells
            .iter()
            .map(|point| TriangleMarker::new(*point, 7, BLUE.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("wrote {output} from six measured medians");
    Ok(())
}
