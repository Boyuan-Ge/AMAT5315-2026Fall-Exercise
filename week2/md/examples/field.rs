use md::{lj_energy, lj_force};
use plotters::prelude::*;
use std::error::Error;

fn potential_color(value: f64) -> RGBColor {
    let value = value.clamp(-1.0, 1.0);
    if value >= 0.0 {
        let fade = (255.0 * (1.0 - value)) as u8;
        RGBColor(220, fade, fade)
    } else {
        let fade = (255.0 * (1.0 + value)) as u8;
        RGBColor(fade, fade, 220)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "field.png".to_string());
    let root = BitMapBackend::new(&output, (900, 760)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Lennard-Jones pair energy and force", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(45)
        .build_cartesian_2d(-3.0..3.0, -3.0..3.0)?;

    let cells = 120;
    let width = 6.0 / cells as f64;
    chart.draw_series((0..cells).flat_map(|ix| {
        (0..cells).map(move |iy| {
            let x = -3.0 + (ix as f64 + 0.5) * width;
            let y = -3.0 + (iy as f64 + 0.5) * width;
            let r = x.hypot(y);
            let energy = if r < 0.7 { 1.0 } else { lj_energy(r) };
            Rectangle::new(
                [
                    (x - width / 2.0, y - width / 2.0),
                    (x + width / 2.0, y + width / 2.0),
                ],
                potential_color(energy).filled(),
            )
        })
    }))?;

    let arrow_scale = 0.22;
    for radius in [0.9, 1.05, 1.25, 1.55, 2.0, 2.45] {
        for k in 0..16 {
            let angle = std::f64::consts::TAU * k as f64 / 16.0;
            let (sin, cos) = angle.sin_cos();
            let force = lj_force(radius).clamp(-4.0, 4.0);
            let start = (radius * cos, radius * sin);
            let end = (
                start.0 + arrow_scale * force.signum() * cos,
                start.1 + arrow_scale * force.signum() * sin,
            );
            chart.draw_series(std::iter::once(PathElement::new(
                vec![start, end],
                BLACK.stroke_width(2),
            )))?;
            chart.draw_series(std::iter::once(Circle::new(end, 3, BLACK.filled())))?;
        }
    }

    chart
        .configure_mesh()
        .x_desc("x / sigma")
        .y_desc("y / sigma")
        .axis_desc_style(("sans-serif", 20))
        .draw()?;
    root.present()?;
    println!("wrote {output}");
    Ok(())
}
