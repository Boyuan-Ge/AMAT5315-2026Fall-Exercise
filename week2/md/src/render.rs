use crate::{radial_distribution, read_trajectory};
use plotters::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn render_video(directory: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (metadata, frames) = read_trajectory(directory)?;
    if frames.is_empty() {
        return Err("cannot render an empty trajectory".into());
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let temporary = tempfile::tempdir()?;
    for (index, frame) in frames.iter().enumerate() {
        let filename = temporary.path().join(format!("frame-{index:06}.png"));
        let root = BitMapBackend::new(&filename, (900, 450)).into_drawing_area();
        root.fill(&WHITE)?;
        let panels = root.split_evenly((1, 2));

        let mut particles = ChartBuilder::on(&panels[0])
            .caption(
                format!("Particles: step {}, t={:.2}", frame.step, frame.t),
                ("sans-serif", 22),
            )
            .margin(12)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..metadata.box_size[0], 0.0..metadata.box_size[1])?;
        particles.configure_mesh().x_desc("x").y_desc("y").draw()?;
        particles.draw_series(
            frame
                .pos
                .iter()
                .map(|position| Circle::new((position[0], position[1]), 4, BLUE.filled())),
        )?;

        let start = index.saturating_sub(19);
        let distribution =
            radial_distribution(&frames[start..=index], metadata.box_size, metadata.rho, 60);
        let max_radius = 0.5 * metadata.box_size[0].min(metadata.box_size[1]);
        let mut structure = ChartBuilder::on(&panels[1])
            .caption("Pair structure g(r)", ("sans-serif", 22))
            .margin(12)
            .x_label_area_size(35)
            .y_label_area_size(42)
            .build_cartesian_2d(0.0..max_radius, 0.0..5.0)?;
        structure
            .configure_mesh()
            .x_desc("pair distance r")
            .y_desc("g(r)")
            .draw()?;
        structure.draw_series(LineSeries::new(distribution, RED.stroke_width(2)))?;
        structure.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 1.0), (max_radius, 1.0)],
            BLACK.mix(0.35),
        )))?;
        root.present()?;
    }

    let pattern = temporary.path().join("frame-%06d.png");
    let encoded = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "20", "-i"])
        .arg(pattern)
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
        ])
        .arg(output)
        .output()?;
    if !encoded.status.success() {
        return Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&encoded.stderr)
        )
        .into());
    }
    Ok(())
}
