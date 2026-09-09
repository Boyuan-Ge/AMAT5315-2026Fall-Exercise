use md::{Frame, RunMetadata, TrajectoryWriter};
use serde_json::Value;
use std::fs;
use std::process::Command;
use tempfile::tempdir;

fn md_command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_md"))
}

#[test]
fn run_writes_readable_contract_files() {
    let temporary = tempdir().unwrap();
    let output = md_command()
        .args([
            "run",
            "--n",
            "4",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "10",
            "--steps",
            "20",
            "--sample-every",
            "10",
            "--seed",
            "2026",
            "--force",
            "naive",
            "--out",
        ])
        .arg(temporary.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata: Value =
        serde_json::from_str(&fs::read_to_string(temporary.path().join("run.json")).unwrap())
            .unwrap();
    assert_eq!(metadata["n"], 4);
    assert_eq!(metadata["rho"], 0.8);
    assert_eq!(metadata["dt"], 0.01);
    assert_eq!(metadata["temperature"], 0.5);
    assert_eq!(metadata["eq_steps"], 10);
    assert_eq!(metadata["steps"], 20);
    assert_eq!(metadata["sample_every"], 10);
    assert_eq!(metadata["seed"], 2026);
    assert_eq!(metadata["integrator"], "velocity-verlet");
    assert_eq!(metadata["force"], "naive");
    assert_eq!(metadata["box"].as_array().unwrap().len(), 2);

    let trajectory = fs::read_to_string(temporary.path().join("traj.jsonl")).unwrap();
    let frames: Vec<Value> = trajectory
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["step"], 10);
    assert_eq!(frames[0]["t"], 0.1);
    assert_eq!(frames[1]["step"], 20);
    assert_eq!(frames[1]["t"], 0.2);
    for frame in &frames {
        let positions = frame["pos"].as_array().unwrap();
        let box_size = metadata["box"].as_array().unwrap();
        assert_eq!(positions.len(), 4);
        assert_eq!(frame["vel"].as_array().unwrap().len(), 4);
        assert!(frame["E_pot"].is_number());
        assert!(frame["E_kin"].is_number());
        for position in positions {
            assert!(position[0].as_f64().unwrap() >= 0.0);
            assert!(position[0].as_f64().unwrap() < box_size[0].as_f64().unwrap());
            assert!(position[1].as_f64().unwrap() >= 0.0);
            assert!(position[1].as_f64().unwrap() < box_size[1].as_f64().unwrap());
        }
    }
}

#[test]
fn run_rejects_zero_sample_interval() {
    let temporary = tempdir().unwrap();
    let status = md_command()
        .args(["run", "--sample-every", "0", "--out"])
        .arg(temporary.path())
        .status()
        .unwrap();
    assert!(!status.success());
}

#[test]
fn run_rejects_unsupported_particle_grid() {
    let temporary = tempdir().unwrap();
    let status = md_command()
        .args(["run", "--n", "9", "--out"])
        .arg(temporary.path())
        .status()
        .unwrap();
    assert!(!status.success());
}

#[test]
fn default_run_passes_independent_physics_checks() {
    let temporary = tempdir().unwrap();
    let run = md_command()
        .args(["run", "--out"])
        .arg(temporary.path())
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let check = md_command()
        .arg("check")
        .arg(temporary.path())
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    let stdout = String::from_utf8_lossy(&check.stdout);
    assert!(stdout.contains("secular drift"));
    assert!(stdout.contains("T_speed"));
    assert!(stdout.contains("chi2/dof"));
    assert!(stdout.contains("PASS"));
}

#[test]
fn video_writes_an_mp4_from_saved_frames() {
    let temporary = tempdir().unwrap();
    let metadata = RunMetadata {
        n: 4,
        rho: 0.16,
        box_size: [5.0, 5.0],
        dt: 0.01,
        temperature: 0.5,
        eq_steps: 0,
        steps: 2,
        sample_every: 1,
        seed: 2026,
        integrator: "velocity-verlet".to_string(),
        force: "naive".to_string(),
        ramp_to: None,
    };
    let mut writer = TrajectoryWriter::create(temporary.path(), &metadata).unwrap();
    for (step, offset) in [(1, 0.0), (2, 0.1)] {
        writer
            .write_frame(&Frame {
                step,
                t: step as f64 * 0.01,
                pos: vec![
                    [1.0 + offset, 1.0],
                    [2.0 + offset, 1.0],
                    [1.0 + offset, 2.0],
                    [2.0 + offset, 2.0],
                ],
                vel: vec![[0.1, 0.0]; 4],
                e_pot: 0.0,
                e_kin: 0.02,
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let movie = temporary.path().join("smoke.mp4");

    let output = md_command()
        .arg("video")
        .arg(temporary.path())
        .arg("--out")
        .arg(&movie)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(movie).unwrap();
    assert!(bytes.len() > 32);
    assert_eq!(&bytes[4..8], b"ftyp");
}
