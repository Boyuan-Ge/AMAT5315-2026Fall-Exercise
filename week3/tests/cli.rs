use std::process::Command;

#[test]
fn cli_is_one_command_and_runs_the_contract() {
    let help = Command::new(env!("CARGO_BIN_EXE_ising"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--update <UPDATE>"));
    assert!(!help.contains("Commands:"));

    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args([
            "--update",
            "metropolis",
            "--l",
            "4",
            "--t-from",
            "1.5",
            "--t-to",
            "1.6",
            "--t-step",
            "0.1",
            "--discard",
            "2",
            "--measure",
            "4",
            "--every",
            "2",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 3);
    assert!(stdout.starts_with("T\tmean_abs_M\tacceptance_rate\n"));
}
