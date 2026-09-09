# Week 2: Molecular dynamics in Rust

## Reproduce the pair-field figure

From `week2/`:

```bash
cargo run --manifest-path md/Cargo.toml --release --example field -- field.png
```

The background color is the Lennard-Jones pair energy (red is repulsive,
blue is attractive). Arrows show the radial force direction.

## Reproduce the dimer integrator figure

From `week2/`:

```bash
cargo run --manifest-path md/Cargo.toml --release --example dimer -- dimer.png
```

Forward Euler evaluates both updates from the beginning of a step, so its
energy error grows sharply during close approaches. Velocity-Verlet uses two
half velocity updates around the position and force update; its energy error
remains bounded rather than drifting secularly.
