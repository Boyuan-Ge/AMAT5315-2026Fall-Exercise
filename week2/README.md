# Week 2: Molecular dynamics in Rust

## Reproduce the pair-field figure

From `week2/`:

```bash
cargo run --manifest-path md/Cargo.toml --release --example field -- field.png
```

The background color is the Lennard-Jones pair energy (red is repulsive,
blue is attractive). Arrows show the radial force direction.
