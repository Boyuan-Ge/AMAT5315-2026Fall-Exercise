/// Return the greeting printed by the `md` executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::{greeting, lj_energy, lj_force};

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_well_has_unit_depth() {
        let r_min = 2.0_f64.powf(1.0 / 6.0);
        assert!((lj_energy(r_min) + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn lennard_jones_force_matches_energy_derivative() {
        let h = 1.0e-5;
        for r in [0.95, 1.05, 2.0_f64.powf(1.0 / 6.0), 1.3, 2.0] {
            let numerical_force = -(lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h);
            let force = lj_force(r);
            let tolerance = 1.0e-6 * force.abs().max(1.0);
            assert!(
                (force - numerical_force).abs() < tolerance,
                "force mismatch at r={r}: analytic={force}, numerical={numerical_force}"
            );
        }
    }
}
