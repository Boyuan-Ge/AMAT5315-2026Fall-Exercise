/// Lennard-Jones pair potential in reduced units (epsilon = sigma = 1).
pub fn lj_energy(r: f64) -> f64 {
    let inv_r6 = r.powi(-6);
    4.0 * (inv_r6 * inv_r6 - inv_r6)
}

/// Radial Lennard-Jones force in reduced units.
pub fn lj_force(r: f64) -> f64 {
    let inv_r6 = r.powi(-6);
    24.0 / r * (2.0 * inv_r6 * inv_r6 - inv_r6)
}

pub fn shifted_energy(r: f64, rc: f64) -> f64 {
    if r < rc {
        lj_energy(r) - lj_energy(rc)
    } else {
        0.0
    }
}

pub fn shifted_force(r: f64, rc: f64) -> f64 {
    if r < rc { lj_force(r) } else { 0.0 }
}
