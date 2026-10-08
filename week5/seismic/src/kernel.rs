#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

unsafe extern "C" { fn abort() -> !; }
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 { x * x * x }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adj) = cube_reverse(x, 1.0);
    unsafe { *out = y; *out.add(1) = dy; *out.add(2) = adj; }
}

#[autodiff_forward(step_forward, Dual, Dual, Dual, Const, Const, Const, Const, Const, Const, Dual)]
#[autodiff_reverse(step_reverse, Duplicated, Duplicated, Duplicated, Const, Const, Const, Const, Const, Const, Duplicated)]
fn step(prev: &[f64], cur: &[f64], speed: &[f64], damping: &[f64], source: &[f64],
        dt: f64, dx: f64, nx: usize, nz: usize, out: &mut [f64]) {
    let dt2 = dt * dt;
    let dx2 = dx * dx;
    for z in 1..nz-1 {
        for x in 1..nx-1 {
            let i = z * nx + x;
            let lap = (cur[i-1] + cur[i+1] + cur[i-nx] + cur[i+nx] - 4.0*cur[i]) / dx2;
            let damp = damping[i] * dt;
            out[i] = (2.0*cur[i] - (1.0-damp)*prev[i]
                      + dt2*(speed[i]*speed[i]*lap + source[i])) / (1.0+damp);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step(prev: *const f64, cur: *const f64, speed: *const f64,
        damping: *const f64, source: *const f64, dt: f64, dx: f64, nx: usize, nz: usize,
        out: *mut f64) {
    let n = nx*nz;
    unsafe { step(core::slice::from_raw_parts(prev,n), core::slice::from_raw_parts(cur,n),
        core::slice::from_raw_parts(speed,n), core::slice::from_raw_parts(damping,n),
        core::slice::from_raw_parts(source,n),dt,dx,nx,nz,
        core::slice::from_raw_parts_mut(out,n)); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_jvp(prev: *const f64, dprev: *const f64,
        cur: *const f64, dcur: *const f64, speed: *const f64, dspeed: *const f64,
        damping: *const f64, source: *const f64, dt: f64, dx: f64, nx: usize, nz: usize,
        out: *mut f64, dout: *mut f64) {
    let n = nx*nz;
    unsafe { step_forward(
        core::slice::from_raw_parts(prev,n), core::slice::from_raw_parts(dprev,n),
        core::slice::from_raw_parts(cur,n), core::slice::from_raw_parts(dcur,n),
        core::slice::from_raw_parts(speed,n), core::slice::from_raw_parts(dspeed,n),
        core::slice::from_raw_parts(damping,n), core::slice::from_raw_parts(source,n),
        dt,dx,nx,nz,core::slice::from_raw_parts_mut(out,n),
        core::slice::from_raw_parts_mut(dout,n)); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_vjp(prev: *const f64, dprev: *mut f64,
        cur: *const f64, dcur: *mut f64, speed: *const f64, dspeed: *mut f64,
        damping: *const f64, source: *const f64, dt: f64, dx: f64, nx: usize, nz: usize,
        out: *mut f64, dout: *mut f64) {
    let n = nx*nz;
    unsafe { step_reverse(
        core::slice::from_raw_parts(prev,n), core::slice::from_raw_parts_mut(dprev,n),
        core::slice::from_raw_parts(cur,n), core::slice::from_raw_parts_mut(dcur,n),
        core::slice::from_raw_parts(speed,n), core::slice::from_raw_parts_mut(dspeed,n),
        core::slice::from_raw_parts(damping,n), core::slice::from_raw_parts(source,n),
        dt,dx,nx,nz,core::slice::from_raw_parts_mut(out,n),
        core::slice::from_raw_parts_mut(dout,n)); }
}
