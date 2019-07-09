use ndarray::arr2;
use num_complex::Complex64;

use crate::ComplexUnitary;

pub fn rot_x(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    arr2(&[
        [half_theta.cos(), negi * half_theta.sin()],
        [negi * half_theta.sin(), half_theta.cos()],
    ])
}

pub fn rot_y(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    arr2(&[
        [half_theta.cos(), -half_theta.sin()],
        [half_theta.sin(), half_theta.cos()],
    ])
}

pub fn rot_z(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    arr2(&[
        [(negi * half_theta).exp(), zero],
        [zero, (posi * half_theta).exp()],
    ])
}

pub fn re_rot_z(u: &mut ComplexUnitary, theta: f64) {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    u[[0, 0]] = (negi * half_theta).exp();
    u[[1, 1]] = (posi * half_theta).exp();
}

// ndarray doesn't have a Kronecker product implementation, so we implement it ourselves.
pub fn kron(a: &ComplexUnitary, b: &ComplexUnitary) -> ComplexUnitary {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = ComplexUnitary::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * b));
    }
    // TODO: Optimize? See simd, rayon, etc.
    // https://docs.rs/ndarray/0.12.1/ndarray/struct.ArrayBase.html#method.exact_chunks_mut
    // https://docs.rs/ndarray/0.12.1/ndarray/struct.Zip.html
    // https://docs.rs/ndarray/0.12.1/ndarray/doc/ndarray_for_numpy_users/index.html
    out
}