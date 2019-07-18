pub use ndarray::prelude::*;
pub use ndarray::{stack, Data};
use num_complex::Complex64;

use crate::ComplexUnitary;

pub trait ComplexFloat {
    fn cj(&self) -> Self;
}

impl ComplexFloat for Complex64 {
    #[inline]
    fn cj(&self) -> Self {
        self.conj()
    }
}

#[inline]
pub fn rot_x(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    arr2(&[
        [half_theta.cos(), negi * half_theta.sin()],
        [negi * half_theta.sin(), half_theta.cos()],
    ])
}

#[inline]
pub fn rot_y(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    arr2(&[
        [half_theta.cos(), -half_theta.sin()],
        [half_theta.sin(), half_theta.cos()],
    ])
}

#[inline]
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

#[inline]
pub fn re_rot_z(u: &mut ComplexUnitary, theta: f64) {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    u[[0, 0]] = (negi * half_theta).exp();
    u[[1, 1]] = (posi * half_theta).exp();
}

#[inline]
fn matrix_distance_squared(a: &ComplexUnitary, b: &ComplexUnitary) -> f64 {
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    1f64 - a.dot(&conj_t(&b)).sum().norm() / a.shape()[0] as f64
}

#[inline]
pub fn matrix_distance(a: &ComplexUnitary, b: &ComplexUnitary) -> f64 {
    matrix_distance_squared(a, b).sqrt()
}

#[inline]
pub fn conj_t<T: ComplexFloat + Clone, D: Data<Elem = T>>(a: &ArrayBase<D, Ix2>) -> Array<T, Ix2> {
    a.t().mapv(|x| x.cj())
}

#[inline]
/// ndarray doesn't have a Kronecker product implementation, so we implement it ourselves.
/// We also optimize for the common case where we are kronecker'ing some matrix with the identity.
pub fn kron(a: &ComplexUnitary, b: &ComplexUnitary) -> ComplexUnitary {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;

    let mut out = ComplexUnitary::zeros((dimout, dimout));

    for (ref mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * b));
    }
    out
}
