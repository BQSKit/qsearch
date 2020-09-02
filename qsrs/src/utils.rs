use ndarray::Array2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

use crate::{i, r};

use std::f64::consts::{E, PI};

pub fn rot_x(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    SquareMatrix::from_vec(
        vec![
            half_theta.cos(),
            negi * half_theta.sin(),
            negi * half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_y(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    SquareMatrix::from_vec(
        vec![
            half_theta.cos(),
            -half_theta.sin(),
            half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_z(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    SquareMatrix::from_vec(
        vec![
            (negi * half_theta).exp(),
            zero,
            zero,
            (posi * half_theta).exp(),
        ],
        2,
    )
}

pub fn rot_z_jac(theta: f64, multiplier: Option<f64>) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let mult = if let Some(mult) = multiplier {
        mult
    } else {
        1.0
    };
    SquareMatrix::from_vec(
        vec![
            mult * 0.5 * (-half_theta.sin() + negi * half_theta.cos()),
            zero,
            zero,
            mult * 0.5 * (-half_theta.sin() + posi * half_theta.cos()),
        ],
        2,
    )
}

pub fn matrix_distance_squared(a: &SquareMatrix, b: &SquareMatrix) -> f64 {
    // 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]
    // converted to
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    let bc = b.conj();
    let mul = a.multiply(&bc);
    let sum = mul.sum();
    let norm = sum.norm();
    let res = 1f64 - (norm / a.size as f64).powf(2.0);
    res
}

pub fn matrix_distance(a: &SquareMatrix, b: &SquareMatrix) -> f64 {
    let dist_sq = matrix_distance_squared(a, b);
    let res = dist_sq.abs().sqrt();
    res
}

pub fn matrix_distance_squared_jac(
    u: &SquareMatrix,
    m: &SquareMatrix,
    j: Vec<SquareMatrix>,
) -> (f64, Vec<f64>) {
    let s = u.multiply(&m.conj()).sum();
    let dsq = 1f64 - (s.norm() / u.size as f64).powf(2.0);
    if s == r!(0.0) {
        return (dsq, vec![std::f64::INFINITY; j.len()]);
    }
    let jus: Vec<Complex64> = j.iter().map(|ji| u.multiply(&ji.conj()).sum()).collect();
    let jacs = jus
        .iter()
        .map(|jusi| -2.0 * (jusi.re * s.re + jusi.im * s.im) / (u.size as f64).powf(2.0))
        .collect();
    (dsq, jacs)
}

/// Calculates the residuals and the jacobian
pub fn matrix_residuals(a: &SquareMatrix, b: &SquareMatrix, i: &Array2<f64>) -> Vec<f64> {
    let m = b.matmul(&a.H());
    let (re, im) = m.split_complex();
    let r = re - i;
    r.iter().chain(im.iter()).map(|i| *i).collect()
}

pub fn matrix_residuals_jac(
    u: &SquareMatrix,
    _m: &SquareMatrix,
    jacs: &Vec<SquareMatrix>,
) -> Array2<f64> {
    let u_h = u.H();
    Array2::from_shape_vec(
        (jacs.len(), u.size * u.size * 2),
        jacs.iter().fold(Vec::new(), |mut acc, j| {
            let m = j.matmul(&u_h.clone());
            let (re, im) = m.split_complex();
            let row: Vec<f64> = re.iter().chain(im.iter()).map(|i| *i).collect();
            acc.extend(row);
            acc
        }),
    )
    .unwrap()
    .t()
    .to_owned()
}

pub fn qft(n: usize) -> SquareMatrix {
    let root = r!(E).powc(i!(2f64) * PI / n as f64);
    SquareMatrix::from_size_fn(n, |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}
