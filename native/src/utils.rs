use num_complex::Complex64;

use squaremat::SquareMatrix;

use crate::r;

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
    let res = 1f64 - norm / a.size as f64;
    res
}

pub fn matrix_distance(a: &SquareMatrix, b: &SquareMatrix) -> f64 {
    let dist_sq = matrix_distance_squared(a, b);
    let res = dist_sq.abs().sqrt();
    res
}

pub fn matrix_distance_squared_jac(u: &SquareMatrix, m: &SquareMatrix, j: Vec<SquareMatrix>) -> (f64, Vec<f64>) {
    let s = u.multiply(&m.conj()).sum();
    let dsq = 1f64 - s.norm() / u.size as f64;
    if s == r!(0.0) {
        return (dsq, vec![std::f64::INFINITY; j.len()]);
    }
    let jus: Vec<Complex64> = j.iter().map(|ji| u.multiply(&ji.conj()).sum()).collect();
    let jacs = jus.iter().map(|jusi| -(jusi.re * s.re + jusi.im * s.im) * u.size as f64 / s.norm()).collect();
    (dsq, jacs)
}
