use num_complex::Complex64;

use complexmat::ComplexUnitary;

pub fn rot_x(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    ComplexUnitary::from_vec(
        vec![
            half_theta.cos(),
            negi * half_theta.sin(),
            negi * half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_y(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    ComplexUnitary::from_vec(
        vec![
            half_theta.cos(),
            -half_theta.sin(),
            half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_z(theta: f64) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    ComplexUnitary::from_vec(
        vec![
            (negi * half_theta).exp(),
            zero,
            zero,
            (posi * half_theta).exp(),
        ],
        2,
    )
}

pub fn rot_z_jac(theta: f64, multiplier: Option<f64>) -> ComplexUnitary {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let mult = if let Some(mult) = multiplier {
        mult
    } else {
        1.0
    };
    ComplexUnitary::from_vec(
        vec![
            mult * 0.5 * (-half_theta.sin() + negi * half_theta.cos()),
            zero,
            zero,
            mult * 0.5 * (-half_theta.sin() + posi * half_theta.cos()),
        ],
        2,
    )
}
