mod bfgs;
use bfgs::Lbfgsb;

use rand::{thread_rng, Rng};
use crate::utils::{matrix_distance_squared, matrix_distance_squared_jac};
use crate::circuits::{Gate, QuantumGate};
use squaremat::SquareMatrix;

pub struct BfgsJacSolver {
}

impl BfgsJacSolver {
    pub fn new() -> Self {
        BfgsJacSolver {}
    }

    pub fn solve_for_unitary(&self, circ: &Gate, unitary: &SquareMatrix) -> (SquareMatrix, Vec<f64>) {
        let i = circ.inputs();
        let f = |x:&Vec<f64>| { matrix_distance_squared(&unitary, &circ.mat(&x))};
        let g = |x:&Vec<f64>| {
            let (m, jac) = circ.mat_jac(&x);
            matrix_distance_squared_jac(&unitary, &m, jac).1
        };
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = (0..i).map(|_| rng.gen_range(0.0, 1.0)).collect();
        let mut fmin = Lbfgsb::new(&mut x0,&f,&g);
        for index in 0..i {
            fmin.set_upper_bound(index, 1.0);
            fmin.set_lower_bound(index, 0.0);
        }
        fmin.set_verbosity(-1);
        fmin.max_iteration(15000);
        fmin.minimize();
        (circ.mat(&x0), x0)
    }
}