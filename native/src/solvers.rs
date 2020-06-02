use crate::circuits::{Gate, QuantumGate};
use crate::utils::{matrix_distance_squared, matrix_distance_squared_jac};
use nlopt::*;
use rand::{thread_rng, Rng};
use squaremat::SquareMatrix;

pub struct BfgsJacSolver {
    size: usize,
}

impl BfgsJacSolver {
    pub fn new(size: usize) -> Self {
        BfgsJacSolver { size }
    }

    pub fn solve_for_unitary(
        &self,
        circ: &Gate,
        unitary: &SquareMatrix,
    ) -> (SquareMatrix, Vec<f64>) {
        let i = circ.inputs();
        let f = |x: &[f64], gradient: Option<&mut [f64]>, _user_data: &mut ()| -> f64 {
            let dsq;
            if let Some(grad) = gradient {
                let (m, jac) = circ.mat_jac(&x);
                let (d, j) = matrix_distance_squared_jac(&unitary, &m, jac);
                dsq = d;
                grad.copy_from_slice(&j);
            } else {
                let m = circ.mat(&x);
                dsq = matrix_distance_squared(&unitary, &m);
            }
            dsq
        };
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = (0..i).map(|_| rng.gen_range(0.0, 1.0)).collect();
        let mut fmin = Nlopt::new(Algorithm::Lbfgs, i, &f, Target::Minimize, ());
        fmin.set_upper_bound(1.0).unwrap();
        fmin.set_lower_bound(0.0).unwrap();
        fmin.set_stopval(1e-16).unwrap();
        fmin.set_maxeval(15000).unwrap();
        fmin.set_vector_storage(Some(self.size)).unwrap();
        match fmin.optimize(&mut x0) {
            Err((nlopt::FailState::Failure, _)) | Err((nlopt::FailState::RoundoffLimited, _)) => (),
            Ok(_) => (),
            Err(e) => panic!("Failed optimization! ({:?}, {})", e.0, e.1),
        }
        (circ.mat(&x0), x0)
    }
}
