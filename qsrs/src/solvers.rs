use crate::circuits::{Gate, QuantumGate};
use crate::utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};
use ceres::CeresSolver;
use ndarray::Array2;
use nlopt::*;
use rand::{thread_rng, Rng};
use squaremat::SquareMatrix;

pub trait Solver {
    fn solve_for_unitary(
        &self,
        circ: &Gate,
        constant_gates: &[SquareMatrix],
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>);
}

pub struct BfgsJacSolver {
    size: usize,
}

impl BfgsJacSolver {
    pub fn new(size: usize) -> Self {
        BfgsJacSolver { size }
    }
}

impl Solver for BfgsJacSolver {
    fn solve_for_unitary(
        &self,
        circ: &Gate,
        constant_gates: &[SquareMatrix],
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>) {
        let i = circ.inputs();
        let f = |x: &[f64], gradient: Option<&mut [f64]>, _user_data: &mut ()| -> f64 {
            let dsq;
            if let Some(grad) = gradient {
                let (m, jac) = circ.mat_jac(&x, constant_gates);
                let (d, j) = matrix_distance_squared_jac(&unitary, &m, jac);
                dsq = d;
                grad.copy_from_slice(&j);
            } else {
                let m = circ.mat(&x, constant_gates);
                dsq = matrix_distance_squared(&unitary, &m);
            }
            dsq
        };
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = if let Some(x) = x0 {
            x
        } else {
            (0..i).map(|_| rng.gen_range(0.0, 1.0)).collect()
        };
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
        (circ.mat(&x0, constant_gates), x0)
    }
}

pub struct LeastSquaresJacSolver {
    solver: CeresSolver,
}

impl LeastSquaresJacSolver {
    pub fn new(num_threads: usize, ftol: f64, gtol: f64) -> Self {
        LeastSquaresJacSolver {
            solver: CeresSolver::new(num_threads, ftol, gtol),
        }
    }
}

impl Solver for LeastSquaresJacSolver {
    fn solve_for_unitary(
        &self,
        circ: &Gate,
        constant_gates: &[SquareMatrix],
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>) {
        let i = circ.inputs();
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = if let Some(x) = x0 {
            x
        } else {
            (0..i).map(|_| rng.gen_range(0.0, 1.0)).collect()
        };
        let eye = Array2::eye(unitary.size);
        let mut cost_fn = |params: &[f64], resids: &mut [f64], jac: Option<&mut [f64]>| {
            let (m, jacs) = circ.mat_jac(&params, constant_gates);
            let res = matrix_residuals(&unitary, &m, &eye);
            resids.copy_from_slice(&res);
            if let Some(jacobian) = jac {
                let jac_mat = matrix_residuals_jac(&unitary, &m, &jacs);
                let v: Vec<f64> = jac_mat.iter().copied().collect();
                jacobian.copy_from_slice(&v)
            }
        };
        self.solver.solve(
            &mut cost_fn,
            &mut x0,
            unitary.size * unitary.size * 2,
            100 * i,
        );
        (circ.mat(&x0, &constant_gates), x0)
    }
}
