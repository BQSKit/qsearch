use search_compiler_rs::solvers::BfgsJacSolver;
use search_compiler_rs::gatesets::{GateSetLinearCNOT, GateSet};
use search_compiler_rs::circuits::{GateProduct, Gate};
use search_compiler_rs::utils::{matrix_distance_squared, qft};
use squaremat::SquareMatrix;
use search_compiler_rs::r;
use num_complex::Complex64;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

fn main() {
    let mut rng = thread_rng();
    let g = GateSetLinearCNOT::new();
    let initial = g.initial_layer(3, 2);
    let search_layers = g.search_layers(3, 2);
    let q = SquareMatrix::from_vec(vec![r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0)], 8);
    let solv = BfgsJacSolver::new();
    for i in 1..10 {
        for _ in 0..10000 {
            let mut layers: Vec<Gate> = vec![initial.clone().into()];
            let rest_layers: Vec<Gate> = (1..i).map(|_| rng.gen::<bool>()).map(|i| search_layers[i as usize].clone().into()).collect();
            layers.extend(rest_layers);
            let circ: Gate = GateProduct::new(layers).into();
            let (mat, x0) = solv.solve_for_unitary(&circ, &q);
            let dsq = matrix_distance_squared(&mat, &q);
            if dsq < 1e-15 {
                println!("{:?}", circ);
                println!("{:?}", x0);
            }
        }
    }
}