#[cfg(feature = "rustsolv")]
use num_complex::Complex64;
#[cfg(feature = "rustsolv")]
use rand::distributions::Uniform;
#[cfg(feature = "rustsolv")]
use rand::{thread_rng, Rng};
#[cfg(feature = "rustsolv")]
use search_compiler_rs::circuits::{Gate, GateProduct};
#[cfg(feature = "rustsolv")]
use search_compiler_rs::gatesets::{GateSet, GateSetLinearCNOT};
#[cfg(feature = "rustsolv")]
use search_compiler_rs::r;
#[cfg(feature = "rustsolv")]
use search_compiler_rs::solvers::BfgsJacSolver;
#[cfg(feature = "rustsolv")]
use search_compiler_rs::utils::{matrix_distance_squared, qft};
#[cfg(feature = "rustsolv")]
use squaremat::SquareMatrix;

#[cfg(feature = "rustsolv")]
fn main() {
    let mut rng = thread_rng();
    let g = GateSetLinearCNOT::new();
    let initial = g.initial_layer(3, 2);
    let search_layers = g.search_layers(3, 2);
    let q = SquareMatrix::from_vec(
        vec![
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(0.0),
            r!(1.0),
        ],
        8,
    );
    let solv = BfgsJacSolver::new();
    for i in 1..10 {
        for _ in 0..10000 {
            let mut layers: Vec<Gate> = vec![initial.clone().into()];
            let rest_layers: Vec<Gate> = (1..i)
                .map(|_| rng.gen::<bool>())
                .map(|i| search_layers[i as usize].clone().into())
                .collect();
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

#[cfg(not(feature = "rustsolv"))]
fn main() {
    panic!("No BFGS!");
}
