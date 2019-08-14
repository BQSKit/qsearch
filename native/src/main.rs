/*use search_compiler::circuits::{
    Gate, GateCNOT, GateIdentity, GateKronecker, GateProduct, GateRX, GateRY, GateRZ,
    GateSingleQubit, QuantumGate,
};
use search_compiler::gatesets::{GateSet, GateSetLinearCNOT};
use search_compiler::utils::matrix_distance;
use search_compiler::ComplexUnitary;

use better_panic::install;
use num_complex::Complex64;

use std::f64::consts::{E, PI};

fn qft(n: usize) -> ComplexUnitary {
    let root = Complex64::new(E, 0f64).powc(Complex64::new(0f64, 2f64) * PI / n as f64);
    ComplexUnitary::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

fn test_qft(qubits: u8, target: &ComplexUnitary, solv: impl Solver) -> (ComplexUnitary, Vec<f64>) {
    let gateset = GateSetLinearCNOT::new();
    let initial = gateset.initial_layer(qubits, 2);
    let search_layers = gateset.search_layers(qubits, 2);
    let mut layers = vec![initial];
    layers.push(search_layers[0].clone());
    layers.push(search_layers[2].clone());
    layers.push(search_layers[1].clone());
    layers.push(search_layers[0].clone());
    layers.push(search_layers[2].clone());
    layers.push(search_layers[1].clone());
    layers.push(search_layers[0].clone());
    layers.push(search_layers[2].clone());
    layers.push(search_layers[1].clone());
    layers.push(search_layers[2].clone());
    layers.push(search_layers[1].clone());
    layers.push(search_layers[0].clone());
    layers.push(search_layers[1].clone());
    layers.push(search_layers[0].clone());
    let search = GateProduct::new(layers).into();
    solv.solve_for_unitary(search, target.clone())
}

fn test_simple(qubits: u8, target: &ComplexUnitary, solv: impl Solver) -> (ComplexUnitary, Vec<f64>) {
    let gateset = GateSetLinearCNOT::new();
    let mut layers = gateset.search_layers(qubits, 2);
    layers.insert(0, gateset.initial_layer(qubits, 2));
    let search = GateProduct::new(layers);
    solv.solve_for_unitary(search.into(), target.clone())
}
*/
fn main() {
    /*
    install();
    let qubits = 4;
    let target = qft(2usize.pow(qubits as u32));
    let solv = CMASolver();
    let sol = test_qft(qubits, &target, solv);
    println!(
        "{}\n\n{}\n\n{:?}",
        &sol.0,
        matrix_distance(&target, &sol.0),
        sol.1
    );*/
}
