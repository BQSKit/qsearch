use search_compiler::circuits::{GateProduct, QuantumGate};
use search_compiler::gatesets::{GateSet, GateSetLinearCNOT};
use search_compiler::solver::{CMASolver, Solver};
use search_compiler::ComplexUnitary;

use better_panic::install;
use num_complex::Complex64;

use std::f64::consts::{E, PI};
use std::rc::Rc;

fn qft(n: usize) -> ComplexUnitary {
    let root = Complex64::new(E, 0f64).powc(Complex64::new(0f64, 2f64) * PI / n as f64);
    ComplexUnitary::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

fn main() {
    install();
    let qubits = 3;
    let gateset = GateSetLinearCNOT::new();
    let solv = CMASolver();
    let mut layers = gateset.search_layers(qubits, 2);
    layers.insert(0, gateset.initial_layer(qubits, 2));
    let search = Rc::new(GateProduct::new(layers));
    let target = qft(2usize.pow(qubits as u32));
    let sol = solv.solve_for_unitary(search, target);
    assert_eq!(sol.0.shape()[0], 2usize.pow(qubits as u32));
}
