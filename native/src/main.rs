use search_compiler::gatesets::{GateSetLinearCNOT, GateSet};
use search_compiler::circuits::{QuantumGate, GateProduct};
use search_compiler::solver::{CMASolver, Solver};

use search_compiler::ComplexUnitary;

use better_panic::install;

fn main() {
    install();
    let qubits = 5;
    let gateset = GateSetLinearCNOT::new();
    let solv = CMASolver();
    let search = Box::new(GateProduct::new(gateset.search_layers(qubits, 1)));
    let id = ComplexUnitary::eye(4);
    let sol = solv.solve_for_unitary(search, id);
    assert_eq!(sol.0.shape()[0], 4);
}

