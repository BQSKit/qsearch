use search_compiler_native::gatesets::{GateSetLinearCNOT, GateSet};
use search_compiler_native::circuits::{QuantumGate, GateProduct};
use search_compiler_native::utils::matrix_distance;

use core::f64::consts::PI;

use better_panic::install;

fn main() {
    install();
    let qubits = 5;
    let gateset = GateSetLinearCNOT::new();
    for t in 1..1000 {
        let vars = [PI/t as f64; 100000];
        let initial = gateset.initial_layer(qubits, 1);
        assert_eq!(initial.matrix(&vars).shape()[0], 32);
        
        let search = GateProduct::new(gateset.search_layers(qubits, 1));
        assert_eq!(search.matrix(&vars).shape()[0], 4)
    }



}

