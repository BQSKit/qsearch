use crate::circuits::{GateKronecker, GateProduct, QuantumGate, GateIdentity, Gate, GateSingleQubit, GateCNOT};

use std::iter::repeat;
use core::f64::consts::PI;

fn linear_toplogy(double_step: &Gate, single_step: &Gate, dits: u8, d: u8) -> Vec<Gate> {
    let id: Box<dyn QuantumGate> = Box::new(GateIdentity::new(d as usize, d));
    (0..(dits-1)).map(|i| {
        let mut id_double: Vec<Gate> = repeat(id.clone()).take((dits - 1) as usize).collect();
        id_double[i as usize] = double_step.clone();
        let mut id_single: Vec<Gate> = repeat(id.clone()).take((dits - 1) as usize).collect();
        id_single[i as usize] = single_step.clone();
        id_single.insert((i + 1) as usize, single_step.clone());
        let double = GateKronecker::new(id_double);
        let single = GateKronecker::new(id_single);
        let b: Gate = Box::new(GateProduct::new(vec![Box::new(double), Box::new(single)]));
        b
    }).collect()
}

fn fill_row(step: &Gate, dits: u8) -> Gate {
    Box::new(GateKronecker::new(repeat(step.clone()).take(dits as usize).collect()))
}

pub trait GateSet {
    // the compiler takes a gateset class as one of its arguments.  The gateset class generate a gateset to compile a specific circuit with based on the circuit size.

    // dits is the number of qudits used in the gate
    // d is the number of states in the qudit (ie 2 for a qubit, 3 for a qutrit)

    // The first layer in the compilation.  Generally a layer of parameterized single-qubit gates
    fn initial_layer(&self, dits: u8, d: u8) -> Gate;

    // the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    fn search_layers(&self, dits: u8, d: u8) -> Vec<Gate>;
}

pub struct GateSetLinearCNOT(Gate, Gate);

impl GateSetLinearCNOT {
    pub fn new() -> Self {
        GateSetLinearCNOT(Box::new(GateSingleQubit::new(1)), Box::new(GateCNOT::new()))
    }
}

impl GateSet for GateSetLinearCNOT {
    fn initial_layer(&self, dits: u8, _d: u8) -> Gate {
        fill_row(&self.0, dits)
    }

    fn search_layers(&self, dits: u8, d: u8) -> Vec<Gate> {
        linear_toplogy(&self.1, &self.0, dits, d)
    }
}

