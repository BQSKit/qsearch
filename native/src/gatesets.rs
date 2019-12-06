use crate::circuits::{Gate, GateCNOT, GateIdentity, GateKronecker, GateProduct, GateU3, GateXZXZ};

use std::iter::repeat;

fn linear_toplogy(
    double_step: &Gate,
    single_step: &Gate,
    alt_single_step: &Gate,
    id: &Gate,
    dits: u8,
    _d: u8,
) -> Vec<Gate> {
    (0..(dits - 1))
        .map(move |i| {
            let mut layer: Vec<Gate> = repeat(id.clone()).take((dits - 2) as usize).collect();
            let singles = GateKronecker::new(vec![single_step.clone(), alt_single_step.clone()]);
            let parameterized = GateProduct::new(vec![double_step.clone(), singles.into()]);
            layer.insert(i as usize, parameterized.into());
            let b: Gate = GateKronecker::new(layer).into();
            b
        })
        .collect()
}

fn fill_row(step: &Gate, dits: u8) -> Gate {
    GateKronecker::new(repeat(step.clone()).take(dits as usize).collect()).into()
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

pub struct GateSetLinearCNOT(Gate, Gate, Gate);

impl GateSetLinearCNOT {
    pub fn new() -> Self {
        GateSetLinearCNOT(
            GateU3::new().into(),
            GateXZXZ::new().into(),
            GateCNOT::new().into(),
        )
    }
}

impl GateSet for GateSetLinearCNOT {
    fn initial_layer(&self, dits: u8, _d: u8) -> Gate {
        fill_row(&self.0, dits)
    }

    fn search_layers(&self, dits: u8, d: u8) -> Vec<Gate> {
        let id = GateIdentity::new(d as usize).into();
        linear_toplogy(&self.2, &self.1, &self.0, &id, dits, d)
    }
}
