use crate::circuits::{Gate, GateCNOT, GateIdentity, GateKronecker, GateProduct, GateU3, GateXZXZ};

use crate::r;
use num_complex::Complex64;
use squaremat::SquareMatrix;
use std::iter::repeat;

fn linear_toplogy(
    double_step: &Gate,
    single_step: &Gate,
    alt_single_step: &Gate,
    id: &Gate,
    dits: u8,
    _d: usize,
    double_weight: usize,
    single_weight: usize,
) -> Vec<(Gate, usize)> {
    let weight = double_weight + 2 * single_weight;
    (0..(dits - 1))
        .map(move |i| {
            let mut layer: Vec<Gate> = repeat(id.clone()).take((dits - 2) as usize).collect();
            let singles = GateKronecker::new(vec![single_step.clone(), alt_single_step.clone()]);
            let parameterized = GateProduct::new(vec![double_step.clone(), singles.into()]);
            layer.insert(i as usize, parameterized.into());
            let b: Gate = GateKronecker::new(layer).into();
            (b, weight)
        })
        .collect()
}

fn fill_row(step: &Gate, dits: u8) -> Gate {
    GateKronecker::new(repeat(step.clone()).take(dits as usize).collect()).into()
}

pub trait Gateset {
    // the compiler takes a gateset class as one of its arguments.  The gateset class generate a gateset to compile a specific circuit with based on the circuit size.

    // dits is the number of qudits used in the gate
    // d is the number of states in the qudit (ie 2 for a qubit, 3 for a qutrit)

    // The first layer in the compilation.  Generally a layer of parameterized single-qubit gates
    fn initial_layer(&self, dits: u8) -> Gate;

    // the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    fn search_layers(&self, dits: u8) -> Vec<(Gate, usize)>;

    fn d(&self) -> usize;

    fn constant_gates(&self) -> Vec<SquareMatrix>;
}

pub struct GateSetLinearCNOT {
    double_step: Gate,
    single_step: Gate,
    single_alt_step: Gate,
    id: Gate,
    pub const_gates: Vec<SquareMatrix>,
}

impl GateSetLinearCNOT {
    pub fn new() -> Self {
        let one = r!(1.0);
        let nil = r!(0.0);
        GateSetLinearCNOT {
            double_step: GateCNOT::new(0).into(),
            single_step: GateU3::new().into(),
            single_alt_step: GateXZXZ::new(1).into(),
            id: GateIdentity::new(2).into(),
            const_gates: vec![
                SquareMatrix::from_vec(
                    vec![
                        one, nil, nil, nil, nil, one, nil, nil, nil, nil, nil, one, nil, nil, one,
                        nil,
                    ],
                    4,
                ),
                crate::utils::rot_x(std::f64::consts::PI / 2.0),
                SquareMatrix::eye(2),
            ],
        }
    }
}

impl Gateset for GateSetLinearCNOT {
    fn initial_layer(&self, dits: u8) -> Gate {
        fill_row(&self.single_step, dits)
    }

    fn search_layers(&self, dits: u8) -> Vec<(Gate, usize)> {
        linear_toplogy(
            &self.double_step,
            &self.single_alt_step,
            &self.single_step,
            &self.id,
            dits,
            self.d(),
            1,
            0,
        )
    }

    fn d(&self) -> usize {
        2
    }

    fn constant_gates(&self) -> Vec<SquareMatrix> {
        self.const_gates.clone()
    }
}
