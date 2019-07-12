use crate::circuits::Gate;
use crate::gatesets::{GateSet, GateSetLinearCNOT};
use crate::solver::{CMASolver, Solver};
use crate::ComplexUnitary;

pub trait Compiler {
    fn compile(&self, u: ComplexUnitary, depth: u8) -> (ComplexUnitary, Gate, Vec<f64>);
}

pub struct SearchCompiler {
    threshold: f64,
    d: u8,
    beams: u8,
}
