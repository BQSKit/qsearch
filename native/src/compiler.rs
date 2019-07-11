use crate::circuits::Gate;
use crate::ComplexUnitary;
use crate::solver::{CMASolver, Solver};
use crate::gatesets::{GateSetLinearCNOT, GateSet};

pub trait Compiler {
    fn compile(&self, U: ComplexUnitary, depth: u8) -> (ComplexUnitary, Gate, Vec<f64>);
}