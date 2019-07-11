use crate::circuits::Gate;
use crate::ComplexUnitary;
use crate::utils::matrix_distance;

use cmaes::*;

pub trait Solver {
    fn solve_for_unitary(&self, circuit: Gate, u: ComplexUnitary) -> (ComplexUnitary, Vec<f64>);
}

#[derive(Clone)]
struct CMAFitness(ComplexUnitary, Gate);

impl FitnessFunction for CMAFitness {
    fn get_fitness(&self, parameters: &[f64]) -> f64 {
        matrix_distance(&self.0, &(self.1.matrix(parameters)))
    }
}

pub struct CMASolver();

impl Solver for CMASolver {
    fn solve_for_unitary(&self, circuit: Gate, u: ComplexUnitary) -> (ComplexUnitary, Vec<f64>) {
        let fitness = CMAFitness(u, circuit.clone());
        let options = CMAESOptions::default(circuit.inputs())
                                    .threads(1)
                                    .initial_step_size(0.25f64)
                                    .fitness_threshold(0.001f64);
        let (res, _) = cmaes_loop(&fitness, options).unwrap();
        (circuit.matrix(&res), res)

    }
}