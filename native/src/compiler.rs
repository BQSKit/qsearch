use crate::circuits::{Gate, QuantumGate, GateProduct};
use crate::gatesets::{GateSet, GateSetLinearCNOT};
use crate::solver::{CMASolver, Solver};
use crate::utils::matrix_distance;
use crate::ComplexUnitary;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::cmp::min;
use std::rc::Rc;
use std::time::{Duration, Instant};

use num_cpus;

use rayon::prelude::*;


fn astar_heuristic(dsq: f64, depth: u8) -> f64 {
    10.0 * dsq + depth as f64
}

#[derive(Clone)]
struct QueueItem(f64, u8, f64, i32, Vec<f64>, GateProduct);

impl Eq for QueueItem {}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &QueueItem) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap()
            .then_with(|| self.3.cmp(&other.3))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &QueueItem) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

pub trait Compiler {
    fn compile(&mut self, u: ComplexUnitary, depth: u8) -> (ComplexUnitary, Gate, Vec<f64>);
}

pub struct SearchCompiler<T, S>
where
    T: GateSet,
    S: Solver,
{
    threshold: f64,
    d: u8,
    beams: usize,
    gateset: T,
    solv: S,
}

impl<T: GateSet, S: Solver> SearchCompiler<T, S> {
    pub fn new(threshold: f64, d: u8, beams: usize, gateset: T, solv: S) -> Self {
        SearchCompiler {
            threshold,
            d,
            beams,
            gateset,
            solv,
        }
    }
}

impl<T: GateSet, S: Solver> Compiler for SearchCompiler<T, S> {
    fn compile(&mut self, u: ComplexUnitary, depth: u8) -> (ComplexUnitary, Gate, Vec<f64>) {
        // figure out the number of qubits needed for the problem
        let size = u.shape()[0];
        let n = (size as f64).log(self.d as f64).round() as u32;
        if self.d.pow(n) as usize != size {
            panic!(format!("The target matrix of size {} is not compatible with qudits of size {}.", size, self.d));
        };

        // generate the needed layers for the problem
        let initial_layer = self.gateset.initial_layer(n as u8, self.d);
        let search_layers = self.gateset.search_layers(n as u8, self.d);

        // Generate a threadpool for the search
        let layers = search_layers.len();
        let cores = num_cpus::get();

        if self.beams < 1 {
            self.beams = cores / layers;
        };

        let pool = rayon::ThreadPoolBuilder::new().num_threads(min(layers * self.beams, cores)).build().unwrap();

        // Problem setup
        let mut best_depth = 0;
        let tiebreaker = 0;

        // Set up the graph
        let root_node = vec![initial_layer];
        let root = GateProduct::new(root_node);
        let mut result = self.solv.solve_for_unitary(root.clone().into(), u.clone());
        let mut best_value = matrix_distance(&result.0, &u);
        let mut best_pair = (result.0, root.clone().into(), result.1.clone());
        if depth == 0 {
            return best_pair;
        };
        let mut queue = BinaryHeap::new();
        queue.push(QueueItem(astar_heuristic(best_value, 0), 0, best_value, -1, result.1, root));
        while queue.len() > 0 {
            if best_value < self.threshold {
                break;
            };
            let mut popped = vec![];
            for _ in 0..self.beams {
                popped.push(queue.pop().unwrap());
            }
            let then = Instant::now();
            let mut new_steps: Vec<(GateProduct, u8, &Vec<f64>)> = vec![];
            for ref current_item in &popped {
                for layer in &search_layers {
                    new_steps.push((current_item.5.push(layer.clone()), current_item.1, &current_item.4));
                }
            }

            for (i, mut step) in new_steps.iter().enumerate() {
                step.0.index = i;
            }
            new_steps.into_par_iter().map(move |step| {

            });

        };
        best_pair
    }
}