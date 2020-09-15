use crate::circuits::*;
use crate::gatesets::Gateset;
use crate::heuristic::astar;
use crate::solvers::Solver;
use crate::utils::*;
use squaremat::SquareMatrix;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::convert::From;
use std::time::Instant;

pub trait Compiler {
    fn compile(&self, u: SquareMatrix, max_depth: Option<usize>) -> (SquareMatrix, Gate, Vec<f64>);
}

pub struct SearchCompiler<G>
where
    G: Gateset + Send + Sync,
{
    threshold: f64,
    gateset: G,
    beams: Option<usize>,
}

impl<G> SearchCompiler<G>
where
    G: Gateset + Send + Sync,
{
    pub fn new(threshold: f64, gateset: G, beams: Option<usize>) -> Self {
        Self {
            threshold,
            gateset,
            beams,
        }
    }
}

#[derive(PartialEq)]
struct QueueItem {
    heuristic: f64,
    depth: usize,
    distance: f64,
    tiebreaker: i64,
    params: Vec<f64>,
    circuit: GateProduct,
}

impl Eq for QueueItem {}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        Ordering::reverse(
            self.heuristic.partial_cmp(&other.heuristic).unwrap_or(
                self.depth.partial_cmp(&other.depth).unwrap_or(
                    self.distance
                        .partial_cmp(&other.distance)
                        .unwrap_or(self.tiebreaker.cmp(&other.tiebreaker)),
                ),
            ),
        )
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<(f64, usize, f64, i64, Vec<f64>, GateProduct)> for QueueItem {
    fn from(item: (f64, usize, f64, i64, Vec<f64>, GateProduct)) -> Self {
        QueueItem {
            heuristic: item.0,
            depth: item.1,
            distance: item.2,
            tiebreaker: item.3,
            params: item.4,
            circuit: item.5,
        }
    }
}

impl<G> Compiler for SearchCompiler<G>
where
    G: Gateset + Send + Sync,
{
    fn compile(&self, u: SquareMatrix, max_depth: Option<usize>) -> (SquareMatrix, Gate, Vec<f64>) {
        let solv = crate::solvers::LeastSquaresJacSolver::new(1, 1e-6, 1e-10);
        let dits = (u.size as f64).log(self.gateset.d() as f64).round() as u8;
        assert_eq!(self.gateset.d().pow(dits as u32), u.size as usize);
        let initial_layer = self.gateset.initial_layer(dits);
        let search_layers = self.gateset.search_layers(dits);

        let cpus = num_cpus::get();
        println!("There are {} processors available in the Pool.", cpus);
        println!("The branching factor is {}.", search_layers.len());
        let beams = match self.beams {
            Some(b) => b,
            None => {
                if !search_layers.is_empty() {
                    cpus / search_layers.len()
                } else {
                    1
                }
            }
        };
        if beams > 1 {
            println!("The beam factor is {}.", beams);
        }
        let pool_size = cpus.min(search_layers.len() * beams);
        println!("Creating a pool of {} workers", pool_size);
        let start = Instant::now();
        // TODO checkpointing with serde
        let mut queue: BinaryHeap<QueueItem> = BinaryHeap::new();
        let mut best_depth = 0;
        let mut tiebreaker = -1;

        let root = GateProduct::new(vec![initial_layer]);
        let result = solv.solve_for_unitary(
            &root.clone().into(),
            &self.gateset.constant_gates(),
            &u,
            None,
        );
        let mut best_val = matrix_distance_squared(&u, &result.0);
        let mut best_pair = (result.0, root.clone().into(), result.1.clone());
        println!("New best! {} at depth 0", best_val);
        if let Some(0) = max_depth {
            return best_pair;
        };
        queue.push((astar(best_val, 0), 0, best_val, tiebreaker, result.1, root).into());

        while !queue.is_empty() {
            if best_val < self.threshold {
                queue.clear();
                break;
            }
            let mut popped = Vec::new();
            for _ in 0..beams {
                match queue.pop() {
                    Some(tup) => {
                        println!(
                            "Popped a node with score: {} at depth: {}",
                            tup.distance, tup.depth
                        );
                        popped.push(tup);
                    }
                    None => break,
                }
            }
            let then = Instant::now();

            let mut new_steps = Vec::with_capacity(popped.len() * search_layers.len());
            for current_tup in popped {
                for search_layer in search_layers.clone() {
                    new_steps.push((
                        current_tup.circuit.clone().append(search_layer.0),
                        current_tup.depth,
                    ))
                }
            }

            let results: Vec<(&GateProduct, (SquareMatrix, Vec<f64>), &usize)> = new_steps
                .iter()
                .map(|(step, depth)| {
                    let solv = crate::solvers::LeastSquaresJacSolver::new(1, 1e-6, 1e-10);
                    (
                        step,
                        solv.solve_for_unitary(
                            &step.clone().into(),
                            &self.gateset.constant_gates(),
                            &u,
                            None,
                        ),
                        depth,
                    )
                })
                .collect();
            for (step, result, current_depth) in results {
                let current_val = matrix_distance_squared(&u.clone(), &result.0);
                if (current_val < best_val
                    && (best_val >= self.threshold || current_depth < &best_depth))
                    || (current_val < self.threshold && current_depth + 1 < best_depth)
                {
                    best_val = current_val;
                    best_pair = (result.0, step.clone().into(), result.1.clone());
                    best_depth = current_depth + 1;
                    println!(
                        "New best! score: {:e} at depth: {}",
                        best_val,
                        current_depth + 1
                    );
                }
                if max_depth.is_none() || current_depth + 1 < max_depth.unwrap() {
                    queue.push(
                        (
                            astar(current_val, current_depth + 1),
                            current_depth + 1,
                            current_val,
                            tiebreaker,
                            result.1,
                            step.clone(),
                        )
                            .into(),
                    );
                    tiebreaker += 1;
                }
            }
            println!(
                "Layer completed after {} seconds",
                then.elapsed().as_millis() as f64 / 1000.0
            );
        }
        println!(
            "Finished compilation at depth {} with score {:e} after {} seconds.",
            best_depth,
            best_val,
            start.elapsed().as_millis() as f64 / 1000.0
        );
        best_pair
    }
}
