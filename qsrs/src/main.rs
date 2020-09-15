use qsrs::compiler::{Compiler, SearchCompiler};
use qsrs::gatesets::GateSetLinearCNOT;
use qsrs::utils::{matrix_distance_squared, qft};

fn main() {
    let u = qft(8);
    let gs = GateSetLinearCNOT::new();
    let comp = SearchCompiler::new(1e-10, gs, None);
    let (m, _circ, _v) = comp.compile(u.clone(), None);
    println!("Distance {}", matrix_distance_squared(&m, &u));
}
