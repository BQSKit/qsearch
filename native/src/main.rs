use search_compiler_rs::solvers::BfgsJacSolver;
use search_compiler_rs::gatesets::{GateSetLinearCNOT, GateSet};
use search_compiler_rs::circuits::{GateProduct, Gate};
use search_compiler_rs::utils::{matrix_distance_squared, qft};
use squaremat::SquareMatrix;
use search_compiler_rs::r;
use num_complex::Complex64;

fn main() {
    /*
    ProductStep(KroneckerStep(QiskitU3QubitStep(), QiskitU3QubitStep(), QiskitU3QubitStep()),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))))
    */
    /*
    ProductStep(KroneckerStep(QiskitU3QubitStep(), QiskitU3QubitStep(), QiskitU3QubitStep()),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(IdentityStep(2), ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep()))),
    KroneckerStep(ProductStep(CNOTStep(), KroneckerStep(XZXZPartialQubitStep(), QiskitU3QubitStep())), IdentityStep(2)))
    */
    //let qft_res = [1, 0, 1, 0, 0, 1, 0, 1];
    let fredkin_res = [1, 1, 0, 1, 0, 1, 1, 0];
    let g = GateSetLinearCNOT::new();
    let initial = g.initial_layer(3, 2);
    let mut layers: Vec<Gate> = vec![initial.into()];
    let search_layers = g.search_layers(3, 2);
    let rest_layers: Vec<Gate> = fredkin_res.iter().map(|i| search_layers[*i].clone().into()).collect();
    layers.extend(rest_layers);
    let circ: Gate = GateProduct::new(layers).into();
    println!("{:?}", circ);
    let q = SquareMatrix::from_vec(vec![r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0),r!(0.0),r!(0.0), r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(0.0),r!(1.0)], 8);
    let solv = BfgsJacSolver::new();
    let (mat, x0) = solv.solve_for_unitary(&circ, &q);
    println!("{}", matrix_distance_squared(&mat, &q));
    println!("{:?}", x0);
}