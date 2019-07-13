use search_compiler::circuits::{
    Gate, GateCNOT, GateIdentity, GateKronecker, GateProduct, GateRX, GateRY, GateRZ,
    GateSingleQubit, QuantumGate,
};
use search_compiler::gatesets::{GateSet, GateSetLinearCNOT};
use search_compiler::solver::{CMASolver, Solver};
use search_compiler::utils::{re_rot_z, rot_x, rot_y, rot_z};
use search_compiler::ComplexUnitary;

use better_panic::install;
use num_complex::Complex64;

use std::f64::consts::{E, PI};

fn qft(n: usize) -> ComplexUnitary {
    let root = Complex64::new(E, 0f64).powc(Complex64::new(0f64, 2f64) * PI / n as f64);
    ComplexUnitary::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

fn main() {
    install();
    let qubits = 3;
    let gateset = GateSetLinearCNOT::new();
    let solv = CMASolver();
    let mut layers = gateset.search_layers(qubits, 2);
    layers.insert(0, gateset.initial_layer(qubits, 2));
    let search = GateProduct::new(layers).into();
    let target = qft(2usize.pow(qubits as u32));
    let sol = solv.solve_for_unitary(search, target);
    println!("{}", sol.0);
}

#[allow(dead_code)]
fn hello() {
    let mut z = rot_z(PI);
    println!("rotx(PI):\n{}", rot_x(PI));
    println!("roty(PI):\n{}", rot_y(PI));
    println!("rotz(PI):\n{}", z);
    re_rot_z(&mut z, PI / 2.0);
    println!("rerotz(PI/2):\n{}", z);
    let vs = [PI, PI, PI];

    let id = GateIdentity::new(2, 1);
    println!("Identity::matrix():\n{}", id.matrix(&vs));

    let rz = GateRZ::new(1);
    println!("RZ::matrix():\n{}", rz.matrix(&vs));

    let rx = GateRX::new(1);
    println!("RX::matrix():\n{}", rx.matrix(&vs));

    let ry = GateRY::new(1);
    println!("RY::matrix():\n{}", ry.matrix(&vs));

    let one = GateSingleQubit::new(1);
    println!("SingleQubit::matrix():\n{}", one.matrix(&vs));

    let cnot = GateCNOT::new();
    println!("CNOT::matrix():\n{}", cnot.matrix(&vs));
    let rzz = GateRZ::new(1);
    let idd = GateIdentity::new(2, 1);

    let steps: Vec<Gate> = vec![rzz.into(), idd.into()];
    let kronecker = GateKronecker::new(steps);
    println!("Kronecker of id and rz:\n{}", kronecker.matrix(&vs));
    let rzz2 = GateRZ::new(1);
    let idd2 = GateIdentity::new(2, 1);
    let steps2: Vec<Gate> = vec![rzz2.into(), idd2.into()];
    let prod = GateProduct::new(steps2);
    println!("Product of id and rz:\n{}", prod.matrix(&vs));
}
