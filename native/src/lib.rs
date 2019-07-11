use core::f64::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;

pub mod utils;
pub mod circuits;
pub mod gatesets;

use utils::{rot_x, rot_y, rot_z, re_rot_z, kron};
use circuits::{GateCNOT, GateIdentity, GateSingleQubit, GateRX, GateRY, GateRZ, GateKronecker, GateProduct, QuantumGate};


pub type ComplexUnitary = Array2<Complex64>;

pub fn hello() {
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

    let steps: Vec<Box<dyn QuantumGate>> = vec![Box::new(rzz), Box::new(idd)];
    let kronecker = GateKronecker::new(steps);
    println!("Kronecker of id and rz:\n{}", kronecker.matrix(&vs));
    let rzz2 = GateRZ::new(1);
    let idd2 = GateIdentity::new(2, 1);
    let steps2: Vec<Box<dyn QuantumGate>> = vec![Box::new(rzz2), Box::new(idd2)];
    let prod = GateProduct::new(steps2);
    println!("Product of id and rz:\n{}", prod.matrix(&vs));
}
