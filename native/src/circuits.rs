use num_complex::Complex64;
use crate::matrix::ComplexUnitary;
use crate::{i,r};
use enum_dispatch::enum_dispatch;
use reduce::Reduce;
use serde::{Serialize, Deserialize};

use core::f64::consts::PI;

#[enum_dispatch]
pub trait QuantumGate: Clone {
    fn mat(&self, v: &[f64]) -> ComplexUnitary;
    fn inputs(&self) -> usize;
}

#[enum_dispatch(QuantumGate)]
#[derive(Serialize, Deserialize, Clone)]
pub enum Gate {
    Identity(GateIdentity),
    CNOT(GateCNOT),
    U3(GateU3),
    XZXZ(GateXZXZ),
    Kronecker(GateKronecker),
    Product(GateProduct),
}


#[derive(Serialize, Deserialize, Clone)]
struct QuantumGateData {
    pub d: u8,
    pub dits: u8,
    pub num_inputs: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateIdentity {
    data: QuantumGateData,
    matrix: ComplexUnitary,
}

impl GateIdentity {
    pub fn new(n: usize, dits: u8) -> Self {
        GateIdentity {
            matrix: ComplexUnitary::eye(n as i32),
            data: QuantumGateData {
                d: n as u8,
                dits: dits,
                num_inputs: 0,
            },
        }
    }
}

impl QuantumGate for GateIdentity {
    fn mat(&self, _v: &[f64]) -> ComplexUnitary {
        self.matrix.clone()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateU3 {
    data: QuantumGateData,
}

impl GateU3 {
    pub fn new(d: u8) -> Self {
        GateU3 {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 3,
            },
        }
    }
}

/// based on https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/002-The_Weird_and_Wonderful_World_of_the_Qubit/004-advanced_qubit_gates.html
impl QuantumGate for GateU3 {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        let ct = r!((v[0] * PI).cos());
        let st = r!((v[0] * PI).sin());
        let cp = (v[1] * PI * 2.0).cos();
        let sp = (v[1] * PI * 2.0).sin();
        let cl = (v[2] * PI * 2.0).cos();
        let sl = (v[2] * PI * 2.0).sin();
        ComplexUnitary::from_vec(vec![ct, -st * (cl + i!(1.0) * sl), st * (cp + i!(1.0) * sp), ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp)], 2)
    }


    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateXZXZ {
    data: QuantumGateData,
}

impl GateXZXZ {
    pub fn new(d: u8) -> Self {
        GateXZXZ {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 2,
            },
        }
    }
}

impl QuantumGate for GateXZXZ {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        ComplexUnitary::from_vec(vec![(-(i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0 + (-i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0)*(-i!(1.0)*v[1] * PI * 2.0 - PI/2.0).exp(), (-i!(1.0)*(i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0 - i!(1.0)*(-i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0)*(i!(1.0)*v[1] * PI * 2.0 - PI/2.0).exp(),
                                      (-i!(1.0)*(i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0 - i!(1.0)*(-i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0)*(-i!(1.0)*v[1] * PI * 2.0 - PI/2.0).exp(), ((i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0 - (-i!(1.0)*v[0] * PI * 2.0 + PI/2.0).exp()/2.0)*(i!(1.0)*v[1] * PI * 2.0 - PI/2.0).exp()], 2)
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateCNOT {
    data: QuantumGateData,
    matrix: ComplexUnitary,
}

impl GateCNOT {
    pub fn new() -> Self {
        let one = Complex64::new(1.0, 0.0);
        let nil = Complex64::new(0.0, 0.0);
        GateCNOT {
            data: QuantumGateData {
                d: 2,
                dits: 1,
                num_inputs: 2,
            },
            matrix: ComplexUnitary::from_vec(vec![
                one, nil, nil, nil,
                nil, one, nil, nil,
                nil, nil, nil, one,
                nil, nil, one, nil,
            ], 4),
        }
    }
}

impl QuantumGate for GateCNOT {
    fn mat(&self, _v: &[f64]) -> ComplexUnitary {
        self.matrix.clone()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateKronecker {
    data: QuantumGateData,
    substeps: Vec<Gate>,
}

impl GateKronecker {
    pub fn new(substeps: Vec<Gate>) -> Self {
        GateKronecker {
            data: QuantumGateData {
                d: 1,
                dits: 1,
                num_inputs: substeps.iter().map(|i| i.inputs()).sum(),
            },
            substeps: substeps,
        }
    }
}

impl QuantumGate for GateKronecker {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        let mut index = 0;
        self.substeps.iter().map(|gate| {
            let g = gate.mat(&v[index..index + gate.inputs()]);
            index += gate.inputs();
            g}
            ).reduce(|mut a: ComplexUnitary, b: ComplexUnitary| {
            a.kron(&b)
        }).unwrap()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateProduct {
    data: QuantumGateData,
    substeps: Vec<Gate>,
    pub index: usize,
}

impl GateProduct {
    pub fn new(substeps: Vec<Gate>) -> Self {
        GateProduct {
            data: QuantumGateData {
                d: 1,
                dits: 1,
                num_inputs: substeps.iter().map(|i| i.inputs()).sum(),
            },
            substeps: substeps,
            index: 0,
        }
    }

}

impl QuantumGate for GateProduct {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        let mut index = 0;
        self.substeps.iter().map(|gate| {
            let g = gate.mat(&v[index..index + gate.inputs()]);
            index += gate.inputs();
            g}
            ).reduce(|mut a: ComplexUnitary, b: ComplexUnitary| {
            a.matmul(&b)
        }).unwrap()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}
