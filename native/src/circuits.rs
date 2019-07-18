use ndarray::arr2;
use num_complex::Complex64;

use crate::utils::{kron, rot_x, rot_y, rot_z};
use crate::ComplexUnitary;

use enum_dispatch::enum_dispatch;

#[enum_dispatch(QuantumGate)]
#[derive(Clone)]
pub enum Gate {
    RX(GateRX),
    RY(GateRY),
    RZ(GateRZ),
    Identity(GateIdentity),
    CNOT(GateCNOT),
    SingleQubit(GateSingleQubit),
    Kronecker(GateKronecker),
    Product(GateProduct),
}

#[enum_dispatch]
pub trait QuantumGate {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary;
    fn assemble(&self, v: &[f64]);
    fn inputs(&self) -> usize;
}

#[derive(Clone)]
struct QuantumGateData {
    pub d: u8,
    pub dits: u8,
    pub num_inputs: usize,
}

#[derive(Clone)]
pub struct GateIdentity {
    data: QuantumGateData,
    mat: ComplexUnitary,
}

impl GateIdentity {
    pub fn new(n: usize, dits: u8) -> Self {
        GateIdentity {
            mat: ComplexUnitary::eye(n),
            data: QuantumGateData {
                d: n as u8,
                dits: dits,
                num_inputs: 0,
            },
        }
    }
}

impl QuantumGate for GateIdentity {
    fn matrix(&self, _v: &[f64]) -> ComplexUnitary {
        self.mat.clone()
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
pub struct GateRZ {
    data: QuantumGateData,
}

impl GateRZ {
    pub fn new(d: u8) -> Self {
        GateRZ {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateRZ {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_z(v[0])
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
pub struct GateRX {
    data: QuantumGateData,
}

impl GateRX {
    pub fn new(d: u8) -> Self {
        GateRX {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateRX {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_x(v[0])
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
pub struct GateRY {
    data: QuantumGateData,
}

impl GateRY {
    pub fn new(d: u8) -> Self {
        GateRY {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateRY {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_y(v[0])
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
pub struct GateSingleQubit {
    data: QuantumGateData,
}

impl GateSingleQubit {
    pub fn new(d: u8) -> Self {
        GateSingleQubit {
            data: QuantumGateData {
                d: d,
                dits: 1,
                num_inputs: 3,
            },
        }
    }
}

/// based on https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/002-The_Weird_and_Wonderful_World_of_the_Qubit/004-advanced_qubit_gates.html
impl QuantumGate for GateSingleQubit {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        let theta = v[0];
        let phi = v[1];
        let lambda = v[2];
        let i_phi = Complex64::new(0.0, phi);
        let i_lambda = Complex64::new(0.0, lambda);
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        arr2(&[
            [Complex64::new(cos, 0.0), -(i_lambda.exp()) * sin],
            [i_phi.exp() * sin, (i_phi + i_lambda).exp() * cos],
        ])
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
pub struct GateCNOT {
    data: QuantumGateData,
    mat: ComplexUnitary,
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
            mat: arr2(&[
                [one, nil, nil, nil],
                [nil, one, nil, nil],
                [nil, nil, nil, one],
                [nil, nil, one, nil],
            ]),
        }
    }
}

impl QuantumGate for GateCNOT {
    fn matrix(&self, _v: &[f64]) -> ComplexUnitary {
        self.mat.clone()
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
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

    pub fn push(mut self, other: Gate) -> Self {
        self.data.num_inputs += other.inputs();
        self.substeps.push(other);
        self
    }
}

impl QuantumGate for GateKronecker {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        let step0 = &self.substeps[0];
        let mut u = step0.matrix(&v[..step0.inputs()]);
        let mut index = step0.inputs();
        for i in 1..self.substeps.len() {
            let step = &self.substeps[i];
            u = kron(&u, &(step.matrix(&v[index..index + step.inputs()])));
            index += step.inputs()
        }
        u
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone)]
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

    pub fn push(&self, g: Gate) -> Self {
        let mut substeps = self.substeps.clone();
        substeps.push(g);
        GateProduct::new(substeps)
    }
}

impl QuantumGate for GateProduct {
    fn matrix(&self, v: &[f64]) -> ComplexUnitary {
        let step0 = &self.substeps[0];
        let mut u = step0.matrix(&v[..step0.inputs()]);
        let mut index = step0.inputs();
        for i in 1..self.substeps.len() {
            let step = &self.substeps[i];
            u = u.dot(&(step.matrix(&v[index..index + step.inputs()])));
            index += step.inputs()
        }
        u
    }

    fn assemble(&self, _v: &[f64]) {}

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}
