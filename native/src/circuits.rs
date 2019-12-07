use crate::utils::{rot_x, rot_z};
use crate::{i, r};
use complexmat::ComplexUnitary;
use enum_dispatch::enum_dispatch;
use num_complex::Complex64;
use reduce::Reduce;
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;
use std::fmt;

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

impl Gate {
    pub fn dits(&self) -> u8 {
        match self {
            Gate::Identity(i) => i.data.dits,
            Gate::CNOT(c) => c.data.dits,
            Gate::U3(u) => u.data.dits,
            Gate::XZXZ(x) => x.data.dits,
            Gate::Kronecker(k) => k.data.dits,
            Gate::Product(p) => p.data.dits,
        }
    }
}

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::CNOT(..) => write!(f, "CNOTStep()")?,
            Gate::Identity(GateIdentity { matrix, ..}) => write!(f, "IdentityStep({})", matrix.size)?,
            Gate::U3(..) => write!(f, "QiskitU3QubitStep()")?,
            Gate::XZXZ(..) => write!(f, "XZXZPartialQubitStep()")?,
            Gate::Kronecker(GateKronecker {substeps, ..}) => {
                write!(f, "KroneckerStep(")?;
                for (i, step) in substeps.iter().enumerate() {
                    write!(f, "{:?},", step)?;
                    if i != substeps.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, ")")?;
            },
            Gate::Product(GateProduct {substeps, ..}) => {
                write!(f, "ProductStep(")?;
                for (i, step) in substeps.iter().enumerate() {
                    write!(f, "{:?},", step)?;
                    if i != substeps.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, ")")?;
            },
        };
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuantumGateData {
    pub dits: u8,
    pub num_inputs: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateIdentity {
    pub data: QuantumGateData,
    pub matrix: ComplexUnitary,
}

impl GateIdentity {
    pub fn new(n: usize) -> Self {
        GateIdentity {
            matrix: ComplexUnitary::eye(n as i32),
            data: QuantumGateData {
                dits: 1,
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
    pub data: QuantumGateData,
}

impl GateU3 {
    pub fn new() -> Self {
        GateU3 {
            data: QuantumGateData {
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
        ComplexUnitary::from_vec(
            vec![
                ct,
                -st * (cl + i!(1.0) * sl),
                st * (cp + i!(1.0) * sp),
                ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
            ],
            2,
        )
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateXZXZ {
    pub data: QuantumGateData,
    x90: ComplexUnitary,
}

impl GateXZXZ {
    pub fn new() -> Self {
        GateXZXZ {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 2,
            },
            x90: rot_x(PI / 2.0),
        }
    }
}

impl QuantumGate for GateXZXZ {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        let rotz = rot_z(v[0] * PI * 2.0 + PI);
        let buffer = self.x90.matmul(&rotz);
        let out = buffer.matmul(&self.x90);
        out.matmul(&rot_z(v[1] * PI * 2.0 - PI))
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateCNOT {
    pub data: QuantumGateData,
    matrix: ComplexUnitary,
}

impl GateCNOT {
    pub fn new() -> Self {
        let one = Complex64::new(1.0, 0.0);
        let nil = Complex64::new(0.0, 0.0);
        GateCNOT {
            data: QuantumGateData {
                dits: 2,
                num_inputs: 0,
            },
            matrix: ComplexUnitary::from_vec(
                vec![
                    one, nil, nil, nil, nil, one, nil, nil, nil, nil, nil, one, nil, nil, one, nil,
                ],
                4,
            ),
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
    pub data: QuantumGateData,
    pub substeps: Vec<Gate>,
}

impl GateKronecker {
    pub fn new(substeps: Vec<Gate>) -> Self {
        GateKronecker {
            data: QuantumGateData {
                dits: substeps.iter().map(|i| i.dits()).sum(),
                num_inputs: substeps.iter().map(|i| i.inputs()).sum(),
            },
            substeps: substeps,
        }
    }
}

impl QuantumGate for GateKronecker {
    fn mat(&self, v: &[f64]) -> ComplexUnitary {
        let mut index = 0;
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            })
            .reduce(|mut a: ComplexUnitary, b: ComplexUnitary| a.kron(&b))
            .unwrap()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateProduct {
    pub data: QuantumGateData,
    pub substeps: Vec<Gate>,
    pub index: usize,
}

impl GateProduct {
    pub fn new(substeps: Vec<Gate>) -> Self {
        GateProduct {
            data: QuantumGateData {
                dits: substeps[0].dits(),
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
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            })
            .reduce(|a: ComplexUnitary, b: ComplexUnitary| a.matmul(&b))
            .unwrap()
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}
