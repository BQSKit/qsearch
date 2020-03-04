use crate::utils::{rot_x, rot_z, rot_z_jac};
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
    fn jac(&self, v: &[f64]) -> Vec<ComplexUnitary>;
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
    ConstantUnitary(GateConstantUnitary),
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
            Gate::ConstantUnitary(pl) => pl.data.dits,
        }
    }
}

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::CNOT(..) => write!(f, "CNOTStep()")?,
            Gate::Identity(GateIdentity { matrix, .. }) => {
                write!(f, "IdentityStep({})", matrix.size)?
            }
            Gate::U3(..) => write!(f, "QiskitU3QubitStep()")?,
            Gate::XZXZ(..) => write!(f, "XZXZPartialQubitStep()")?,
            Gate::Kronecker(GateKronecker { substeps, .. }) => {
                write!(f, "KroneckerStep(")?;
                for (i, step) in substeps.iter().enumerate() {
                    write!(f, "{:?},", step)?;
                    if i != substeps.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, ")")?;
            }
            Gate::Product(GateProduct { substeps, .. }) => {
                write!(f, "ProductStep(")?;
                for (i, step) in substeps.iter().enumerate() {
                    write!(f, "{:?},", step)?;
                    if i != substeps.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, ")")?;
            }
            Gate::ConstantUnitary(GateConstantUnitary { matrix, ..}) => {
                write!(f, "Unitary({:?})", matrix)?;
            }
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
pub struct GateConstantUnitary {
    pub data: QuantumGateData,
    pub matrix: ComplexUnitary,
}

impl GateConstantUnitary {
    pub fn new(mat: ComplexUnitary, dits: u8) -> Self {
        GateConstantUnitary {
            data: QuantumGateData {
                dits,
                num_inputs: 0,
            },
            matrix: mat,
        }
    }
}

impl QuantumGate for GateConstantUnitary {
    fn mat(&self, _v: &[f64]) -> ComplexUnitary {
        self.matrix.clone()
    }

    fn jac(&self, _v: &[f64]) -> Vec<ComplexUnitary> {
        vec![]
    }

    fn inputs(&self) -> usize {
        0
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateIdentity {
    pub data: QuantumGateData,
    pub matrix: ComplexUnitary,
}

impl GateIdentity {
    pub fn new(n: usize) -> Self {
        GateIdentity {
            matrix: ComplexUnitary::eye(n as usize),
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

    fn jac(&self, _v: &[f64]) -> Vec<ComplexUnitary> {
        vec![]
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

    fn jac(&self, v: &[f64]) -> Vec<ComplexUnitary> {
        let ct = r!((v[0] * PI).cos());
        let st = r!((v[0] * PI).sin());
        let cp = (v[1] * PI * 2.0).cos();
        let sp = (v[1] * PI * 2.0).sin();
        let cl = (v[2] * PI * 2.0).cos();
        let sl = (v[2] * PI * 2.0).sin();
        vec![
            ComplexUnitary::from_vec(
                vec![
                    -PI*st, -PI*ct * (cl + i!(1.0) * sl), PI*ct * (cp + i!(1.0) * sp), -PI*st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp)
                ],
                2,
            ),
            ComplexUnitary::from_vec(
                vec![
                    r!(0.0),r!(0.0),
                    st * r!(2.0)*PI*(-sp + i!(1.0) * cp), ct * r!(2.0)*PI*(cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp)
                ],
                2,
            ),
            ComplexUnitary::from_vec(
                vec![
                    r!(0.0), -st * r!(2.0)*PI*(-sl + i!(1.0) * cl), r!(0.0), ct * r!(2.0)*PI*(-sl * cp - cl * sp + i!(1.0) * -sl * sp + i!(1.0) * cl * cp)
                ],
                2,
            ),
        ]
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

    #[allow(non_snake_case)]
    fn jac(&self, v: &[f64]) -> Vec<ComplexUnitary> {
        let rotz = rot_z_jac(v[0] * PI * 2.0 + PI, Some(PI * 2.0));
        let buffer = self.x90.matmul(&rotz);
        let out = buffer.matmul(&self.x90);
        let J1 = out.matmul(&rot_z(v[1] * PI * 2.0 - PI));

        let rotz2 = rot_z(v[0] * PI * 2.0 + PI);
        let buffer2 = self.x90.matmul(&rotz2);
        let out2 = buffer2.matmul(&self.x90);
        let J2 = out2.matmul(&rot_z_jac(v[1]*PI*2.0-PI, Some(PI * 2.0)));
        vec![J1, J2]
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

    fn jac(&self, _v: &[f64]) -> Vec<ComplexUnitary> {
        vec![]
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

    #[allow(non_snake_case)]
    fn jac(&self, v: &[f64]) -> Vec<ComplexUnitary> {
        let mut index = 0;
        let mut jacs = Vec::with_capacity(self.substeps.len());
        let matrices: Vec<ComplexUnitary> = self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            }).collect();
        index = 0;
        for (i, jacstep) in self.substeps.iter().enumerate() {
            if jacstep.inputs() == 0 {
                continue;
            }
            let Js = jacstep.jac(&v[index..index+jacstep.inputs()]);
            index += jacstep.inputs();
            for J in Js {
                let mut M = if i == 0 {J.clone()} else {matrices[0].clone()};
                for k in 1..matrices.len() {
                    let M1 = if i == k {J.clone()} else {matrices[k].clone()};
                    M = M.kron(&M1);
                }
                jacs.push(M);
            }

        }
        jacs
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

    #[allow(non_snake_case)]
    fn jac(&self, v: &[f64]) -> Vec<ComplexUnitary> {
        let mut index = 0;
        let mut jacs = Vec::with_capacity(self.substeps.len());
        let matrices: Vec<ComplexUnitary> = self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            }).collect();
        index = 0;
        for (i, jacstep) in self.substeps.iter().enumerate() {
            if jacstep.inputs() == 0 {
                continue;
            }
            let Js = jacstep.jac(&v[index..index+jacstep.inputs()]);
            index += jacstep.inputs();
            for J in Js {
                let mut M = if i == 0 {J.clone()} else {matrices[0].clone()};
                for k in 1..matrices.len() {
                    let M1 = if i == k {J.clone()} else {matrices[k].clone()};
                    M = M.matmul(&M1);
                }
                jacs.push(M);
            }

        }
        jacs
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}
