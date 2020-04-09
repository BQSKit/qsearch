use crate::utils::{rot_x, rot_z, rot_z_jac};
use crate::{i, r};
use squaremat::SquareMatrix;
use enum_dispatch::enum_dispatch;
use num_complex::Complex64;
use reduce::Reduce;
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;
use std::fmt;

#[enum_dispatch]
pub trait QuantumGate: Clone {
    fn mat(&self, v: &[f64]) -> SquareMatrix;
    fn mat_jac(&self, v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>);
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
            Gate::ConstantUnitary(GateConstantUnitary { matrix, .. }) => {
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
    pub matrix: SquareMatrix,
}

impl GateConstantUnitary {
    pub fn new(mat: SquareMatrix, dits: u8) -> Self {
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
    fn mat(&self, _v: &[f64]) -> SquareMatrix {
        self.matrix.clone()
    }

    fn mat_jac(&self, _v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        (self.matrix.clone(), vec![])
    }

    fn inputs(&self) -> usize {
        0
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateIdentity {
    pub data: QuantumGateData,
    pub matrix: SquareMatrix,
}

impl GateIdentity {
    pub fn new(n: usize) -> Self {
        GateIdentity {
            matrix: SquareMatrix::eye(n as usize),
            data: QuantumGateData {
                dits: 1,
                num_inputs: 0,
            },
        }
    }
}

impl QuantumGate for GateIdentity {
    fn mat(&self, _v: &[f64]) -> SquareMatrix {
        self.matrix.clone()
    }

    fn mat_jac(&self, _v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        (self.matrix.clone(), vec![])
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
    fn mat(&self, v: &[f64]) -> SquareMatrix {
        let ct = r!((v[0] * PI).cos());
        let st = r!((v[0] * PI).sin());
        let cp = (v[1] * PI * 2.0).cos();
        let sp = (v[1] * PI * 2.0).sin();
        let cl = (v[2] * PI * 2.0).cos();
        let sl = (v[2] * PI * 2.0).sin();
        SquareMatrix::from_vec(
            vec![
                ct,
                -st * (cl + i!(1.0) * sl),
                st * (cp + i!(1.0) * sp),
                ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
            ],
            2,
        )
    }

    fn mat_jac(&self, v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        let ct = r!((v[0] * PI).cos());
        let st = r!((v[0] * PI).sin());
        let cp = (v[1] * PI * 2.0).cos();
        let sp = (v[1] * PI * 2.0).sin();
        let cl = (v[2] * PI * 2.0).cos();
        let sl = (v[2] * PI * 2.0).sin();
        (
            SquareMatrix::from_vec(
                vec![
                    ct,
                    -st * (cl + i!(1.0) * sl),
                    st * (cp + i!(1.0) * sp),
                    ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                ],
                2,
            ),
            vec![
                SquareMatrix::from_vec(
                    vec![
                        -PI * st,
                        -PI * ct * (cl + i!(1.0) * sl),
                        PI * ct * (cp + i!(1.0) * sp),
                        -PI * st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                    ],
                    2,
                ),
                SquareMatrix::from_vec(
                    vec![
                        r!(0.0),
                        r!(0.0),
                        st * r!(2.0) * PI * (-sp + i!(1.0) * cp),
                        ct * r!(2.0)
                            * PI
                            * (cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp),
                    ],
                    2,
                ),
                SquareMatrix::from_vec(
                    vec![
                        r!(0.0),
                        -st * r!(2.0) * PI * (-sl + i!(1.0) * cl),
                        r!(0.0),
                        ct * r!(2.0)
                            * PI
                            * (-sl * cp - cl * sp + i!(1.0) * -sl * sp + i!(1.0) * cl * cp),
                    ],
                    2,
                ),
            ],
        )
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateXZXZ {
    pub data: QuantumGateData,
    x90: SquareMatrix,
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
    fn mat(&self, v: &[f64]) -> SquareMatrix {
        let rotz = rot_z(v[0] * PI * 2.0 + PI);
        let buffer = rotz.matmul(&self.x90);
        let out = self.x90.matmul(&buffer);
        let rotz2 = rot_z(v[1] * PI * 2.0 - PI);
        rotz2.matmul(&out)
    }

    #[allow(non_snake_case)]
    fn mat_jac(&self, v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        let rotz_jac = rot_z_jac(v[0] * PI * 2.0 + PI, Some(PI * 2.0));
        let buffer = rotz_jac.matmul(&self.x90);
        let out = self.x90.matmul(&buffer);
        let rotz = rot_z(v[1] * PI * 2.0 - PI);
        let J1 = rotz.matmul(&out);

        let rotz2 = rot_z(v[0] * PI * 2.0 + PI);
        let buffer2 = rotz2.matmul(&self.x90);
        let out2 = self.x90.matmul(&buffer2);
        let rotz_jac2 = rot_z_jac(v[1] * PI * 2.0 - PI, Some(PI * 2.0));
        let J2 = rotz_jac2.matmul(&out2);

        let rotz3 = rot_z(v[1] * PI * 2.0 - PI);
        let U = rotz3.matmul(&out2);

        (U, vec![J1, J2])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GateCNOT {
    pub data: QuantumGateData,
    matrix: SquareMatrix,
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
            matrix: SquareMatrix::from_vec(
                vec![
                    one, nil, nil, nil, nil, one, nil, nil, nil, nil, nil, one, nil, nil, one, nil,
                ],
                4,
            ),
        }
    }
}

impl QuantumGate for GateCNOT {
    fn mat(&self, _v: &[f64]) -> SquareMatrix {
        self.matrix.clone()
    }

    fn mat_jac(&self, _v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        (self.matrix.clone(), vec![])
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
    fn mat(&self, v: &[f64]) -> SquareMatrix {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat(v);
        }
        let mut index = 0;
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            })
            .reduce(|mut a: SquareMatrix, b: SquareMatrix| a.kron(&b))
            .unwrap()
    }

    #[allow(non_snake_case)]
    fn mat_jac(&self, v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat_jac(v);
        }
        let mut index = 0;
        let matjacs: Vec<(SquareMatrix, Vec<SquareMatrix>)> = self
            .substeps
            .iter()
            .map(|gate| {
                let MJ = gate.mat_jac(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                MJ
            })
            .collect();

        let mut jacs: Vec<SquareMatrix> = Vec::with_capacity(self.substeps.len());
        let mut U: Option<SquareMatrix> = None;
        for (M, Js) in matjacs {
            jacs = jacs.iter_mut().map(|J| J.kron(&M)).collect();
            for J in Js {
                jacs.push(if let Some(ref mut u) = U {
                    u.kron(&J)
                } else {
                    J.clone()
                });
            }
            U = if let Some(ref mut u) = U {
                Some(u.kron(&M))
            } else {
                Some(M.clone())
            };
        }
        (U.unwrap(), jacs)
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
    fn mat(&self, v: &[f64]) -> SquareMatrix {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat(v);
        }
        let mut index = 0;
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            })
            .reduce(|a: SquareMatrix, b: SquareMatrix| b.matmul(&a))
            .unwrap()
    }

    #[allow(non_snake_case)]
    fn mat_jac(&self, v: &[f64]) -> (SquareMatrix, Vec<SquareMatrix>) {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat_jac(v);
        }
        let mut index = 0;
        let mut submats: Vec<SquareMatrix> = Vec::with_capacity(self.substeps.len());
        let mut subjacs: Vec<Vec<SquareMatrix>> = Vec::with_capacity(self.substeps.len());
        for gate in &self.substeps {
            let (U, Js) = gate.mat_jac(&v[index..index + gate.inputs()]);
            submats.push(U);
            subjacs.push(Js);
            index += gate.inputs();
        }
        let mut B = SquareMatrix::eye(submats[0].size);
        index = 0;
        let mut A = self
            .substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()]);
                index += gate.inputs();
                g
            })
            .reduce(|a: SquareMatrix, b: SquareMatrix| b.matmul(&a))
            .unwrap();
        let mut jacs = Vec::with_capacity(self.substeps.len());
        for (i, Js) in subjacs.iter().enumerate() {
            A = A.matmul(&submats[i].clone().H());
            for J in Js {
                let tmp = J.matmul(&B);
                jacs.push(A.matmul(&tmp));
            }
            B = B.matmul(&submats[i].clone());
        }
        (B, jacs)
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}
