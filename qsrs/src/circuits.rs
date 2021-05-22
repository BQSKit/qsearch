use crate::utils::{rot_x, rot_x_jac, rot_y, rot_y_jac, rot_z, rot_z_jac, rot_z_jac_mul};
use crate::{i, r};
use enum_dispatch::enum_dispatch;
use num_complex::Complex64;
use squaremat::SquareMatrix;

#[enum_dispatch]
pub trait QuantumGate: Clone {
    fn mat(&self, v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix;
    fn mat_jac(
        &self,
        v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>);
    fn inputs(&self) -> usize;
}

#[enum_dispatch(QuantumGate)]
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    Identity(GateIdentity),
    CNOT(GateCNOT),
    U3(GateU3),
    U2(GateU2),
    U1(GateU1),
    X(GateX),
    Y(GateY),
    Z(GateZ),
    XZXZ(GateXZXZ),
    ZXZXZ(GateZXZXZ),
    Kronecker(GateKronecker),
    Product(GateProduct),
    ConstantUnitary(GateConstantUnitary),
    SingleQutrit(GateSingleQutrit),
}

impl Gate {
    pub fn dits(&self) -> u8 {
        match self {
            Gate::Identity(i) => i.data.dits,
            Gate::CNOT(c) => c.data.dits,
            Gate::U3(u) => u.data.dits,
            Gate::U2(u2) => u2.data.dits,
            Gate::U1(u1) => u1.data.dits,
            Gate::X(x) => x.data.dits,
            Gate::Y(y) => y.data.dits,
            Gate::Z(z) => z.data.dits,
            Gate::XZXZ(x) => x.data.dits,
            Gate::ZXZXZ(x) => x.data.dits,
            Gate::Kronecker(k) => k.data.dits,
            Gate::Product(p) => p.data.dits,
            Gate::ConstantUnitary(pl) => pl.data.dits,
            Gate::SingleQutrit(sq) => sq.data.dits,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGateData {
    pub dits: u8,
    pub num_inputs: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateConstantUnitary {
    pub data: QuantumGateData,
    pub index: usize,
}

impl GateConstantUnitary {
    pub fn new(index: usize, dits: u8) -> Self {
        GateConstantUnitary {
            data: QuantumGateData {
                dits,
                num_inputs: 0,
            },
            index,
        }
    }
}

impl QuantumGate for GateConstantUnitary {
    fn mat(&self, _v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        constant_gates[self.index].clone()
    }

    fn mat_jac(
        &self,
        _v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (constant_gates[self.index].clone(), vec![])
    }

    fn inputs(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateIdentity {
    pub data: QuantumGateData,
    pub index: usize,
}

impl GateIdentity {
    pub fn new(n: usize) -> Self {
        GateIdentity {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 0,
            },
            index: n,
        }
    }
}

impl QuantumGate for GateIdentity {
    fn mat(&self, _v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        constant_gates[self.index].clone()
    }

    fn mat_jac(
        &self,
        _v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (constant_gates[self.index].clone(), vec![])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let ct = r!((v[0] / 2.0).cos());
        let st = r!((v[0] / 2.0).sin());
        let cp = (v[1]).cos();
        let sp = (v[1]).sin();
        let cl = (v[2]).cos();
        let sl = (v[2]).sin();
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

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let ct = r!((v[0] / 2.0).cos());
        let st = r!((v[0] / 2.0).sin());
        let cp = (v[1]).cos();
        let sp = (v[1]).sin();
        let cl = (v[2]).cos();
        let sl = (v[2]).sin();
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
                        -0.5 * st,
                        -0.5 * ct * (cl + i!(1.0) * sl),
                        0.5 * ct * (cp + i!(1.0) * sp),
                        -0.5 * st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                    ],
                    2,
                ),
                SquareMatrix::from_vec(
                    vec![
                        r!(0.0),
                        r!(0.0),
                        st * r!(2.0) / 2.0 * (-sp + i!(1.0) * cp),
                        ct * r!(2.0) / 2.0
                            * (cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp),
                    ],
                    2,
                ),
                SquareMatrix::from_vec(
                    vec![
                        r!(0.0),
                        -st * r!(2.0) / 2.0 * (-sl + i!(1.0) * cl),
                        r!(0.0),
                        ct * r!(2.0) / 2.0
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

#[derive(Clone, Debug, PartialEq)]
pub struct GateU2 {
    pub data: QuantumGateData,
}

impl GateU2 {
    pub fn new() -> Self {
        GateU2 {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 2,
            },
        }
    }
}

/// based on https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/002-The_Weird_and_Wonderful_World_of_the_Qubit/004-advanced_qubit_gates.html
impl QuantumGate for GateU2 {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * v[1]).exp();
        let e2 = (i!(1.0) * v[0]).exp();
        let e3 = (i!(1.0) * (v[0] + v[1])).exp();
        SquareMatrix::from_vec(vec![phase, -phase * e1, phase * e2, phase * e3], 2)
    }

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * v[1]).exp();
        let e2 = (i!(1.0) * v[0]).exp();
        let e3 = (i!(1.0) * (v[0] + v[1])).exp();

        (
            SquareMatrix::from_vec(vec![phase, -phase * e1, phase * e2, phase * e3], 2),
            vec![
                SquareMatrix::from_vec(vec![r!(0.0), r!(0.0), phase * i!(1.0) * e2, phase * i!(1.0) * e3], 2),
                SquareMatrix::from_vec(vec![r!(0.0), phase * i!(-1.0) * e1, r!(0.0), phase * i!(1.0) * e3], 2),
            ],
        )
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateU1 {
    pub data: QuantumGateData,
}

impl GateU1 {
    pub fn new() -> Self {
        GateU1 {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateU1 {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let phase = (i!(1.0) * v[0] / 2.0).exp();
        rot_z(v[0]) * phase
    }

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let phase = (i!(1.0) * v[0] / 2.0).exp();
        let dphase = i!(1.0) / 2.0 * phase;
        (
            rot_z(v[0]) * phase,
            vec![
                rot_z(v[0]) * dphase + rot_z_jac(v[0]) * phase,
            ],
        )
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateX {
    pub data: QuantumGateData,
}

impl GateX {
    pub fn new() -> Self {
        GateX {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateX {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_x(v[0])
    }

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (rot_x(v[0]), vec![rot_x_jac(v[0])])
    }

    fn inputs(&self) -> usize {
        1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateY {
    pub data: QuantumGateData,
}

impl GateY {
    pub fn new() -> Self {
        GateY {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateY {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_y(v[0])
    }

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (rot_y(v[0]), vec![rot_y_jac(v[0])])
    }

    fn inputs(&self) -> usize {
        1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateZ {
    pub data: QuantumGateData,
}

impl GateZ {
    pub fn new() -> Self {
        GateZ {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 1,
            },
        }
    }
}

impl QuantumGate for GateZ {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_z(v[0])
    }

    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (rot_z(v[0]), vec![rot_z_jac(v[0])])
    }

    fn inputs(&self) -> usize {
        1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateXZXZ {
    pub data: QuantumGateData,
    x90_index: usize,
}

impl GateXZXZ {
    pub fn new(x90_index: usize) -> Self {
        GateXZXZ {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 2,
            },
            x90_index,
        }
    }
}

impl QuantumGate for GateXZXZ {
    fn mat(&self, v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let rotz = rot_z(v[0]);
        let buffer = rotz.matmul(&constant_gates[self.x90_index]);
        let out = constant_gates[self.x90_index].matmul(&buffer);
        let rotz2 = rot_z(v[1]);
        rotz2.matmul(&out)
    }

    #[allow(non_snake_case)]
    fn mat_jac(
        &self,
        v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let rotz_jac = rot_z_jac_mul(v[0], Some(1.0));
        let buffer = rotz_jac.matmul(&constant_gates[self.x90_index]);
        let out = constant_gates[self.x90_index].matmul(&buffer);
        let rotz = rot_z(v[1]);
        let J1 = rotz.matmul(&out);

        let rotz2 = rot_z(v[0]);
        let buffer2 = rotz2.matmul(&constant_gates[self.x90_index]);
        let out2 = constant_gates[self.x90_index].matmul(&buffer2);
        let rotz_jac2 = rot_z_jac_mul(v[1], Some(1.0));
        let J2 = rotz_jac2.matmul(&out2);

        let rotz3 = rot_z(v[1]);
        let u = rotz3.matmul(&out2);

        (u, vec![J1, J2])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateZXZXZ {
    pub data: QuantumGateData,
    x90_index: usize,
}

impl GateZXZXZ {
    pub fn new(x90_index: usize) -> Self {
        GateZXZXZ {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 3,
            },
            x90_index,
        }
    }
}

impl QuantumGate for GateZXZXZ {
    fn mat(&self, v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let rotz = rot_z(v[0]);
        let start = constant_gates[self.x90_index].matmul(&rotz);
        let rotz1 = rot_z(v[1]);
        let buffer = rotz1.matmul(&start);
        let out = constant_gates[self.x90_index].matmul(&buffer);
        let rotz2 = rot_z(v[2]);
        rotz2.matmul(&out)
    }

    #[allow(non_snake_case)]
    fn mat_jac(
        &self,
        v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let rotz_jac = rot_z_jac_mul(v[0], Some(1.0));
        let start = constant_gates[self.x90_index].matmul(&rotz_jac);
        let rotz1_1 = rot_z(v[1]);
        let buffer = rotz1_1.matmul(&start);
        let out = constant_gates[self.x90_index].matmul(&buffer);
        let rotz2_1 = rot_z(v[2]);
        let J1 = rotz2_1.matmul(&out);

        let rotz0_2 = rot_z(v[0]);
        let start2 = constant_gates[self.x90_index].matmul(&rotz0_2);
        let rotz_jac1 = rot_z_jac_mul(v[1], Some(1.0));
        let buffer2 = rotz_jac1.matmul(&start2);
        let out2 = constant_gates[self.x90_index].matmul(&buffer2);
        let rotz2_2 = rot_z(v[2]);
        let J2 = rotz2_2.matmul(&out2);

        let rotz_jac2 = rot_z_jac_mul(v[2], Some(1.0));
        let rotz0_3 = rot_z(v[0]);
        let start3 = constant_gates[self.x90_index].matmul(&rotz0_3);
        let rotz1_3 = rot_z(v[1]);
        let buffer3 = rotz1_3.matmul(&start3);
        let out3 = constant_gates[self.x90_index].matmul(&buffer3);
        let J3 = rotz_jac2.matmul(&out3);

        let rotz2_3 = rot_z(v[2]);
        let u = rotz2_3.matmul(&out3);

        (u, vec![J1, J2, J3])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateCNOT {
    pub data: QuantumGateData,
    index: usize,
}

impl GateCNOT {
    pub fn new(index: usize) -> Self {
        GateCNOT {
            data: QuantumGateData {
                dits: 2,
                num_inputs: 0,
            },
            index,
        }
    }
}

impl QuantumGate for GateCNOT {
    fn mat(&self, _v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        constant_gates[self.index].clone()
    }

    fn mat_jac(
        &self,
        _v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (constant_gates[self.index].clone(), vec![])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
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
            substeps,
        }
    }
}

impl QuantumGate for GateKronecker {
    fn mat(&self, v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat(v, constant_gates);
        }
        let mut index = 0;
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()], constant_gates);
                index += gate.inputs();
                g
            })
            .reduce(|mut a: SquareMatrix, b: SquareMatrix| a.kron(&b))
            .unwrap()
    }

    #[allow(non_snake_case)]
    fn mat_jac(
        &self,
        v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat_jac(v, constant_gates);
        }
        let mut index = 0;
        let matjacs: Vec<(SquareMatrix, Vec<SquareMatrix>)> = self
            .substeps
            .iter()
            .map(|gate| {
                let MJ = gate.mat_jac(&v[index..index + gate.inputs()], constant_gates);
                index += gate.inputs();
                MJ
            })
            .collect();

        let mut jacs: Vec<SquareMatrix> = Vec::with_capacity(self.substeps.len());
        let mut u: Option<SquareMatrix> = None;
        for (M, Js) in matjacs {
            jacs = jacs.iter_mut().map(|J| J.kron(&M)).collect();
            for J in Js {
                jacs.push(if let Some(ref mut u) = u {
                    u.kron(&J)
                } else {
                    J.clone()
                });
            }
            u = if let Some(ref mut u) = u {
                Some(u.kron(&M))
            } else {
                Some(M.clone())
            };
        }
        (u.unwrap(), jacs)
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
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
            substeps,
            index: 0,
        }
    }

    pub fn append(mut self, gate: Gate) -> Self {
        self.data.num_inputs += gate.inputs();
        self.substeps.push(gate);
        self
    }
}

impl QuantumGate for GateProduct {
    fn mat(&self, v: &[f64], constant_gates: &[SquareMatrix]) -> SquareMatrix {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat(v, constant_gates);
        }
        let mut index = 0;
        self.substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()], constant_gates);
                index += gate.inputs();
                g
            })
            .reduce(|a: SquareMatrix, b: SquareMatrix| b.matmul(&a))
            .unwrap()
    }

    #[allow(non_snake_case)]
    fn mat_jac(
        &self,
        v: &[f64],
        constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        if self.substeps.len() < 2 {
            return self.substeps[0].mat_jac(v, constant_gates);
        }
        let mut index = 0;
        let mut submats: Vec<SquareMatrix> = Vec::with_capacity(self.substeps.len());
        let mut subjacs: Vec<Vec<SquareMatrix>> = Vec::with_capacity(self.substeps.len());
        for gate in &self.substeps {
            let (u, Js) = gate.mat_jac(&v[index..index + gate.inputs()], constant_gates);
            submats.push(u);
            subjacs.push(Js);
            index += gate.inputs();
        }
        let mut B = SquareMatrix::eye(submats[0].size);
        index = 0;
        let mut A = self
            .substeps
            .iter()
            .map(|gate| {
                let g = gate.mat(&v[index..index + gate.inputs()], constant_gates);
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
            B = submats[i].clone().matmul(&B);
        }
        (B, jacs)
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateSingleQutrit {
    pub data: QuantumGateData,
}

impl GateSingleQutrit {
    pub fn new() -> Self {
        GateSingleQutrit {
            data: QuantumGateData {
                dits: 1,
                num_inputs: 8,
            },
        }
    }
}

impl QuantumGate for GateSingleQutrit {
    fn mat(&self, v: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let s1 = (v[0]).sin();
        let c1 = (v[0]).cos();
        let s2 = (v[1]).sin();
        let c2 = (v[1]).cos();
        let s3 = (v[2]).sin();
        let c3 = (v[2]).cos();

        let p1 = (i!(1.0) * v[3]).exp();
        let m1 = (i!(-1.0) * v[3]).exp();
        let p2 = (i!(1.0) * v[4]).exp();
        let m2 = (i!(-1.0) * v[4]).exp();
        let p3 = (i!(1.0) * v[5]).exp();
        let m3 = (i!(-1.0) * v[5]).exp();
        let p4 = (i!(1.0) * v[6]).exp();
        let m4 = (i!(-1.0) * v[6]).exp();
        let p5 = (i!(1.0) * v[7]).exp();
        let m5 = (i!(-1.0) * v[7]).exp();

        SquareMatrix::from_vec(
            vec![
                c1 * c2 * p1,
                s1 * p3,
                c1 * s2 * p4,
                s2 * s3 * m4 * m5 - s1 * c2 * c3 * p1 * p2 * m3,
                c1 * c3 * p2,
                -c2 * s3 * m1 * m5 - s1 * s2 * c3 * p2 * m3 * p4,
                -s1 * c2 * s3 * p1 * m3 * p5 - s2 * c3 * m2 * m4,
                c1 * s3 * p5,
                c2 * c3 * m1 * m2 - s1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        )
    }
    fn mat_jac(
        &self,
        v: &[f64],
        _constant_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let s1 = (v[0]).sin();
        let c1 = (v[0]).cos();
        let s2 = (v[1]).sin();
        let c2 = (v[1]).cos();
        let s3 = (v[2]).sin();
        let c3 = (v[2]).cos();

        let p1 = (i!(1.0) * v[3]).exp();
        let m1 = (i!(-1.0) * v[3]).exp();
        let p2 = (i!(1.0) * v[4]).exp();
        let m2 = (i!(-1.0) * v[4]).exp();
        let p3 = (i!(1.0) * v[5]).exp();
        let m3 = (i!(-1.0) * v[5]).exp();
        let p4 = (i!(1.0) * v[6]).exp();
        let m4 = (i!(-1.0) * v[6]).exp();
        let p5 = (i!(1.0) * v[7]).exp();
        let m5 = (i!(-1.0) * v[7]).exp();
        let u = SquareMatrix::from_vec(
            vec![
                c1 * c2 * p1,
                s1 * p3,
                c1 * s2 * p4,
                s2 * s3 * m4 * m5 - s1 * c2 * c3 * p1 * p2 * m3,
                c1 * c3 * p2,
                -c2 * s3 * m1 * m5 - s1 * s2 * c3 * p2 * m3 * p4,
                -s1 * c2 * s3 * p1 * m3 * p5 - s2 * c3 * m2 * m4,
                c1 * s3 * p5,
                c2 * c3 * m1 * m2 - s1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        );
        let jt1 = SquareMatrix::from_vec(
            vec![
                -s1 * c2 * p1,
                c1 * p3,
                -s1 * s2 * p4,
                -c1 * c2 * c3 * p1 * p2 * m3,
                -s1 * c3 * p2,
                -c1 * s2 * c3 * p2 * m3 * p4,
                -c1 * c2 * s3 * p1 * m3 * p5,
                -s1 * s3 * p5,
                -c1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        );

        let jt2 = SquareMatrix::from_vec(
            vec![
                -c1 * s2 * p1,
                r!(0.0),
                c1 * c2 * p4,
                c2 * s3 * m4 * m5 + s1 * s2 * c3 * p1 * p2 * m3,
                r!(0.0),
                s2 * s3 * m1 * m5 - s1 * c2 * c3 * p2 * m3 * p4,
                s1 * s2 * s3 * p1 * m3 * p5 - c2 * c3 * m2 * m4,
                r!(0.0),
                -s2 * c3 * m1 * m2 - s1 * c2 * s3 * m3 * p4 * p5,
            ],
            3,
        );

        let jt3 = SquareMatrix::from_vec(
            vec![
                r!(0.0),
                r!(0.0),
                r!(0.0),
                s2 * c3 * m4 * m5 + s1 * c2 * s3 * p1 * p2 * m3,
                -c1 * s3 * p2,
                -c2 * c3 * m1 * m5 + s1 * s2 * s3 * p2 * m3 * p4,
                -s1 * c2 * c3 * p1 * m3 * p5 + s2 * s3 * m2 * m4,
                c1 * c3 * p5,
                -c2 * s3 * m1 * m2 - s1 * s2 * c3 * m3 * p4 * p5,
            ],
            3,
        );

        let je1 = SquareMatrix::from_vec(
            vec![
                i!(1.0) * c1 * c2 * p1,
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                r!(0.0),
                i!(1.0) * c2 * s3 * m1 * m5,
                -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                r!(0.0),
                -i!(1.0) * c2 * c3 * m1 * m2,
            ],
            3,
        );

        let je2 = SquareMatrix::from_vec(
            vec![
                r!(0.0),
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                i!(1.0) * c1 * c3 * p2,
                -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s2 * c3 * m2 * m4,
                r!(0.0),
                -i!(1.0) * c2 * c3 * m1 * m2,
            ],
            3,
        );

        let je3 = SquareMatrix::from_vec(
            vec![
                r!(0.0),
                i!(1.0) * s1 * p3,
                r!(0.0),
                i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                r!(0.0),
                i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                r!(0.0),
                i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        );

        let je4 = SquareMatrix::from_vec(
            vec![
                r!(0.0),
                r!(0.0),
                i!(1.0) * c1 * s2 * p4,
                -i!(1.0) * s2 * s3 * m4 * m5,
                r!(0.0),
                -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s2 * c3 * m2 * m4,
                r!(0.0),
                -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        );

        let je5 = SquareMatrix::from_vec(
            vec![
                r!(0.0),
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s2 * s3 * m4 * m5,
                r!(0.0),
                i!(1.0) * c2 * s3 * m1 * m5,
                -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                i!(1.0) * c1 * s3 * p5,
                -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
            ],
            3,
        );

        (u, vec![jt1, jt2, jt3, je1, je2, je3, je4, je5])
    }

    fn inputs(&self) -> usize {
        self.data.num_inputs
    }
}
