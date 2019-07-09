use ndarray::{Array2, arr2};
use numpy::array::{PyArray2, PyArray1};
use num_complex::Complex64;

use pyo3::prelude::*;
use pyo3::{AsPyPointer, PyTypeInfo, wrap_pyfunction};
use pyo3::types::PyAny;

use better_panic::install;

pub mod utils;
pub mod circuits;
pub mod gatesets;
use utils::{rot_x, rot_y, rot_z, re_rot_z, kron};

pub type ComplexUnitary = Array2<Complex64>;
pub type PyComplexUnitary  = PyArray2<Complex64>;

pub trait QuantumGate {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary;
    fn assembly(&self, v: &[f64]);
    fn input_num(&self) -> usize;
}

impl QuantumGate for PyObject {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        let gil = Python::acquire_gil();
        let py = gil.python();
        unsafe { PyComplexUnitary::from_borrowed_ptr(py,
            self.call_method1(py, "matrix", 
                    (PyArray1::from_slice(py, v),))
                .expect("Invalid array")
            .as_ptr()).to_owned_array()}
    }
    fn assembly(&self, v: &[f64]) {

    }
    fn input_num(&self) -> usize {
        let gil = Python::acquire_gil();
        let py = gil.python();
        unsafe { self.call_method0(py, "inputs")
                .expect("Wat")
            .extract::<usize>(py).expect("wat")}
    }
}

struct QuantumGateData {
    pub d: u8,
    pub dits: u8,
    pub num_inputs: usize,
}

#[pyclass]
pub struct GateIdentity {
    data: QuantumGateData,
    mat: ComplexUnitary,
}

#[pymethods]
impl GateIdentity {
    #[new]
    fn new(obj: &PyRawObject, n: usize, dits: u8) {
        obj.init({
            GateIdentity {
                mat: ComplexUnitary::eye(n),
                data: QuantumGateData {
                    d: n.pow((1/dits) as u32) as u8,
                    dits: dits,
                    num_inputs: 0,
                }
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl QuantumGate for GateIdentity {
    fn make_matrix(&self, _v: &[f64]) -> ComplexUnitary {
        self.mat.clone()
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateRZ {
    data: QuantumGateData,
}

#[pymethods]
impl GateRZ {
    #[new]
    fn new(obj: &PyRawObject, d: u8) {
        obj.init({
            GateRZ {
                data: QuantumGateData {
                    d: d,
                    dits: 1,
                    num_inputs: 1,
                }
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl QuantumGate for GateRZ {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_z(v[0])
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateRX {
    data: QuantumGateData,
}

#[pymethods]
impl GateRX {
    #[new]
    fn new(obj: &PyRawObject, d: u8) {
        obj.init({
            GateRX {
                data: QuantumGateData {
                    d: d,
                    dits: 1,
                    num_inputs: 1,
                }
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl QuantumGate for GateRX {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_x(v[0])
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateRY {
    data: QuantumGateData,
}

#[pymethods]
impl GateRY {
    #[new]
    fn new(obj: &PyRawObject, d: u8) {
        obj.init({
            GateRY {
                data: QuantumGateData {
                    d: d,
                    dits: 1,
                    num_inputs: 1,
                }
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}


impl QuantumGate for GateRY {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        rot_y(v[0])
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateSingleQubit {
    data: QuantumGateData,
}

#[pymethods]
impl GateSingleQubit {
    #[new]
    fn new(obj: &PyRawObject, d: u8) {
        obj.init({
            GateSingleQubit {
                data: QuantumGateData {
                    d: d,
                    dits: 1,
                    num_inputs: 3,
                },
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

/// based on https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/002-The_Weird_and_Wonderful_World_of_the_Qubit/004-advanced_qubit_gates.html
impl QuantumGate for GateSingleQubit {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        let theta = v[0];
        let phi = v[1];
        let lambda = v[2];
        let i_phi = Complex64::new(0.0, phi);
        let i_lambda = Complex64::new(0.0, lambda);
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        arr2(&[[Complex64::new(cos, 0.0), -(i_lambda.exp()) * sin],
              [i_phi.exp() * sin, (i_phi + i_lambda).exp() * cos],
              ])
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateCNOT {
    data: QuantumGateData,
    mat: ComplexUnitary
}

#[pymethods]
impl GateCNOT {
    #[new]
    fn new(obj: &PyRawObject) {
        let one = Complex64::new(1.0, 0.0);
        let nil = Complex64::new(0.0, 0.0);
        obj.init({
            GateCNOT {
                data: QuantumGateData {
                    d: 2,
                    dits: 1,
                    num_inputs: 2,
                },
                mat: arr2(&[[one,nil,nil,nil],
                            [nil,one,nil,nil],
                            [nil,nil,nil,one],
                            [nil,nil,one,nil]])
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl QuantumGate for GateCNOT {
    fn make_matrix(&self, _v: &[f64]) -> ComplexUnitary {
        self.mat.clone()
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}


#[pyclass]
pub struct GateKronecker {
    data: QuantumGateData,
    substeps: Vec<PyObject>,
}

#[pymethods]
impl GateKronecker {
    #[new]
    fn new(obj: &PyRawObject, substeps: Vec<PyObject>) {
        obj.init({
            GateKronecker {
                data: QuantumGateData {
                    d: 1,
                    dits: 1,
                    num_inputs: substeps.iter().map(|i| i.input_num()).sum(),
                },
                substeps: substeps,
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl QuantumGate for GateKronecker {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        let step0 = &self.substeps[0];
        let mut u = step0.make_matrix(&v[..step0.input_num()]);
        let mut index = step0.input_num();
        for i in 1..self.substeps.len() {
            let step = &self.substeps[i];
            u = kron(&u, &(step.make_matrix(&v[index..index + step.input_num()])));
            index += step.input_num()
        }
        u
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}

#[pyclass]
pub struct GateProduct {
    data: QuantumGateData,
    substeps: Vec<PyObject>,
}

#[pymethods]
impl GateProduct {
    #[new]
    fn new(obj: &PyRawObject, py: Python, substeps: Vec<PyObject>) {
        obj.init({
            GateProduct {
                data: QuantumGateData {
                    d: 1,
                    dits: 1,
                    num_inputs: substeps.iter().map(|i| i.input_num()).sum(),
                },
                substeps: substeps,
            }
        });
    }

    fn matrix(&self, py: Python, v: &PyArray1<f64>)  -> PyResult<Py<PyComplexUnitary>> {
        Ok(PyArray2::from_owned_array(py, self.make_matrix(v.as_slice())).to_owned())
    }
    fn assemble(&self, v: &PyArray1<f64>) -> PyResult<()> {
        Ok(self.assembly(v.as_slice()))
    }
    fn inputs(&self) -> PyResult<usize> {
        Ok(self.input_num())
    }
}

impl<'a> QuantumGate for GateProduct {
    fn make_matrix(&self, v: &[f64]) -> ComplexUnitary {
        let step0 = &self.substeps[0];
        let mut u = step0.make_matrix(&v[..step0.input_num()]);
        let mut index = step0.input_num();
        for i in 1..self.substeps.len() {
            let step = &self.substeps[i];
            u = u.dot(&(step.make_matrix(&v[index..index + step.input_num()])));
            index += step.input_num()
        }
        u
    }

    fn assembly(&self, _v: &[f64]) {
        
    }

    fn input_num(&self) -> usize {
        self.data.num_inputs as usize
    }
}


/*
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

    let steps: Vec<&dyn QuantumGate> = vec![&rzz, &idd];
    let kronecker = GateKronecker::new(steps.clone());
    println!("Kronecker of id and rz:\n{}", kronecker.matrix(&vs));
    let prod = GateProduct::new(steps.clone());
    println!("Product of id and rz:\n{}", prod.matrix(&vs));
}
*/

#[pymodule]
fn search_compiler_native(py: Python, m: &PyModule) -> PyResult<()> {
    install();
    m.add_class::<GateIdentity>()?;
    m.add_class::<GateCNOT>()?;
    m.add_class::<GateKronecker>()?;
    m.add_class::<GateProduct>()?;
    m.add_class::<GateSingleQubit>()?;
    m.add_class::<GateRX>()?;
    m.add_class::<GateRY>()?;
    m.add_class::<GateRZ>()?;

    Ok(())
}