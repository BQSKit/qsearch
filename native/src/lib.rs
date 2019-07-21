use ndarray::Array2;
use num_complex::Complex64;

use numpy::{PyArray2, PyArray1};
use pyo3::prelude::*;

pub mod circuits;
pub mod gatesets;
pub mod utils;

pub type ComplexUnitary = Array2<Complex64>;
pub type PyComplexUnitary = PyArray2<Complex64>;

use gatesets::{GateSetLinearCNOT, GateSet};
use circuits::{Gate, QuantumGate};


#[pyclass(name=Gate)]
struct PyGateWrapper {
    gate: Gate,
}

#[pymethods]
impl PyGateWrapper {
    pub fn matrix(&self, py: Python, v: &PyArray1<f64>) -> Py<PyComplexUnitary> {
        PyComplexUnitary::from_array(py, &self.gate.matrix(v.as_slice().unwrap())).to_owned()
    }

    #[getter]
    pub fn num_inputs(&self) -> usize {
        self.gate.inputs()
    }
}

#[pyclass(name=QubitCNOTLinearNative)]
struct PyGateSetLinearCNOT {
    gateset: GateSetLinearCNOT,
    d: u8,
}

#[pymethods]
impl PyGateSetLinearCNOT {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init( PyGateSetLinearCNOT {
            gateset: GateSetLinearCNOT::new(),
            d: 2,
        });
    }

    pub fn initial_layer(&self, py: Python, n: u8) -> Py<PyGateWrapper> {
        Py::new(py, PyGateWrapper { gate: self.gateset.initial_layer(n, self.d) } ).unwrap()
    }

    pub fn search_layers(&self, py: Python, n: u8) -> Vec<Py<PyGateWrapper>> {
        let layers = self.gateset.search_layers(n, self.d);
        let mut pygates = Vec::with_capacity(layers.len());
        for i in layers {
            pygates.push(Py::new(py, PyGateWrapper { gate: i }).unwrap())
        };
        pygates
    }
}


#[pymodule]
fn search_compiler_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGateSetLinearCNOT>().unwrap();
    m.add_class::<PyGateWrapper>().unwrap();
    Ok(())
}

