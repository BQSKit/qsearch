#![feature(arbitrary_self_types)]
#![feature(custom_attribute)]
use num_complex::Complex64;

use numpy::{PyArray1, PyArray2};
use pyo3::class::basic::PyObjectProtocol;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyTuple};

use bincode::{deserialize, serialize};

use better_panic::install;

pub mod circuits;
pub mod gatesets;
pub mod utils;

#[macro_export]
macro_rules! c {
    ($re:expr, $im:expr) => {
        Complex64::new($re, $im)
    };
}

#[macro_export]
macro_rules! r {
    ($re:expr) => {
        Complex64::new($re, 0.0)
    };
}

#[macro_export]
macro_rules! i {
    ($im:expr) => {
        Complex64::new(0.0, $im)
    };
}

pub type PyComplexUnitary = PyArray2<Complex64>;

use circuits::{Gate, QuantumGate};
use gatesets::{GateSet, GateSetLinearCNOT};

fn gate_to_object(gate: &Gate, py: Python, circuits: &PyModule) -> PyResult<PyObject> {
    Ok(match gate {
        Gate::CNOT(..) => {
            let gate: PyObject = circuits.get("CNOTStep")?.extract()?;
            gate.call0(py)?
        }
        Gate::Identity(id) => {
            let gate: PyObject = circuits.get("IdentityStep")?.extract()?;
            let args = PyTuple::new(py, vec![id.data.d, id.data.dits]);
            gate.call1(py, args)?
        }
        Gate::U3(..) => {
            let gate: PyObject = circuits.get("QiskitU3QubitStep")?.extract()?;
            gate.call0(py)?
        }
        Gate::XZXZ(..) => {
            let gate: PyObject = circuits.get("XZXZPartialQubitStep")?.extract()?;
            gate.call0(py)?
        }
        Gate::Kronecker(kron) => {
            let gate: PyObject = circuits.get("KroneckerStep")?.extract()?;
            let steps: Vec<PyObject> = kron
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, circuits).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::Product(prod) => {
            let gate: PyObject = circuits.get("ProductStep")?.extract()?;
            let steps: Vec<PyObject> = prod
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, circuits).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
    })
}

#[pyclass(name=Gate, dict)]
struct PyGateWrapper {
    gate: Gate,
}

#[pymethods]
impl PyGateWrapper {
    #[new]
    pub fn new(obj: &PyRawObject, gate: &PyBytes) {
        obj.init(PyGateWrapper {
            gate: deserialize(gate.as_bytes()).unwrap(),
        });
    }

    pub fn matrix(&self, py: Python, v: &PyArray1<f64>) -> Py<PyComplexUnitary> {
        PyComplexUnitary::from_array(py, &self.gate.mat(v.as_slice()).into_ndarray()).to_owned()
    }

    #[getter]
    pub fn num_inputs(&self) -> usize {
        self.gate.inputs()
    }

    fn kind(&self) -> PyResult<String> {
        Ok(match self.gate {
            Gate::CNOT(..) => String::from("CNOT"),
            Gate::Identity(..) => String::from("Identity"),
            Gate::U3(..) => String::from("U3"),
            Gate::XZXZ(..) => String::from("XZXZ"),
            Gate::Kronecker(..) => String::from("Kronecker"),
            Gate::Product(..) => String::from("Product"),
        })
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        self.gate = match state.extract::<&PyBytes>(py) {
            Ok(s) => deserialize(s.as_bytes()).unwrap(),
            Err(e) => panic!(format!("{:?}", e)),
        };
        Ok(())
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.gate).unwrap()).into_object(py))
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let cls = slf.to_object(py).getattr(py, "__class__")?;
        Ok((
            cls,
            (PyBytes::new(py, &serialize(&slf.gate).unwrap()).into_object(py),).into_object(py),
        ))
    }

    pub fn as_python(&self, py: Python) -> PyResult<PyObject> {
        let circuits = py.import("search_compiler.circuits")?;
        gate_to_object(&self.gate, py, circuits)
    }
}

#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyGateWrapper {
    fn __str__(&self) -> PyResult<String> {
        self.kind()
    }

    fn __repr__(&self) -> PyResult<String> {
        self.kind()
    }
}

#[pyclass(name=QubitCNOTLinearNative, dict)]
struct PyGateSetLinearCNOT {
    gateset: GateSetLinearCNOT,
    #[pyo3(get)]
    d: u8,
}

#[pymethods]
impl PyGateSetLinearCNOT {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init(PyGateSetLinearCNOT {
            gateset: GateSetLinearCNOT::new(),
            d: 2,
        });
    }

    pub fn initial_layer(&self, py: Python, n: u8) -> Py<PyGateWrapper> {
        Py::new(
            py,
            PyGateWrapper {
                gate: self.gateset.initial_layer(n, self.d),
            },
        )
        .unwrap()
    }

    pub fn search_layers(&self, py: Python, n: u8) -> Vec<Py<PyGateWrapper>> {
        let layers = self.gateset.search_layers(n, self.d);
        let mut pygates = Vec::with_capacity(layers.len());
        for i in layers {
            pygates.push(Py::new(py, PyGateWrapper { gate: i }).unwrap())
        }
        pygates
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let cls = slf.to_object(py).getattr(py, "__class__")?;
        Ok((cls, PyTuple::empty(py).into_object(py)))
    }
}

#[pymodule]
fn search_compiler_rs(py: Python, _m: &PyModule) -> PyResult<()> {
    install();
    // TODO: Remove this massive hack. Essentially we put our classes in builtins so that they work with pickling
    let builtins = py.import("builtins")?;
    builtins.add_class::<PyGateSetLinearCNOT>()?;
    builtins.add_class::<PyGateWrapper>()?;
    pyo3::type_object::initialize_type::<PyGateSetLinearCNOT>(py, Some("builtins"))?;
    pyo3::type_object::initialize_type::<PyGateWrapper>(py, Some("builtins"))?;
    Ok(())
}
