#![feature(arbitrary_self_types)]
use num_complex::Complex64;

use numpy::{PyArray1, PyArray2};
use pyo3::class::basic::PyObjectProtocol;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::wrap_pyfunction;
use pyo3::exceptions;

use bincode::{deserialize, serialize};

use better_panic::install;
use complexmat::ComplexUnitary;

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

use circuits::{
    Gate, GateCNOT, GateIdentity, GateKronecker, GateConstantUnitary, GateProduct, GateU3, GateXZXZ,
    QuantumGate,
};
use gatesets::{GateSet, GateSetLinearCNOT};

fn gate_to_object(gate: &Gate, py: Python, circuits: &PyModule) -> PyResult<PyObject> {
    Ok(match gate {
        Gate::CNOT(..) => {
            let gate: PyObject = circuits.get("CNOTStep")?.extract()?;
            gate.call0(py)?
        }
        Gate::Identity(id) => {
            let gate: PyObject = circuits.get("IdentityStep")?.extract()?;
            let args = PyTuple::new(py, vec![id.matrix.size, id.data.dits as usize]);
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
        },
        _ => unreachable!(),
    })
}

fn object_to_gate(obj: &PyObject, py: Python) -> PyResult<Gate> {
    let cls = obj.getattr(py, "__class__")?;
    let dunder_name = cls.getattr(py, "__name__")?;
    let name: &str = dunder_name.extract(py)?;
    match name {
        "CNOTStep" => Ok(GateCNOT::new().into()),
        "IdentityStep" => {
            let n = obj.getattr(py, "_n")?.extract(py)?;
            Ok(GateIdentity::new(n).into())
        }
        "QiskitU3QubitStep" => Ok(GateU3::new().into()),
        "XZXZPartialQubitStep" => Ok(GateXZXZ::new().into()),
        "ProductStep" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_substeps")?.extract(py)?;
            let steps: Vec<Gate> = substeps
                .iter()
                .map(|obj| object_to_gate(obj, py).unwrap())
                .collect();
            Ok(GateProduct::new(steps).into())
        }
        "KroneckerStep" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_substeps")?.extract(py)?;
            let steps: Vec<Gate> = substeps
                .iter()
                .map(|obj| object_to_gate(obj, py).unwrap())
                .collect();
            Ok(GateKronecker::new(steps).into())
        }
        _ => {
            if obj.getattr(py, "num_inputs")?.extract::<usize>(py)? == 0 {
                let dits = obj.getattr(py, "dits")?.extract::<u8>(py)?;
                let args: Vec<u8> = vec![];
                let pymat = obj.call_method(py, "matrix", (args,), None)?;
                let mat = pymat.extract::<&PyArray2<Complex64>>(py)?;
                Ok(GateConstantUnitary::new(ComplexUnitary::from_ndarray(mat.to_owned_array()), dits).into())
            } else {
                Err(exceptions::ValueError::py_err(format!(
                    "Unknown gate {}",
                    name
                )))
            }
        }
    }
}

#[pyclass(name=Gate, dict, module = "search_compiler_rs")]
struct PyGateWrapper {
    #[pyo3(get)]
    dits: u8,
    gate: Gate,
}

#[pymethods]
impl PyGateWrapper {
    #[new]
    pub fn new(obj: &PyRawObject, gate: &PyBytes) {
        let gate: Gate = deserialize(gate.as_bytes()).unwrap();
        obj.init(PyGateWrapper {
            dits: gate.dits(),
            gate: gate,
        });
    }

    pub fn jac(&self, py: Python, v: &PyArray1<f64>) -> Vec<Py<PyComplexUnitary>> {
        self.gate.jac(v.as_slice().unwrap()).iter().map(|m| PyComplexUnitary::from_array(py, &m.clone().into_ndarray()).to_owned()).collect()
    }

    pub fn matrix(&self, py: Python, v: &PyArray1<f64>) -> Py<PyComplexUnitary> {
        PyComplexUnitary::from_array(py, &self.gate.mat(v.as_slice().unwrap()).into_ndarray())
            .to_owned()
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
            Gate::ConstantUnitary(..) => String::from("ConstantUnitary")
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
        Ok(PyBytes::new(py, &serialize(&self.gate).unwrap()).to_object(py))
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let cls = slf.to_object(py).getattr(py, "__class__")?;
        Ok((
            cls,
            (PyBytes::new(py, &serialize(&slf.gate).unwrap()).to_object(py),).to_object(py),
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
        Ok(format!("{:?}", self.gate))
    }

    fn __hash__(&self) -> PyResult<isize> {
        let digest = md5::compute(format!("{:?}", self.gate).as_bytes());
        Ok(digest.iter().enumerate().fold(0, |acc, (i, j)| {
            acc + *j as isize * (256isize).pow(i as u32)
        }))
    }
}

#[pyclass(name=QubitCNOTLinearNative, dict, module = "search_compiler_rs")]
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
        let gate = self.gateset.initial_layer(n, self.d);
        Py::new(
            py,
            PyGateWrapper {
                dits: gate.dits(),
                gate: gate,
            },
        )
        .unwrap()
    }

    pub fn search_layers(&self, py: Python, n: u8) -> Vec<Py<PyGateWrapper>> {
        let layers = self.gateset.search_layers(n, self.d);
        let mut pygates = Vec::with_capacity(layers.len());
        for i in layers {
            pygates.push(
                Py::new(
                    py,
                    PyGateWrapper {
                        dits: i.dits(),
                        gate: i,
                    },
                )
                .unwrap(),
            )
        }
        pygates
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let cls = slf.to_object(py).getattr(py, "__class__")?;
        Ok((cls, PyTuple::empty(py).to_object(py)))
    }
}

#[pyfunction]
fn native_from_object(obj: PyObject, py: Python) -> PyResult<Py<PyGateWrapper>> {
    let gate = object_to_gate(&obj, py)?;
    Py::new(
        py,
        PyGateWrapper {
            dits: gate.dits(),
            gate: gate,
        },
    )
}

fn add_module(module: &PyModule, py: Python) -> PyResult<()> {
    py.import("sys")?
        .dict()
        .get_item("modules")
        .unwrap()
        .downcast_mut::<PyDict>()?
        .set_item(module.name()?, module)
}

#[pymodule]
fn search_compiler_rs(py: Python, m: &PyModule) -> PyResult<()> {
    install();
    m.add_wrapped(wrap_pyfunction!(native_from_object))?;
    m.add_class::<PyGateSetLinearCNOT>()?;
    m.add_class::<PyGateWrapper>()?;
    add_module(m, py)?;
    pyo3::type_object::initialize_type::<PyGateSetLinearCNOT>(py, Some(m.name()?))?;
    pyo3::type_object::initialize_type::<PyGateWrapper>(py, Some(m.name()?))?;
    Ok(())
}
