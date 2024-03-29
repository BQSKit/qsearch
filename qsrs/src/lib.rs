#[cfg(feature = "python")]
use num_complex::Complex64;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2};
#[cfg(feature = "python")]
use pyo3::class::basic::PyObjectProtocol;
#[cfg(feature = "python")]
use pyo3::exceptions;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyTuple;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "python")]
use better_panic::install;
#[cfg(all(feature = "python", feature = "rustopt"))]
use solvers::{BfgsJacSolver, LeastSquaresJacSolver, Solver};
#[cfg(feature = "python")]
use squaremat::SquareMatrix;

pub mod circuits;
pub mod compiler;
pub mod gatesets;
pub mod heuristic;
#[cfg(feature = "rustopt")]
pub mod solvers;
pub mod utils;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(any(feature = "static", feature = "default"))]
extern crate openblas_src;

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

#[cfg(feature = "python")]
pub type PySquareMatrix = PyArray2<Complex64>;

#[cfg(feature = "python")]
use circuits::{
    Gate, GateCNOT, GateConstantUnitary, GateIdentity, GateKronecker, GateProduct, GateRXX,
    GateRYY, GateRZZ, GateSingleQutrit, GateU1, GateU2, GateU3, GateX, GateXZXZ, GateY, GateZ,
    GateZXZXZ, QuantumGate,
};

#[cfg(feature = "python")]
use utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log_2(x: usize) -> usize {
    num_bits::<usize>() - x.leading_zeros() as usize - 1
}

#[cfg(feature = "python")]
fn gate_to_object(
    gate: &Gate,
    py: Python,
    constant_gates: &[SquareMatrix],
    gates: &PyModule,
) -> PyResult<PyObject> {
    Ok(match gate {
        Gate::CNOT(..) => {
            let gate: PyObject = gates.getattr("CNOTGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Identity(id) => {
            let gate: PyObject = gates.getattr("IdentityGate")?.extract()?;
            let args = PyTuple::new(
                py,
                vec![log_2(constant_gates[id.index].size), id.data.dits as usize],
            );
            gate.call1(py, args)?
        }
        Gate::U3(..) => {
            let gate: PyObject = gates.getattr("U3Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::U2(..) => {
            let gate: PyObject = gates.getattr("U2Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::U1(..) => {
            let gate: PyObject = gates.getattr("U1Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::X(..) => {
            let gate: PyObject = gates.getattr("XGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Y(..) => {
            let gate: PyObject = gates.getattr("YGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Z(..) => {
            let gate: PyObject = gates.getattr("ZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::RXX(..) => {
            let gate: PyObject = gates.getattr("RXXGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::RYY(..) => {
            let gate: PyObject = gates.getattr("RYYGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::RZZ(..) => {
            let gate: PyObject = gates.getattr("RZZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::XZXZ(..) => {
            let gate: PyObject = gates.getattr("XZXZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::ZXZXZ(..) => {
            let gate: PyObject = gates.getattr("ZXZXZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Kronecker(kron) => {
            let gate: PyObject = gates.getattr("KroneckerGate")?.extract()?;
            let steps: Vec<PyObject> = kron
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, &constant_gates, gates).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::Product(prod) => {
            let gate: PyObject = gates.getattr("ProductGate")?.extract()?;
            let steps: Vec<PyObject> = prod
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, &constant_gates, gates).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::SingleQutrit(..) => {
            let gate: PyObject = gates.getattr("SingleQutritGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::ConstantUnitary(u) => {
            let mat = constant_gates[u.index].clone();
            let gate: PyObject = gates.getattr("UGate")?.extract()?;
            let tup = PyTuple::new(
                py,
                [PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned()].iter(),
            );
            gate.call1(py, tup)?
        }
    })
}

#[cfg(feature = "python")]
fn object_to_gate(
    obj: &PyObject,
    constant_gates: &mut Vec<SquareMatrix>,
    py: Python,
) -> PyResult<Gate> {
    let cls = obj.getattr(py, "__class__")?;
    let dunder_name = cls.getattr(py, "__name__")?;
    let name: &str = dunder_name.extract(py)?;
    match name {
        "CNOTGate" => {
            let one = r!(1.0);
            let nil = r!(0.0);
            let index = constant_gates.len();
            constant_gates.push(SquareMatrix::from_vec(
                vec![
                    one, nil, nil, nil, nil, one, nil, nil, nil, nil, nil, one, nil, nil, one, nil,
                ],
                4,
            ));
            Ok(GateCNOT::new(index).into())
        }
        "IdentityGate" => {
            let index = constant_gates.len();
            let n = obj.getattr(py, "qudits")?.extract(py)?;
            constant_gates.push(SquareMatrix::eye(2usize.pow(n)));
            Ok(GateIdentity::new(index).into())
        }
        "U3Gate" => Ok(GateU3::new().into()),
        "U2Gate" => Ok(GateU2::new().into()),
        "U1Gate" => Ok(GateU1::new().into()),
        "XGate" => Ok(GateX::new().into()),
        "YGate" => Ok(GateY::new().into()),
        "ZGate" => Ok(GateZ::new().into()),
        "RXXGate" => Ok(GateRXX::new().into()),
        "RYYGate" => Ok(GateRYY::new().into()),
        "RZZGate" => Ok(GateRZZ::new().into()),
        "XZXZGate" => {
            let unitaries = py.import("qsearch.unitaries")?;
            let sx = unitaries.getattr("sqrt_x")?;
            let pymat = sx.extract::<&PyArray2<Complex64>>()?;
            let mat = unsafe { pymat.as_array() };
            let index = constant_gates.len();
            constant_gates.push(SquareMatrix::from_ndarray(mat.to_owned()).T());
            Ok(GateXZXZ::new(index).into())
        }
        "ZXZXZGate" => {
            let unitaries = py.import("qsearch.unitaries")?;
            let sx = unitaries.getattr("sqrt_x")?;
            let pymat = sx.extract::<&PyArray2<Complex64>>()?;
            let mat = unsafe { pymat.as_array() };
            let index = constant_gates.len();
            constant_gates.push(SquareMatrix::from_ndarray(mat.to_owned()).T());
            Ok(GateZXZXZ::new(index).into())
        }
        "ProductGate" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_subgates")?.extract(py)?;
            let mut steps: Vec<Gate> = Vec::with_capacity(substeps.len());
            for step in substeps {
                steps.push(object_to_gate(&step, constant_gates, py)?);
            }
            Ok(GateProduct::new(steps).into())
        }
        "KroneckerGate" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_subgates")?.extract(py)?;
            let mut steps: Vec<Gate> = Vec::with_capacity(substeps.len());
            for step in substeps {
                steps.push(object_to_gate(&step, constant_gates, py)?);
            }
            Ok(GateKronecker::new(steps).into())
        }
        "SingleQutritGate" => Ok(GateSingleQutrit::new().into()),
        "Gate" => {
            let g = obj.extract::<Py<PyGateWrapper>>(py)?;
            let wrapper = g.as_ref(py).try_borrow()?;
            Ok(wrapper.gate.clone())
        }
        _ => {
            if obj.getattr(py, "num_inputs")?.extract::<usize>(py)? == 0 {
                let dits = obj.getattr(py, "qudits")?.extract::<u8>(py)?;
                let args: Vec<u8> = vec![];
                let pyobj = obj.call_method(py, "matrix", (args,), None)?;
                let pymat = pyobj.extract::<&PyArray2<Complex64>>(py)?;
                let mat = unsafe { pymat.as_array() };
                let index = constant_gates.len();
                constant_gates.push(SquareMatrix::from_ndarray(mat.to_owned()).T());
                Ok(GateConstantUnitary::new(index, dits).into())
            } else {
                Err(exceptions::PyValueError::new_err(format!(
                    "Unknown gate {}",
                    name
                )))
            }
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Gate", module = "qsrs")]
struct PyGateWrapper {
    #[pyo3(get)]
    dits: u8,
    pub gate: Gate,
    constant_gates: Vec<SquareMatrix>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGateWrapper {
    #[new]
    pub fn new(pygate: PyObject, py: Python) -> Self {
        let mut constant_gates = Vec::new();
        let gate = object_to_gate(&pygate, &mut constant_gates, py).unwrap();
        PyGateWrapper {
            dits: gate.dits(),
            gate: gate,
            constant_gates,
        }
    }

    pub fn mat_jac(
        &mut self,
        py: Python,
        v: &PyArray1<f64>,
    ) -> (Py<PySquareMatrix>, Vec<Py<PySquareMatrix>>) {
        let (m, jac) = self
            .gate
            .mat_jac(unsafe { v.as_slice().unwrap() }, &mut self.constant_gates);
        (
            PySquareMatrix::from_array(py, &m.clone().into_ndarray()).to_owned(),
            jac.iter()
                .map(|j| PySquareMatrix::from_array(py, &j.clone().into_ndarray()).to_owned())
                .collect(),
        )
    }

    pub fn matrix(&mut self, py: Python, v: &PyArray1<f64>) -> Py<PySquareMatrix> {
        PySquareMatrix::from_array(
            py,
            &self
                .gate
                .mat(unsafe { v.as_slice().unwrap() }, &mut self.constant_gates)
                .into_ndarray(),
        )
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
            Gate::U2(..) => String::from("U2"),
            Gate::U1(..) => String::from("U1"),
            Gate::X(..) => String::from("X"),
            Gate::Y(..) => String::from("Y"),
            Gate::Z(..) => String::from("Z"),
            Gate::RXX(..) => String::from("RXX"),
            Gate::RYY(..) => String::from("RYY"),
            Gate::RZZ(..) => String::from("RZZ"),
            Gate::XZXZ(..) => String::from("XZXZ"),
            Gate::ZXZXZ(..) => String::from("ZXZXZ"),
            Gate::Kronecker(..) => String::from("Kronecker"),
            Gate::Product(..) => String::from("Product"),
            Gate::ConstantUnitary(..) => String::from("ConstantUnitary"),
            Gate::SingleQutrit(..) => String::from("SingleQutrit"),
        })
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let tup: (PyObject,) = (slf.as_python(py)?,);
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, tup.into_py(py)))
    }

    pub fn as_python(&self, py: Python) -> PyResult<PyObject> {
        let gates = py.import("qsearch.gates")?;
        gate_to_object(&self.gate, py, &self.constant_gates, gates)
    }
}

#[cfg(feature = "python")]
#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyGateWrapper {
    fn __str__(&self) -> PyResult<String> {
        self.kind()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("RustGate()"))
    }

    fn __hash__(&self) -> PyResult<isize> {
        let digest = md5::compute(format!("{:?}", self.gate).as_bytes());
        Ok(digest.iter().enumerate().fold(0, |acc, (i, j)| {
            acc + *j as isize * (256isize).pow(i as u32)
        }))
    }
}

#[cfg(all(feature = "python", feature = "rustopt"))]
#[pyclass(name = "BFGS_Jac_SolverNative", module = "qsrs")]
struct PyBfgsJacSolver {
    size: usize,
    #[pyo3(get)]
    distance_metric: String,
}

#[cfg(all(feature = "python", feature = "rustopt"))]
#[pymethods]
impl PyBfgsJacSolver {
    #[new]
    fn new(memory_size: Option<usize>) -> Self {
        if let Some(size) = memory_size {
            PyBfgsJacSolver {
                size,
                distance_metric: String::from("Frobenius"),
            }
        } else {
            PyBfgsJacSolver {
                size: 10,
                distance_metric: String::from("Frobenius"),
            }
        }
    }

    fn solve_for_unitary(
        &self,
        py: Python,
        circuit: PyObject,
        options: PyObject,
        x0: Option<PyObject>,
    ) -> PyResult<(Py<PySquareMatrix>, Py<PyArray1<f64>>)> {
        let u = options.getattr(py, "target")?;
        let (circ, constant_gates) = match circuit.extract::<Py<PyGateWrapper>>(py) {
            Ok(c) => {
                let pygate = c.as_ref(py).try_borrow().unwrap();
                (pygate.gate.clone(), pygate.constant_gates.clone())
            }
            Err(_) => {
                let mut constant_gates = Vec::new();
                let gate = object_to_gate(&circuit, &mut constant_gates, py)?;
                (gate, constant_gates)
            }
        };
        let unitary =
            SquareMatrix::from_ndarray(u.extract::<&PySquareMatrix>(py)?.to_owned_array());
        let x0_rust = if let Some(x) = x0 {
            Some(x.extract::<Vec<f64>>(py)?)
        } else {
            None
        };
        let solv = BfgsJacSolver::new(self.size);
        let (mat, x0) = solv.solve_for_unitary(&circ, &constant_gates, &unitary, x0_rust);
        Ok((
            PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned(),
            PyArray1::from_vec(py, x0).to_owned(),
        ))
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, PyTuple::empty(py).into_py(py)))
    }
}

#[cfg(all(feature = "python", feature = "rustopt"))]
#[pyclass(name = "LeastSquares_Jac_SolverNative", module = "qsrs")]
struct PyLeastSquaresJacSolver {
    #[pyo3(get)]
    distance_metric: String,
    num_threads: usize,
    ftol: f64,
    gtol: f64,
}

#[cfg(all(feature = "python", feature = "rustopt"))]
#[pymethods]
impl PyLeastSquaresJacSolver {
    #[new]
    fn new(num_threads: Option<usize>, ftol: Option<f64>, gtol: Option<f64>) -> Self {
        let threads = if let Some(threads) = num_threads {
            threads
        } else {
            1
        };
        let ftol = if let Some(ftol) = ftol {
            ftol
        } else {
            5e-16
        };
        let gtol = if let Some(gtol) = gtol {
            gtol
        } else {
            1e-15
        };
        Self {
            distance_metric: String::from("Residuals"),
            num_threads: threads,
            ftol,
            gtol,
        }
    }

    fn solve_for_unitary(
        &self,
        py: Python,
        circuit: PyObject,
        options: PyObject,
        x0: Option<PyObject>,
    ) -> PyResult<(Py<PySquareMatrix>, Py<PyArray1<f64>>)> {
        let u = options.getattr(py, "target")?;
        let (circ, constant_gates) = match circuit.extract::<Py<PyGateWrapper>>(py) {
            Ok(c) => {
                let pygate = c.as_ref(py).try_borrow().unwrap();
                (pygate.gate.clone(), pygate.constant_gates.clone())
            }
            Err(_) => {
                let mut constant_gates = Vec::new();
                let gate = object_to_gate(&circuit, &mut constant_gates, py)?;
                (gate, constant_gates)
            }
        };
        let unitary =
            SquareMatrix::from_ndarray(u.extract::<&PySquareMatrix>(py)?.to_owned_array());
        let x0_rust = if let Some(x) = x0 {
            Some(x.extract::<Vec<f64>>(py)?)
        } else {
            None
        };
        let solv = LeastSquaresJacSolver::new(self.num_threads, self.ftol, self.gtol);
        let (mat, x0) = solv.solve_for_unitary(&circ, &constant_gates, &unitary, x0_rust);
        Ok((
            PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned(),
            PyArray1::from_vec(py, x0).to_owned(),
        ))
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let num_threads = PyTuple::new(py, &[slf.num_threads]).into_py(py);
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, num_threads))
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn native_from_object(obj: PyObject, py: Python) -> PyResult<Py<PyGateWrapper>> {
    let mut constant_gates = Vec::new();
    let gate = object_to_gate(&obj, &mut constant_gates, py)?;
    Py::new(
        py,
        PyGateWrapper {
            dits: gate.dits(),
            gate,
            constant_gates,
        },
    )
}

#[cfg(feature = "python")]
#[pymodule]
fn qsrs(_py: Python, m: &PyModule) -> PyResult<()> {
    install();
    #[pyfn(m)]
    #[pyo3(name = "matrix_distance_squared")]
    fn matrix_distance_squared_py(a: &PySquareMatrix, b: &PySquareMatrix) -> f64 {
        matrix_distance_squared(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
        )
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_distance_squared_jac")]
    fn matrix_distance_squared_jac_py(
        a: &PySquareMatrix,
        b: &PySquareMatrix,
        jacs: Vec<&PySquareMatrix>,
    ) -> (f64, Vec<f64>) {
        matrix_distance_squared_jac(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
            jacs.iter()
                .map(|j| SquareMatrix::from_ndarray(j.to_owned_array()))
                .collect(),
        )
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_residuals")]
    fn matrix_residuals_py(
        a: &PySquareMatrix,
        b: &PySquareMatrix,
        eye: &PyArray2<f64>,
    ) -> Vec<f64> {
        matrix_residuals(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
            &eye.to_owned_array(),
        )
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_residuals_jac")]
    fn matrix_residuals_jac_py(
        py: Python,
        u: &PySquareMatrix,
        m: &PySquareMatrix,
        jacs: Vec<&PySquareMatrix>,
    ) -> Py<PyArray2<f64>> {
        let v: Vec<SquareMatrix> = jacs
            .iter()
            .map(|i| SquareMatrix::from_ndarray(i.to_owned_array()))
            .collect();
        PyArray2::from_array(
            py,
            &matrix_residuals_jac(
                &SquareMatrix::from_ndarray(u.to_owned_array()),
                &SquareMatrix::from_ndarray(m.to_owned_array()),
                &v,
            ),
        )
        .to_owned()
    }
    #[pyfn(m)]
    #[pyo3(name = "qft")]
    fn qft_py(py: Python, n: usize) -> Py<PySquareMatrix> {
        PySquareMatrix::from_array(py, &crate::utils::qft(n).into_ndarray()).to_owned()
    }
    m.add_wrapped(wrap_pyfunction!(native_from_object))?;
    m.add_class::<PyGateWrapper>()?;
    #[cfg(all(feature = "python", feature = "rustopt"))]
    m.add_class::<PyBfgsJacSolver>()?;
    #[cfg(all(feature = "python", feature = "rustopt"))]
    m.add_class::<PyLeastSquaresJacSolver>()?;
    Ok(())
}
