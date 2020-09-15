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
    Gate, GateCNOT, GateConstantUnitary, GateIdentity, GateKronecker, GateProduct,
    GateSingleQutrit, GateU3, GateXZXZ, QuantumGate,
};

#[cfg(feature = "python")]
use utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

#[cfg(feature = "python")]
fn gate_to_object(
    gate: &Gate,
    py: Python,
    constant_gates: &[SquareMatrix],
    circuits: &PyModule,
) -> PyResult<PyObject> {
    Ok(match gate {
        Gate::CNOT(..) => {
            let gate: PyObject = circuits.get("CNOTStep")?.extract()?;
            gate.call0(py)?
        }
        Gate::Identity(id) => {
            let gate: PyObject = circuits.get("IdentityStep")?.extract()?;
            let args = PyTuple::new(
                py,
                vec![constant_gates[id.index].size, id.data.dits as usize],
            );
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
                .map(|i| gate_to_object(i, py, &constant_gates, circuits).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::Product(prod) => {
            let gate: PyObject = circuits.get("ProductStep")?.extract()?;
            let steps: Vec<PyObject> = prod
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, &constant_gates, circuits).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::SingleQutrit(..) => {
            let gate: PyObject = circuits.get("SingleQutritStep")?.extract()?;
            gate.call0(py)?
        }
        _ => unreachable!(),
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
        "CNOTStep" => {
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
        "IdentityStep" => {
            let index = constant_gates.len();
            let n = obj.getattr(py, "_n")?.extract(py)?;
            constant_gates.push(SquareMatrix::eye(n));
            Ok(GateIdentity::new(index).into())
        }
        "QiskitU3QubitStep" => Ok(GateU3::new().into()),
        "XZXZPartialQubitStep" => {
            let index = constant_gates.len();
            constant_gates.push(crate::utils::rot_x(std::f64::consts::PI / 2.0));
            Ok(GateXZXZ::new(index).into())
        }
        "ProductStep" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_substeps")?.extract(py)?;
            let mut steps: Vec<Gate> = Vec::with_capacity(substeps.len());
            for step in substeps {
                steps.push(object_to_gate(&step, constant_gates, py)?);
            }
            Ok(GateProduct::new(steps).into())
        }
        "KroneckerStep" => {
            let substeps: Vec<PyObject> = obj.getattr(py, "_substeps")?.extract(py)?;
            let mut steps: Vec<Gate> = Vec::with_capacity(substeps.len());
            for step in substeps {
                steps.push(object_to_gate(&step, constant_gates, py)?);
            }
            Ok(GateKronecker::new(steps).into())
        }
        "SingleQutritStep" => Ok(GateSingleQutrit::new().into()),
        "Gate" => {
            let g = obj.extract::<Py<PyGateWrapper>>(py)?;
            let wrapper = g.as_ref(py).try_borrow()?;
            Ok(wrapper.gate.clone())
        }
        _ => {
            if obj.getattr(py, "num_inputs")?.extract::<usize>(py)? == 0 {
                let dits = obj.getattr(py, "dits")?.extract::<u8>(py)?;
                let args: Vec<u8> = vec![];
                let pyobj = obj.call_method(py, "matrix", (args,), None)?;
                let pymat = pyobj.extract::<&PyArray2<Complex64>>(py)?;
                let mat = unsafe { pymat.as_array() };
                let index = constant_gates.len();
                constant_gates.push(SquareMatrix::from_ndarray(mat.to_owned()).T());
                Ok(GateConstantUnitary::new(index, dits).into())
            } else {
                Err(exceptions::ValueError::py_err(format!(
                    "Unknown gate {}",
                    name
                )))
            }
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name=Gate, dict, module = "qsrs")]
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
            Gate::XZXZ(..) => String::from("XZXZ"),
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
        let circuits = py.import("qsearch.circuits")?;
        gate_to_object(&self.gate, py, &self.constant_gates, circuits)
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
#[pyclass(name=BFGS_Jac_SolverNative, dict, module = "qsrs")]
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
#[pyclass(name=LeastSquares_Jac_SolverNative, dict, module = "qsrs")]
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
            1e-6 // Ceres documented default
        };
        let gtol = if let Some(gtol) = gtol {
            gtol
        } else {
            1e-10 // Ceres documented default
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
    #[pyfn(m, "matrix_distance_squared")]
    fn matrix_distance_squared_py(a: &PySquareMatrix, b: &PySquareMatrix) -> f64 {
        matrix_distance_squared(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
        )
    }
    #[pyfn(m, "matrix_distance_squared_jac")]
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
    #[pyfn(m, "matrix_residuals")]
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
    #[pyfn(m, "matrix_residuals_jac")]
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
    #[pyfn(m, "qft")]
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
