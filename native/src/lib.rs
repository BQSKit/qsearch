use ndarray::Array2;
use num_complex::Complex64;

use numpy::PyArray2;
use pyo3::prelude::*;

pub mod circuits;
//pub mod compiler;
pub mod gatesets;
pub mod solver;
pub mod utils;

pub type ComplexUnitary = Array2<Complex64>;

use gatesets::GateSetLinearCNOT;
use circuits::GateProduct;


#[pyclass(name=QubitCNOTLinearNative)]
struct PyGateSetLinearCNOT {
    gateset: GateSetLinearCNOT,
}

#[pymethods]
impl PyGateSetLinearCNOT {
    #[new]
    pub fn new(obj: &PyRawObject) {
        obj.init( PyGateSetLinearCNOT {
            gateset: GateSetLinearCNOT::new(),
        });
    }

    pub fn 
}


#[pymodule]
fn search_compiler(_py: Python, m: &PyModule) -> PyResult<()> {
    
    Ok(())
}

