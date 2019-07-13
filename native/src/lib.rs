use ndarray::Array2;
use num_complex::Complex64;

pub mod circuits;
//pub mod compiler;
pub mod gatesets;
pub mod solver;
pub mod utils;

pub type ComplexUnitary = Array2<Complex64>;
