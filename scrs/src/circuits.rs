use pyo3::prelude::*;
use pyo3::ffi;
use pyo3::types::PyAny;
use ndarray::arr2;
use num_complex::Complex64;

use numpy::array::{PyArray1, PyArray2};

use crate::utils::{rot_x, rot_y, rot_z, kron};
use crate::{ComplexUnitary, PyComplexUnitary};

use std::mem;

