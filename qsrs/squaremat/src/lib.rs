use cblas::*;
use ndarray::Array2;

use num_complex::Complex64;

use std::fmt;
use std::ops::{Div, Mul, Sub};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
extern crate cblas;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(any(feature = "openblas-static", feature = "openblas-system"))]
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

fn kron(a: &[Complex64], row_a: usize, b: &[Complex64], row_b: usize, out: &mut [Complex64]) {
    let a_data = &a[..row_a * row_a];
    let b_data = &b[..row_b * row_b];
    let out_data = &mut out[..row_a * row_b * row_a * row_b];
    for i in 0..row_a {
        for j in 0..row_a {
            let row_start = i * row_b;
            let col_start = j * row_b;
            for k in 0..row_b {
                for l in 0..row_b {
                    out_data[(row_start + k) * row_a * row_b + col_start + l] =
                        a_data[i * row_a + j] * b_data[k * row_b + l];
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct SquareMatrix {
    data: Array2<Complex64>,
    pub size: usize,
}

impl SquareMatrix {
    pub fn from_ndarray(arr: Array2<Complex64>) -> Self {
        let size = arr.ncols();
        SquareMatrix { data: arr, size }
    }

    pub fn from_size_fn<F>(n: usize, f: F) -> Self
    where
        F: FnMut((usize, usize)) -> Complex64,
    {
        SquareMatrix {
            data: Array2::from_shape_fn((n, n), f),
            size: n,
        }
    }

    pub fn from_vec(v: Vec<Complex64>, size: usize) -> Self {
        SquareMatrix {
            data: Array2::from_shape_vec((size, size), v).unwrap(),
            size,
        }
    }

    pub fn zeros(size: usize) -> Self {
        SquareMatrix {
            data: Array2::zeros((size, size)),
            size,
        }
    }

    pub fn eye(size: usize) -> Self {
        SquareMatrix {
            data: Array2::eye(size),
            size,
        }
    }

    pub fn matmul(&self, other: &SquareMatrix) -> SquareMatrix {
        assert_eq!(self.size, other.size);
        let t = Transpose::None;
        let mut out = SquareMatrix::zeros(self.size);
        unsafe {
            zgemm(
                Layout::RowMajor,
                t,
                t,
                self.size as i32,
                self.size as i32,
                self.size as i32,
                r!(1.0),
                self.data.as_slice().unwrap(),
                self.size as i32,
                &other.data.as_slice().unwrap(),
                other.size as i32,
                r!(0.0),
                out.data.as_slice_mut().unwrap(),
                out.size as i32,
            )
        };
        out
    }

    pub fn kron(&mut self, other: &SquareMatrix) -> SquareMatrix {
        let row_a = self.size;
        let row_b = other.size;
        let mut out = SquareMatrix::zeros(row_a * row_b);
        kron(
            self.data.as_slice().unwrap(),
            row_a,
            other.data.as_slice().unwrap(),
            row_b,
            out.data.as_slice_mut().unwrap(),
        );
        out
    }

    pub fn into_ndarray(self) -> Array2<Complex64> {
        self.data
    }

    pub fn dot(&self, other: &SquareMatrix) -> Complex64 {
        assert_eq!(self.size, other.size);
        let mut res = [r!(0.0)];
        unsafe {
            zdotc_sub(
                self.size as i32,
                self.data.as_slice().unwrap(),
                1,
                other.data.as_slice().unwrap(),
                1,
                &mut res,
            )
        };
        res[0]
    }

    pub fn sum(&self) -> Complex64 {
        self.data.sum()
    }

    pub fn multiply(&self, other: &SquareMatrix) -> SquareMatrix {
        assert_eq!(self.size, other.size);
        SquareMatrix::from_vec(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            self.size,
        )
    }

    pub fn conj(&self) -> SquareMatrix {
        SquareMatrix::from_vec(self.data.iter().map(|i| i.conj()).collect(), self.size)
    }

    #[allow(non_snake_case)]
    pub fn T(&self) -> SquareMatrix {
        let mut out = SquareMatrix::zeros(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                out.data[[i, j]] = self.data[[j, i]];
                out.data[[j, i]] = self.data[[i, j]];
            }
        }
        out
    }

    #[allow(non_snake_case)]
    pub fn H(&self) -> SquareMatrix {
        let mut out = SquareMatrix::zeros(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                out.data[[i, j]] = self.data[[j, i]].conj();
                out.data[[j, i]] = self.data[[i, j]].conj();
            }
        }
        out
    }

    pub fn split_complex(&self) -> (Array2<f64>, Array2<f64>) {
        let mut rev = Vec::with_capacity(self.size * self.size);
        let mut imv = Vec::with_capacity(self.size * self.size);
        for elem in self.data.iter() {
            rev.push(elem.re);
            imv.push(elem.im);
        }
        (
            Array2::from_shape_vec((self.size, self.size), rev).unwrap(),
            Array2::from_shape_vec((self.size, self.size), imv).unwrap(),
        )
    }
}

impl Sub<SquareMatrix> for SquareMatrix {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        SquareMatrix::from_ndarray(self.data - other.data)
    }
}

impl Mul<Complex64> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, rhs: Complex64) -> Self {
        unsafe {
            zscal(
                (self.size * self.size) as i32,
                rhs,
                self.data.as_slice_mut().unwrap(),
                1,
            )
        };
        self
    }
}

impl Mul<f64> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        unsafe {
            zscal(
                (self.size * self.size) as i32,
                r!(rhs),
                self.data.as_slice_mut().unwrap(),
                1,
            )
        };
        self
    }
}

impl Div<Complex64> for SquareMatrix {
    type Output = Self;
    fn div(mut self, rhs: Complex64) -> Self {
        unsafe {
            zscal(
                (self.size * self.size) as i32,
                1f64 / rhs,
                self.data.as_slice_mut().unwrap(),
                1,
            )
        };
        self
    }
}

impl Div<f64> for SquareMatrix {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        unsafe {
            zscal(
                (self.size * self.size) as i32,
                Complex64::new(1.0 / rhs, 0.0),
                self.data.as_slice_mut().unwrap(),
                1,
            )
        };
        self
    }
}

impl PartialEq for SquareMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.data == other.data
    }
}

impl Eq for SquareMatrix {}

impl fmt::Debug for SquareMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ ")?;
        for i in 0..self.size {
            for j in 0..self.size {
                let elem = self.data[[i, j]];
                if elem.re == 0.0 {
                    write!(f, "{:?}i, ", elem.im)?;
                } else if elem.im == 0.0 {
                    write!(f, "{:?}, ", elem.re)?;
                } else {
                    write!(f, "{:?}+{:?}i, ", elem.re, elem.im)?;
                }
            }
            if i + 1 == self.size {
                write!(f, "]")?;
                break;
            }
            write!(f, "\n  ")?;
        }
        Ok(())
    }
}
