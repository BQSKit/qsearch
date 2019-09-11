use ndarray::Array2;
use rblas::attribute::Transpose;
use rblas::matrix::ops::Gemm;
use rblas::vector::ops::{Dotc, Scal};
use rblas::{Matrix, Vector};

use libc::c_int;
use num_complex::Complex64;

use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::{Div, Mul};

use float_cmp::*;

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

#[derive(Serialize, Deserialize, Clone)]
pub struct ComplexUnitary {
    data: Vec<Complex64>,
    pub size: i32,
}

impl ComplexUnitary {
    pub fn from_size_fn<F>(n: i32, mut f: F) -> Self
    where
        F: FnMut((i32, i32)) -> Complex64,
    {
        let mut s = ComplexUnitary::zeros(n);
        for i in 0..n {
            for j in 0..n {
                s.data[(i * n + j) as usize] = f((i, j));
            }
        }
        s
    }

    pub fn from_vec(v: Vec<Complex64>, size: i32) -> Self {
        ComplexUnitary { data: v, size }
    }

    pub fn zeros(size: i32) -> Self {
        ComplexUnitary {
            data: vec![r!(0.0); (size * size) as usize],
            size,
        }
    }

    pub fn eye(size: i32) -> Self {
        let mut s = Self::zeros(size);
        for i in 0..size {
            s.data[(i * size + i) as usize] = r!(1.0);
        }
        s
    }

    pub fn matmul(&self, other: &ComplexUnitary) -> ComplexUnitary {
        assert_eq!(self.size, other.size);
        let t = Transpose::NoTrans;
        let mut out = ComplexUnitary::zeros(self.size);
        Gemm::gemm(&r!(1.0), t, self, t, other, &r!(0.0), &mut out);
        out
    }

    pub fn kron(&mut self, other: &ComplexUnitary) -> ComplexUnitary {
        let row_a = self.size;
        let row_b = other.size;
        let mut out = ComplexUnitary::zeros(row_a * row_b);

        for i in 0..row_a {
            for j in 0..row_a {
                let row_start = i * row_b;
                let col_start = j * row_b;
                for k in 0..row_b {
                    for l in 0..row_b {
                        //println!("{} * {}; {} * {}; {},{}", i, j, k, l, row_a, row_b);
                        out.data[((row_start + k) * out.size + (col_start + l)) as usize] = self
                            .data[(i * self.size + j) as usize]
                            * other.data[(k * other.size + l) as usize];
                    }
                }
            }
        }
        out
    }

    pub fn into_ndarray(self) -> Array2<Complex64> {
        Array2::from_shape_vec((self.size as usize, self.size as usize), self.data).unwrap()
    }

    pub fn dot(&self, other: &ComplexUnitary) -> Complex64 {
        assert_eq!(self.size, other.size);
        Dotc::dotc(other, self)
    }

    pub fn sum(&self) -> Complex64 {
        self.data.iter().sum()
    }

    pub fn multiply(&self, other: &ComplexUnitary) -> ComplexUnitary {
        assert_eq!(self.size, other.size);
        ComplexUnitary::from_vec(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            self.size,
        )
    }

    pub fn conj(&self) -> ComplexUnitary {
        ComplexUnitary::from_vec(self.data.iter().map(|i| i.conj()).collect(), self.size)
    }
}

impl Mul<Complex64> for ComplexUnitary {
    type Output = Self;
    fn mul(mut self, rhs: Complex64) -> Self {
        Scal::scal_mat(&rhs, &mut self);
        self
    }
}

impl Mul<f64> for ComplexUnitary {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        Scal::scal_mat(&r!(rhs), &mut self);
        self
    }
}

impl Div<Complex64> for ComplexUnitary {
    type Output = Self;
    fn div(mut self, rhs: Complex64) -> Self {
        Scal::scal_mat(&(1.0 / rhs), &mut self);
        self
    }
}

impl Div<f64> for ComplexUnitary {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        Scal::scal_mat(&Complex64::new(1.0 / rhs, 0.0), &mut self);
        self
    }
}

impl Matrix<Complex64> for ComplexUnitary {
    fn rows(&self) -> c_int {
        self.size
    }

    fn cols(&self) -> c_int {
        self.size
    }

    fn as_ptr(&self) -> *const Complex64 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.data.as_mut_ptr()
    }
}

impl Vector<Complex64> for ComplexUnitary {
    fn len(&self) -> i32 {
        self.data.len() as i32
    }

    fn as_ptr(&self) -> *const Complex64 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.data.as_mut_ptr()
    }
}

impl PartialEq for ComplexUnitary {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(s, o)| s.re.approx_eq(o.re, (1e-16, 3)) && s.im.approx_eq(o.im, (1e-16, 3)))
    }
}

impl Eq for ComplexUnitary {}

impl fmt::Debug for ComplexUnitary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ ")?;
        for i in 0..self.size {
            for j in 0..self.size {
                let elem = self.data[(i * self.size + j) as usize];
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
