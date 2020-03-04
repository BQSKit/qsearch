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

use smallvec::{smallvec, SmallVec};

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
    data: SmallVec<[Complex64; 32]>,
    pub size: usize,
}

impl ComplexUnitary {
    pub fn from_ndarray(arr: Array2<Complex64>) -> Self {
        let size = arr.ncols();
        ComplexUnitary::from_vec(arr.into_raw_vec(), size)
    }

    pub fn from_size_fn<F>(n: usize, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> Complex64,
    {
        let mut s = ComplexUnitary::zeros(n);
        for i in 0..n {
            for j in 0..n {
                s.data[(i * n + j) as usize] = f((i, j));
            }
        }
        s
    }

    pub fn from_vec(v: Vec<Complex64>, size: usize) -> Self {
        ComplexUnitary {
            data: SmallVec::from_vec(v),
            size,
        }
    }

    pub fn zeros(size: usize) -> Self {
        ComplexUnitary {
            data: smallvec![r!(0.0); (size * size) as usize],
            size,
        }
    }

    pub fn eye(size: usize) -> Self {
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
        Array2::from_shape_vec(
            (self.size as usize, self.size as usize),
            self.data.into_vec(),
        )
        .unwrap()
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

    #[allow(non_snake_case)]
    pub fn T(&self) -> ComplexUnitary {
        let mut out = ComplexUnitary::zeros(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                out.data[i * self.size + j] = self.data[j * self.size + i];
                out.data[j * self.size + i] = self.data[i * self.size + j];
            }
        }
        out
    }

    #[allow(non_snake_case)]
    pub fn H(&self) -> ComplexUnitary {
        let mut out = ComplexUnitary::zeros(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                out.data[i * self.size + j] = self.data[j * self.size + i].conj();
                out.data[j * self.size + i] = self.data[i * self.size + j].conj();
            }
        }
        out
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
        self.size as c_int
    }

    fn cols(&self) -> c_int {
        self.size as c_int
    }

    fn as_ptr(&self) -> *const Complex64 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.data.as_mut_ptr()
    }
}

impl Vector<Complex64> for ComplexUnitary {
    fn len(&self) -> c_int {
        self.data.len() as c_int
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
