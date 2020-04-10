use cblas::*;
use ndarray::Array2;

use num_complex::Complex64;

use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::{Div, Mul};

use float_cmp::*;

extern crate cblas;
#[cfg(all(feature = "openblas-static", feature = "openblas-system"))]
extern crate openblas_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

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
pub struct SquareMatrix {
    data: Vec<Complex64>,
    pub size: usize,
}

impl SquareMatrix {
    pub fn from_ndarray(arr: Array2<Complex64>) -> Self {
        let size = arr.ncols();
        SquareMatrix::from_vec(arr.into_raw_vec(), size)
    }

    pub fn from_size_fn<F>(n: usize, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> Complex64,
    {
        let mut s = SquareMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                s.data[(i * n + j) as usize] = f((i, j));
            }
        }
        s
    }

    pub fn from_vec(v: Vec<Complex64>, size: usize) -> Self {
        SquareMatrix { data: v, size }
    }

    pub fn zeros(size: usize) -> Self {
        SquareMatrix {
            data: vec![r!(0.0); (size * size) as usize],
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
                &self.data,
                self.size as i32,
                &other.data,
                other.size as i32,
                r!(0.0),
                &mut out.data,
                out.size as i32,
            )
        };
        out
    }

    pub fn kron(&mut self, other: &SquareMatrix) -> SquareMatrix {
        let row_a = self.size;
        let row_b = other.size;
        let mut out = SquareMatrix::zeros(row_a * row_b);

        for i in 0..row_a {
            for j in 0..row_a {
                let row_start = i * row_b;
                let col_start = j * row_b;
                for k in 0..row_b {
                    for l in 0..row_b {
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

    pub fn dot(&self, other: &SquareMatrix) -> Complex64 {
        assert_eq!(self.size, other.size);
        let mut res = [r!(0.0)];
        unsafe { zdotc_sub(self.size as i32, &self.data, 1, &other.data, 1, &mut res) };
        res[0]
    }

    pub fn sum(&self) -> Complex64 {
        self.data.iter().sum()
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
                out.data[i * self.size + j] = self.data[j * self.size + i];
                out.data[j * self.size + i] = self.data[i * self.size + j];
            }
        }
        out
    }

    #[allow(non_snake_case)]
    pub fn H(&self) -> SquareMatrix {
        let mut out = SquareMatrix::zeros(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                out.data[i * self.size + j] = self.data[j * self.size + i].conj();
                out.data[j * self.size + i] = self.data[i * self.size + j].conj();
            }
        }
        out
    }
}

impl Mul<Complex64> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, rhs: Complex64) -> Self {
        unsafe { zscal(self.size as i32, rhs, &mut self.data, 1) };
        self
    }
}

impl Mul<f64> for SquareMatrix {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self {
        unsafe { zscal(self.size as i32, r!(rhs), &mut self.data, 1) };
        self
    }
}

impl Div<Complex64> for SquareMatrix {
    type Output = Self;
    fn div(mut self, rhs: Complex64) -> Self {
        unsafe { zscal(self.size as i32, 1f64 / rhs, &mut self.data, 1) };
        self
    }
}

impl Div<f64> for SquareMatrix {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self {
        unsafe {
            zscal(
                self.size as i32,
                Complex64::new(1.0 / rhs, 0.0),
                &mut self.data,
                1,
            )
        };
        self
    }
}

impl PartialEq for SquareMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(s, o)| s.re.approx_eq(o.re, (1e-16, 3)) && s.im.approx_eq(o.im, (1e-16, 3)))
    }
}

impl Eq for SquareMatrix {}

impl fmt::Debug for SquareMatrix {
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
