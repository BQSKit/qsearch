# qsrs: qsearch Rust implementation

This is a (partial) implementation of the algorithm described in the paper
*[Heuristics for Quantum Compiling with a Continuous Gate Set](https://arxiv.org/abs/1912.02727)*.
However, re-writing in Rust has provided up to a 20x speedup in compilation time.
At some future point, this project will be usable as a stand-alone Rust program that will be able
to compile quantum circuits.

There are Python bindings to the Rust code that are used in the qsearch package. These
should automatically be installed from PyPi when you install qsearch.


## Installing

`qsrs` is available on PyPi:

```
pip3 install qsrs
```

If you would like to build from source see "Building and Installing" below.


## Supported gates

The current list of "gates" supported is:

1. Identity

2. X90ZX90Z (combined for performance reasons)

3. IBM U3

4. Kronecker

5. Product

6. Any constant, unparameterized gate such as CNOT

This list will likely grow as needed.

## Building and Installing

If you are on a 64 bit macOS, Windows, or Linux, you can install via `pip3 install qsrs`.

On Linux or MacOS, you can also build and install qsrs from source using a system installation of `blas` or staticly link one in.

1. Install the blas library if needed. If you want to use Apple Accelerate, it should come with macOS. For openblas,
   on Debian based Linux distros you will need to install `libopenblas-dev`. You will also need
   the `libgfortran-<version>-dev` and `gfortran`  packages (for whatever version your distro supplies). For openblas on macOS, you
   can install these via Homebrew as `brew install openblas gcc`.
2. (Optional) Install ceres-solver for the native LeastSquares optimizer. Experiments have shown it to be 3-20x faster. On
   Debian based Linux, you can install the `libceres-dev` package. On macOS, you can install it via homebrew:
   `brew install ceres-solver`.
3. Install rustup via https://rustup.rs. The defaults for platform should be fine.
4. Switch to the nightly toolchain using `rustup default nightly`. You may need to do `source ~/.cargo/env` first.
5. If you are not using openblas, uncomment the corresponding line in the `pyproject.toml` file for Accelerate or MKL.
   You should then be able to run `pip install .` Make sure the version of pip you have is at least 20.0.2.
   You can check via `pip -V`.
6. Verify your install is working by running `python3 -c 'import qsearch_rs'`
   Done! You should now be able to use the `Native` backend in `qsearch`.

On Linux, you can also build wheels with Docker, staticly linking a custom built `openblas`.
Incidently, this is how the packages on PyPi are built.

1. Install docker https://docs.docker.com/install/
3. Run the container to build wheels: `docker run --rm -v $(pwd):/io ethanhs/maturin-manylinux-2010:0.5 build --cargo-extra-args="--no-default-features --features python,static,rustopt" --release --manylinux 2010 --no-sdist`
4. Install the correct wheel for your Python version in `target/wheels` (e.g. `qsrs-0.13.0-cp37-cp37m-manylinux2010_x86_64.whl` for Python 3.7)

On Windows, you can build from source and statically link to openblas:
1. Install rustup via https://rustup.rs. The defaults for platform `should be fine.
2. Switch to the nightly toolchain using `rustup default nightly`. You should close your shell and open a new one.
3. Install `cargo-vcpkg`, via `cargo install cargo-vcpkg` which will install the native dependencies for us.
4. Run `cargo vcpkg build`, which will build and make the native dependencies available for us to use.
5. Build a wheel via `maturin build -i python --cargo-extra-args="--no-default-features --features python,static,rustopt" --release --no-sdist`
6. Install the correct wheel for your Python version in `target/wheels` (e.g. `qsrs-0.13.0-cp37-none-win_amd64.whl` for Python 3.7)

## Usage

Please see the [`qsearch` wiki](https://github.com/WolfLink/qsearch/wiki/Native-Gateset)
for how to use this with qsearch.
`

## Testing and Contributing

To run the tests, clone the repo then run:

```
$ pip install .
$ pip install -r test-requirements.txt
$ pytest
```

## License

Since NLOpt's L-BFGS implementation is LGPL licensed, if the `rustopt` feature is enabled
(it is enabled by default), qsrs is LGPL licensed as it statically links to NLOpt.
If you disable the `rustopt` feature, then the combined work is BSD 3 clause licensed.

In addition, if you disable `rustopt`, you can still enable the `ceres` feature for a BSD-3
licensed product, but since this crate is usually dynamically linked in the end, it doesn't
really matter.
Please see LICENSE for more information.