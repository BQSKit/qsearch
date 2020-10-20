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

6. X, Y, Z

7. Any constant, unparameterized gate such as CNOT

This list will likely grow as needed.

## Building and Installing

If you are on a 64 bit macOS, Windows, or Linux, you can install via `pip3 install qsrs`.

You can also install from source.

### Linux

Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

First, install the dependencies. On modern Debian based machines you should be able to install the dependencies like the following,
note that the version of libgfortran-10-dev may be different.

```
sudo apt install libopenblas-dev libceres-dev libgfortran-10-dev
```

Once that is complete you should then install Rust like as follows:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Accept all of the default prompts. You probably want to `source ~/.cargo/env`, then switch to the nightly toolchain:

```
rustup default nightly
```

Then you can enter the `qsrs` directory and run

```
pip install .
```

This will take a while. Once it is done, verify that the installation succeeded by running

```
python3 -c 'import qsrs'
```

It should not print anything out nor give any error.


On Linux, you can also build wheels with Docker, staticly linking a custom built `openblas`.
Incidently, this is how the packages on PyPi are built.

1. Install docker https://docs.docker.com/install/
2. Run the container to build wheels: `docker run --rm -v $(pwd):/io ethanhs/maturin-manylinux-2010:0.6 build --cargo-extra-args="--no-default-features --features python,static,rustopt" --release --manylinux 2010 --no-sdist`
3. Install the correct wheel for your Python version in `target/wheels` (e.g. `qsrs-0.13.0-cp37-cp37m-manylinux2010_x86_64.whl` for Python 3.7)


### MacOS


Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

First, install the dependencies. We use homebrew here, which is what we build the official package against.

```
brew install gcc ceres-solver
```

Once that is complete you should then install Rust like as follows:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Accept all of the default prompts. You probably want to `source ~/.cargo/env`, then switch to the nightly toolchain:

```
rustup default nightly
```

Then you can enter the `qsrs` directory and run the following to build a wheel linking to Apple's Accelerate

```
maturin build -i python --cargo-extra-args="--no-default-features --features=accelerate,python,rustopt,ceres/system" --release --no-sdist
```

This will take a while. Once it is done, install the wheel like

```
pip3 install target/wheels/qsrs-0.15.1-cp38-cp38-macosx_10_7_x86_64.whl
```

Note the name of the wheel may be slightly different depending on your Python version, but it should be the only wheel in the `target/wheels/` folder.

Verify that the installation succeeded by running

```
python3 -c 'import qsrs'
```

It should not print anything out nor give any error.

### Windows


Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

Download and install rust via the installer found at https://rustup.rs/. Accept all of the defaults.

Close your shell and open a new one (this updates the enviroment). Then run

```
rustup default nightly
```

Then install `cargo-vcpkg`, which will help install dependencies for us. You can install it via

```
cargo install cargo-vcpkg
```

`cargo` is installed with Rust so this should work.

Then in the qsrs folder, run

```
cargo vcpkg build
```

This will likely take a while.

Once it is done, build a wheel via 

```
maturin build -i python --cargo-extra-args="--no-default-features --features python,static,rustopt" --release --no-sdist
```

Then you should be able to install the generated wheel package like

```
pip install -U target\wheels\qsrs-0.15.1-cp37-none-win_amd64.whl
```

Note the name of the wheel may be slightly different depending on your Python version, but it should be the only wheel in the `target/wheels/` folder.

Verify that the installation succeeded by running

```
python3 -c 'import qsrs'
```

It should not print anything out nor give any error.

## Usage

Please see the [`qsearch` wiki](https://github.com/WolfLink/qsearch/wiki/Native-Gateset)
for how to use this with qsearch.
`

## Testing and Contributing

To run the tests, clone the repo then run:

```
$ pip install .
$ pip install -r ../test-requirements.txt
$ cd .. && pytest
```

## License

Since NLOpt's L-BFGS implementation is LGPL licensed, if the `rustopt` feature is enabled
(it is enabled by default), qsrs is LGPL licensed as it statically links to NLOpt.
If you disable the `rustopt` feature, then the combined work is BSD 3 clause licensed.

In addition, if you disable `rustopt`, you can still enable the `ceres` feature for a BSD-3
licensed product, but since this crate is usually dynamically linked in the end, it doesn't
really matter.
Please see LICENSE for more information.
