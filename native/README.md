# Native Gate implementation

There is an experimental gateset that is implemented in native code to be faster. Currently it
implements the `QubitCNOTLinear` gateset. It implements many common gates in native code for the
speed benefit.

The current list of "gates" supported is:

1. Identity

2. X90ZX90Z (combined for performance reasons)

3. IBM U3

4. Kronecker

5. Product

6. Any constant, unparameterized gate such as CNOT

This list will likely grow as needed.

## Building and Installing

If you are on a 64 bit macOS or Linux, you can install via `pip3 install search_compiler[native]`.

On Linux or MacOS, you can also build and install scrs from source using a system installation of `openblas`.

1. Install the blas library if needed. If you want to use Apple Accelerate, it should come with macOS. For openblas,
   on Debian based Linux distros you will need to install `libopenblas-dev`. You will also need
   the `libgfortran-<version>-dev`  package (for whatever version your distro supplies). For openblas on macOS, you
   can install these via Homebrew as `brew install openblas gcc`.
2. Install rustup via https://rustup.rs. The defaults for platform should be fine.
3. Switch to the nightly toolchain using `rustup default nightly`. You may need to do `source ~/.cargo/env` first.
4. If you are not using openblas, uncomment the corresponding line in the `pyproject.toml` file for Accelerate or MKL.
   In the `native/` directory run `pip install .` Make sure the version of pip you have is at least 20.0.2.
   You can check via `pip -V`.
5. Done! You should now be able to use the any of the `*_SolverNative` solvers from `search_compiler.solver`.
   Verify your install is working by running `python3 -c 'import search_compiler_rs'`

On Linux, you can also build wheels with Docker, staticly linking a custom built `openblas`.
Incidently, this is how the packages on PyPi are built.

1. Install docker https://docs.docker.com/install/
3. Run the container to build wheels: `docker run --rm -v $(pwd):/io ethanhs/maturin-manylinux-2010:0.3 build --cargo-extra-args="--no-default-features --features python,static" --release --manylinux 2010 --no-sdist`
4. Install the correct wheel for your Python version in `native/target/wheels` (e.g. `scrs-0.6.0-cp37-cp37m-manylinux2010_x86_64.whl` for Python 3.7)


## Usage
You can use the native gateset in order to achieve significantly faster synthesis speeds.
A "supported gateset" is one that uses only the above "gates". Currently if you use any of the
following solvers with a supported gateset, you should get the speed advantage:

 - `COBYLA_SolverNative`
 - `BFGS_Jac_SolverNative`
 - `LeastSquares_Jac_SolverNative`

```
import search_compiler as sc
# Use with a project (remember to use the default sc.QubitCNOTLinear() gateset or one that uses only supported gates)
p = sc.Project("myproject")
p["solver"] = sc.solver.LeastSquares_Jac_SolverNative()

# Use with SearchCompiler directly
compiler = sc.SearchCompiler(solver=sc.solver.LeastSquares_Jac_SolverNative())
```
