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

On Linux or MacOS, you can build and install scrs using a system installation of `openblas`.

1. Install openblas and gfortran. On Debian based Linux distros this is `libopenblas-dev`. You will also need
   the `gfortran` package. On macOS, you can install these via Homebrew as `brew install openblas gcc`.
2. Install rustup via https://rustup.rs. The defaults for platform should be fine.
3. Switch to the nightly toolchain using `rustup default nightly`. You may need to do `source ~/.cargo/env` first.
4. In the `native/` directory run `pip install .` Make sure the version of pip you have is at least 20.0.2. You can check via `pip -V`. 
5. Done! You should now be able to use the `COBYLA_SolverNative` solver from `search_compiler.solver`.

On Linux, you can also build wheels with Docker, staticly linking a custom built `openblas`.
Incidently, this is how the packages on PyPi are built.

1. Install docker https://docs.docker.com/install/
3. Run the container to build wheels: `docker run --rm -v $(pwd):/io ethanhs/maturin-manylinux-2010:0.1 build --cargo-extra-args="--no-default-features --features static" --release --manylinux 2010`
4. Install the correct wheel for your Python version in `native/target/wheels` (e.g. `scrs-0.1.0-cp37-cp37m-manylinux2010_x86_64.whl` for Python 3.7)


## Usage
You can use the native gateset in order to achieve significantly faster synthesis speeds.
A "supported gateset" is one that uses only the above "gates". Currently if you use any of the
following solvers with a supported gateset, you should get the speed advantage:

 - `COBYLA_SolverNative`
 - `Jac_SolverNative`

```
import search_compiler as sc
# Use with a project (remember to use the default sc.QubitCNOTLinear() gateset or one that uses only supported gates)
p = sc.Project("myproject")
p["solver"] = sc.solver.COBYLA_SolverNative()

# Use with SearchCompiler directly
compiler = sc.SearchCompiler(solver=sc.solver.COBYLA_SolverNative())
```

## Packaging for release

