# search_compiler
A compiler for quantum computers based on A* and numerical optimization.


# Getting Started - search_compiler Projects
The simplest way to use the search_compiler library is by using a Project. When you create a project, you provide a path where a directory will be created to contain the project's files.
```
import search_compiler as sc
myproject = sc.Project("desired/path/to/project/directory")
```
You can then add unitaries to compile, and set compiler properties. Unitary matrices should be provided as `numpy` matrices using `dtype="complex128"`. Details on compiler options are provided in a later section.
```
myproject.add_compilation("gate_name", gate_matrix)
myproject["compiler_option"] = value
```
Once your project is configured, you can start your project by calling `run()`. The compiler uses an automatic checkpoint system, so if it is killed while in-progress, it can be resumed by calling `run()` again.
```
myproject.run()
```
Once your project is finished, you can get openqasm output:
```
myproject.assemble("gate_name") # This will write the qasm to stdout
myproject.assemble("gate_name", write_location="path/to/output/file") # This will write the qasm to the specified path.
```

## Advanced Project Features
There are a few more features that the Project class provides that might be needed. You may need to remove or reset compilations.
```
myproject.remove_compilation("gate_name") # deletes all data relating to the specified gate and removes it from the project
myproject.reset() # deletes all in-progress or completed compilation data in the project. You may need this if you decide to tweak compiler parameters.
myproject.reset("gate_name") # deletes all in-progress or completed compilation data for the specified gate. You may need this if the compiler does not always find the same solution for your gate.
myproject.clear() # deletes all data from the project, putting it in the same state as a fresh project.
```
You can also get output in the form of `search_compiler.circuits.QuantumStep` objects. More detail on how to use these objects below.
```
# circuit is a QuantumStep object, and vector is a numpy array of floats, to be used by certain QuantumStep functions
circuit, vector = myproject.get_result("gate_name")
```
You can use 
```
Document status checking stuff after you rewrite it.
```
## Project Compiler Options
You can configure options that the compiler used by a project using `myproject["config_key"] = config_value`. The supported keys are described here.
- **`threshold`** is a `float` that defines the termination condition of the compilation. The compiler will return when it finds a circuit with a `error_func` value less than this threshold. The default value is `1e-10`.
- **`gateset`** is a `search_compiler.gatesets.Gateset` object that is used by the compiler. The default value is `search_compiler.gatesets.QubitCNOTLinear()`.
- **`error_func`** is a distance function that compares two `numpy.matrix` objects. It must return `float` values that greater than or equal to zero such that input matrices that are close to the same will result in outputs close to zero. The default value is `search_compiler.utils.matrix_distance_squared`.
- **`search_type`** is a `string` that can be set to `"breadth"` to perform a breadth-first search, or `"greedy"` to perform a greedy search using only `error_func`. When set to any other value, including the default, astar search is performed using `heuristic`.
- **`heuristic`** is a function that takes a value from `error_func` and a search depth, and returns a `float`. It is used to order the priority queue used for searching. Setting this option overrides `search_type`.
- **`solver`** is a solver object as defined in `search_compiler.solver`. It is used to set the numerical optimizer used to solve for circuit parameters. The default is `search_compiler.solver.COBYLA_Solver()`.
- **`beams`** can be sets the number of nodes popped off of the priority queue during each search layer. The default is `1`. Setting a higher value will cause the compiler to examine multiple search paths in parallel, and may result in faster runtimes. Setting a negative value will have the compiler calculate a number of beams to maximize CPU utilization. This does not always result in a speedup, so caution is advised when adjusting this value.

# Gatesets
Several gatesets are supported, and a framework to implement custom gatesets is provided. The `search_compiler.gatesets.Gateset` class implements two functions:
 - `initial_layer(self, dits)` takes a number of qudits and generates a `search_compiler.circuits.QuantumStep` object that represents an initial layer for the search, usually a single qudit gate for each qudit.
 - `search_layers(self, dits)` takes a number of qudits and generates a `list` of `search_compiler.circuits.QuantumStep` objects that represents the possible branches the search can take at each step, usually one for each possible placement of the desired 2-qudit gate, followed by single-qudit gates after the 2-qudit gate.
 
 There are several built-in gatesets.
 - `QubitCNOTLinear` is the default gateset. It represents hardware that supports arbitrary single-qubit gates and CNOT gates between adjacent qubits, arranged in a line.
 - `QubitCNOTAdjacencyList` allows for targeting of hardware with different CNOT adjacency. Adjacency is described via a `list` of `tuples` of `int` numbers, in the form of `(control, target)`. Note that it is generally not necessary to add CNOT gates in both directions, because flipping direction can be performed via single-qubit gates.
 
## Experimental Native Gateset

There is an *experimental* gateset that is implemented in native code to be faster. It implements a `QubitCNOTLinear` gateset. To use it, you must build and install the native library located in the `native/` directory. Below are the steps for doing so:

1. Install openblas. On Debian based distros this is `libopenblas-dev`. On MacOS you can `brew install openblas`.

2. Install rustup via https://rustup.rs. Make sure to choose a nightly toolchain. THe defaults for platform should be fine.

3. Install maturin with `cargo install maturin`.

4. In the `native/` directory, run `RUSTFLAGS="-Ctarget-cpu=native" maturin build --release -i python3 --manylinux 1-unchecked`. This should build a   wheel for your Python executable in `target/wheels/` directory. This will take a while. Then `pip install target/wheels/scrs-0.1.0-<your platform tag>.whl`, the wheel generated by `maturin`.

5. Done! You should now be able to use the `QubitCNOTLinearNative` gateset from `search_compiler.gatesets`.

 # Circuits
 `search_compiler.circuits.QuantumStep` objects descibe sequences of quantum gates as a graph of objects, in terms of the matrix multiplications necessary to describe the circuit. There are several proivded `QuantumStep` objects, as well as a framework to implement custom gates.
 - 
