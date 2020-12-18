![run tests](https://github.com/BQSKit/qsearch/workflows/run%20tests/badge.svg?branch=master)

# qsearch
An implementation of a quantum gate synthesis algorithm based on A* and numerical optimization.  It relies on [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org).  It can export code for [Qiskit](https://qiskit.org) and [OpenQASM](https://github.com/Qiskit/OpenQASM).

This is an implementation of the algorithm described in the paper *[Towards Optimal Topology Aware Quantum Circuit Synthesis](https://ieeexplore.ieee.org/document/9259942)*.

These are some results showing how qsearch can provide optimal or near optimal results. We compare results to the [UniversalQ Compiler](https://github.com/Q-Compiler/UniversalQCompiler).

| Circuit       | # of Qubits | Ref # | CNOT Linear | CNOT Ring | UQ (CNOT Ring) | CNOT Linear Unitary Distance | CNOT Ring Unitary Distance   |
|---------------|--------|-----|-------------|-----------|----------------|-------------------------|-------------------------|
| QFT           | 3      | 6   | 7*          | 6*        | 15             | 1.33 * 10<sup>-14</sup> | 2.22 * 10<sup>-16</sup> |
| Fredkin       | 3      | 8   | 8           | 7         | 9              | 1.76 * 10<sup>-14</sup> | 0.0                     |
| Toffoli       | 3      | 6   | 8           | 6         | 9              | 1.14 * 10<sup>-14</sup> | 0.0                     |
| Peres         | 3      | 5   | 7           | 6         | 19             | 1.13 * 10<sup>-14</sup> | 0.0                     |
| HHL           | 3      | N/A | 3*          | 3*        | 16             | 1.25 * 10<sup>-14</sup> | 0.0                     |
| Or            | 3      | 6   | 8           | 6         | 10             | 1.72 * 10<sup>-14</sup> | 0.0                     |
| EntangledX    | 3      | 4   | 2,3,4       | 2,3,4     | 9              | 1.26 * 10<sup>-14</sup> | 0.0                     |
| TFIM_3_3      | 3      | 4   | 4           | 4         | 17             | 0.0                     | 0.0                     |
| TFIM_6_3      | 3      | 8   | 6           | 6         | 17             | 4.44 * 10<sup>-16</sup> | 0.0                     |
| TFIM_42_3     | 3      | 56  | 6           | 6         | 17             | 8.88 * 10<sup>-16</sup> | 0.0                     |
| TFIM_60_3     | 3      | 80  | 6           | 6         | 17             | 6.66 * 10<sup>-16</sup> | 0.0                     |
| QFT           | 4      | N/A | 13*          |           | 89             | 6.66 * 10<sup>-16</sup> |                         |
| TFIM_30_4     | 4      | 60  | 11          |           | 87             | 9.08 * 10<sup>-11</sup> |                         |
| IBM Challenge | 4      | N/A | 4           |           | DNR            | 0.0                     |                         |

\* Some gates occasionally resulted in circuits with different CNOT counts due to the optimizers getting stuck in local minima. The best run out of 10 is listed in these cases. The CNOT count for these circuits was occasionally 1 more than listed. The gate "EntangledX" is a parameterized gate, and for certain combinations of parameters we were able to produce solutions with fewer CNOTs than the hand-optimized general solution.

# Installation
This is a python package which can be installed using pip.  You will need a Python version of at least 3.6. The qsearch compiler currently runs on macOS, Linux (including [the Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)) and Windows (performance is much worse on Windows). You can install it from [PyPi](https://pypi.org) using:
```
pip3 install qsearch
```
You can also install from the git repository:
```
pip3 install https://github.com/BQSKit/qsearch/archive/dev.zip
```
or download and install it:
```
git clone https://github.com/BQSKit/qsearch
pip3 install --upgrade ./qsearch
```
If you make changes to your local copy, you can reinstall the package:
```
pip3 install --upgrade ./qsearch
```


Once installed, you can import the library like any other python package:
```
import qsearch
```
# Getting Started: qsearch Projects
The simplest way to use the qsearch library is by using a project. When you create a project, you provide a path where a directory will be created to contain the project's files.
```
import qsearch
myproject = qsearch.Project("desired/path/to/project/directory")
```
You can then add unitaries to compile, and set compiler properties. Unitary matrices should be provided as `numpy` ndarrays using `dtype="complex128"`.
```
myproject.add_compilation("gate_name", gate_unitary)
myproject["compiler_option"] = value
```
Once your project is configured, you can start your project by calling `run()`. The compiler uses an automatic checkpointing system, so if it is killed while in-progress, it can be resumed by calling `run()` again.
```
myproject.run()
```
Once your project is finished, you can get OpenQASM output:
```
myproject.assemble("gate_name") # This will return a string of OpenQASM
myproject.assemble("gate_name", write_location="path/to/output/file") # This will write the qasm to the specified path.
```

# Compiling Without Projects
If you would like to avoid working with Projects, you can use the `SearchCompiler` class directly.
```
import qsearch
compiler = qsearch.SearchCompiler()
result = compiler.compile(target=target_unitary)
```
The `SearchCompiler` class and the `compile` function can take extra arguments to further configure the compiler.  The returned value is a dictionary that contains the unitary that represents the implemented circuit, the `qsearch.gates.Gate` representation of the circuit structure, and the vector of parameters for the circuit structure.

# A Note On Endianness
We use the physics convention of using big endian when naming our qubits.  Some quantum programs, including IBM's Qiskit, use little endian.  This means you will need to reverse the endianness of a unitary designed for Qiskit in order to work with our code, or visa versa.  We provide a function that performs endian reversal on numpy matrices:
```
little_endian = qsearch.utils.endian_reverse(big_endian) # you can use the same function to convert in the other direction as well
```

# Documentation and Examples

The documentation and API reference can be found [on readthedocs](https://qsearch.readthedocs.io/en/latest/).

Also check out the [examples](https://github.com/BQSKit/qsearch/tree/master/examples)!

# Legal/Copyright information

Please read our [LICENSE](https://github.com/BQSKit/qsearch/blob/master/LICENSE)
