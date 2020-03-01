# search_compiler
An implementation of a quantum gate synthesis algorithm based on A* and numerical optimization.  It relies on [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org).  It can export code for [Qiskit](https://qiskit.org) and [OpenQASM](https://github.com/Qiskit/openqasm).

This is an implementation of the algorithm described in the paper *[Heuristics for Quantum Compiling with a Continuous Gate Set](https://arxiv.org/abs/1912.02727)*.

# Installation
This is a python package which can be installed using pip.  You will need a Python version of at least 3.6.  You can install from [PyPi](https://pypi.org) using:
```
pip3 install search_compiler 
```
You can also install from a downloaded copy of the repository:
```
git clone git@github.com:WolfLink/search_compiler.git
pip3 install ./search_compiler
```
If you make changes to your local copy, you can reinstall the package:
```
pip3 install --upgrade ./search_compiler
```


Once installed, you can import the library like any other python package:
```
import search_compiler as sc
```
### Experimental Native Gateset
There is an *experimental* gateset that is implemented in native code to be faster. It implements a `QubitCNOTLinear` gateset. To use it, you must install `search_compiler` from source, and additionally must build and install the native library located in the `native/` directory.  [See the wiki for instructions on doing so](https://github.com/WolfLink/search_compiler/wiki/Experimental-Native-Gateset).
# Getting Started: search_compiler Projects
The simplest way to use the search_compiler library is by using a Project. When you create a project, you provide a path where a directory will be created to contain the project's files.
```
import search_compiler as sc
myproject = sc.Project("desired/path/to/project/directory")
```
You can then add unitaries to compile, and set compiler properties. Unitary matrices should be provided as `numpy` matrices using `dtype="complex128"`.
```
myproject.add_compilation("gate_name", gate_unitary)
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

# Compiling Without Projects
If you would like to avoid working with Projects, you can call the compiler function directly.
```
import search_compiler as sc
compiler = sc.SearchCompiler()
U_implemented, circuit, vector = compiler.compile(target_unitary)
```
The `SearchCompiler` class and the `compile` function can take extra arguments to further configure the compiler.  The return values are, in order, the unitary that represents the implemented circuit, the `sc.QuantumStep` representation of the circuit structure, and the vector of parameters for the circuit structure.

To export openqasm code, use the `assemble` function from `assembler.py`.
```
myqasm = sc.assembler.assemble(circuit, vector, sc.assembler.ASSEMBLY_IBMOPENQASM) # to get output as a string
sc.assembler.assemble(circuit, vector, sc.assembler.ASSEMBLY_IBMOPENQASM, write_location="myqasm.txt") # to write the output to a file
```

# A Note On Endianness
We use the physics convention of using big endian when naming our qubits.  Some quantum programs, including IBM's Qiskit, use little endian.  This means you will need to reverse the endianness of a unitary designed for Qiskit in order to work with our code, or visa versa.  We provide a function that performs endian reversal on numpy matrices:
```
little_endian = sc.utils.endian_reverse(big_endian) # you can use the same function to convert in the other direction as well
```

## Find information on customizing the compiler in the [wiki](https://github.com/WolfLink/search_compiler/wiki).
