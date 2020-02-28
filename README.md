# search_compiler
A compiler for quantum computers based on A* and numerical optimization.

# Installation
This is a python package which can be installed using pip:
```
git clone git@github.com:WolfLink/search_compiler.git
python3 -m pip install ./search_compiler
```
Then you can import the library as any other python package:
```
import search_compiler as sc
```

# Getting Started - search_compiler Projects
The simplest way to use the search_compiler library is by using a Project. When you create a project, you provide a path where a directory will be created to contain the project's files.
```
import search_compiler as sc
myproject = sc.Project("desired/path/to/project/directory")
```
You can then add unitaries to compile, and set compiler properties. Unitary matrices should be provided as `numpy` matrices using `dtype="complex128"`. Details on compiler options are provided in a later section.
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
