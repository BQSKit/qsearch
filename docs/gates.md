# Gates in qsearch

The classes representing quantum gates are found in `gates.py`, and are subclasses of `qsearch.gates.Gate`.  You will need to work with `Gate` objects to crate custom gatesets, and you will get a `Gate` object as a return value from compilation.
```
# Here are some examples of what you can do with Gate objects
U3 = qsearch.U3Gate()
CNOT = qsearch.CNOTGate()

# get the matrix that a gate represents in a numpy matrix format
U3_unitary = U3.matrix([np.pi/2, np.pi/4, np./pi/6]) # the array of parameters must be provided
CNOT_unitary = CNOT.matrix([]) # cnot takes no parameters so an empty array is provided

# combine multiple gates to form a larger circuit
mycircuit = ProductGate(KroneckerGate(U3,U3), CNOT) # note that mycircuit is itself an instance of Gate
```
## Provided Gates

For more information, see the API documentation in `qsearch.gates`.

## Custom Gates
There is an existing gate that can be customized to your needs.  However it will not show up when you assembly the circuit to OpenQASM or Qiskit.
* `UGate` - represents the gate deqsribed by the unitary `U` passed to `__init__`, and takes up `qudits` qudits.

You can also write your own `Gate` subclasses the required functions are:
* `__init__` - you must customize the initializer to set `self.num_inputs` to the number of parameters for the gate (e.g. 3 for U3 or ZXZXZ, 0 for CNOT), and `self.qudits` to the number of qudits used by the gate (e.g. 1 for U3 or ZXZXZ, 2 for CNOT).
* `matrix(v)` - here you generate and return the matrix represented by your gate when passed the parameters provided in the array `v`.

### Assembling with Custom Gates
If you want your code to output your custom gates when assembling, you must implement `assemble` as well.
* `assemble(v, i)` - here you are given `v`, the list of parameters needed for your gate, and `i`, the index of the first qubit in the set of qubits that your gate is assigned.  You must return an array of the form `[gate1, gate2]` where `gate1` and `gate2` are tuples that represent gates that the assembler will be able to interpret.  For example, `ZXZXZGate` returns an array for 5 tuples, one for each of the Z and X gates that it is based on, but `U3Gate` only returns an array of 1 tuple because the assembler interprets it as a single gate.  The tuples take the form `("gate", gatename, parameters, indices)`, where the word "gate" is included to specify that this tuple represents a well defined gate as opposed to a Kronecker product of gates, `gatename` is a string that will be used to look up the relevant format to print this gate when assembling, and `parameters` is a list of  the parameters formatted and organized the way they are needed to fill the format specified in the assembler, and `indices` is a list of the indices of the involved qubits.

### Faster Solving with Jacobians
If you would like to take advantage of faster solvers that can take advantage of the Jacobian (marked with `Jac` in their name), and your custom gate uses one or more parameters, you will need to implement `mat_jac` as well.
* `mat_jac(v)` - here you generate and return a tuple `(U, [J1, ... ,Jn])` where `U` is the same matrix you would return in `matrix`, and `[J1, ... ,Jn]` is a list of matrix derivatives, where J1 is the matrix of derivatives with respect to the first parameter in `v`, and so on for all the parameters in v.  **If your custom gate is constant (`self.num_inputs == 0`), then you can take advantage of Jacobian solvers without implementing `mat_jac` yourself.**
