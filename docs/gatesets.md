# Gatesets in qsearch

To synthesize with different gates or topologies, you will need to create an instance of an `qsearch.gatesets.Gateset` subclass.
```
# example: synthesizing for the ring topology
import qsearch as qs
ring_gateset = qs.gatesets.QubitCNOTRing()

# use your gateset with a project
myproject = qs.Project("myproject")
myproject["gateset"] = ring_gateset

# or use it with SearchCompiler directly
mycompiler = qs.SearchCompiler(gateset=ring_gateset)
```
## Provided Gatesets
### Basic Gatesets
* `QubitCNOTLinear` - a gateset that is useful for synthesizing circuits with CNOTs and single qubit gates with the linear topology.  It is similar to `U3CNOTLinear`, but is slightly more efficient without sacrificing generality.  It is the default gateset.
* `U3CNOTLinear` - a gateset based on IBM's U3 gate and CNOTs for the linear topology.  It is generally better to use `QubitCNOTLinear`, which is more efficient.
* `ZXZXZCNOTLinear` - a gateset based on the RZ-RX90-RZ-RX90-RZ decomposition of single qubit gates for the linear topology.  It is generally better to use `U3CNOTLinear`, which is more efficient.
### Nonlinear Topologies
* `QubitCNOTRing` - a gateset that is equivalent to `QubitCNOTLinear` except it implements the ring topology.  For 3 qubits, this is the triangle topology and is all-to-all.
* `QubitCNOTAdjacencyList` - a gateset that takes a list of CNOT connections, and creates a gateset that is similar to `QubitCNOTLinear` but uses a toplogy based on the adjacency list.  If the desired topology can be achieved by using `QubitCNOTLinear` or `QubitCNOTRing`, it is recommended to choose one of those because it will be more efficient.
```
# This would create a gateset for 4 qubits with CNOT connections 0 to 1, 0 to 2, and 1 to 3
mygateset = qs.gatesets.QubitCNOTAdjacencyList([(0,1), (0,2), (1,3)])
```
### Qutrits
* `QutritCPIPhaseLinear` - a gateset designed for qutrits that uses single qutrit gates and the CPI two-qutrit gate with a phase applied.
## Custom Gatesets
If none of these gatesets suite your needs, you can write your own!  Make a subclass of `qs.gatesets.Gateset` and implement these two functions:
* `intial_layer(n)` The single input, `n`, is an integer which deqsribes how many qudits will be in the circuit.  The function returns a single `qs.gates.Gate` object representing an initial layer for the search.  Normally, this is a kronecker product of single-qudit gates, and you can use the provided `fill_row` helper function to produce this.
* `search_layers(n)` The single input, `n`, is an integer which deqsribes how many qudits will be in the circuit.  The function returns a list of `qs.gates.Gate` objects, each representing a possible building block in a possible location for expanding the current circuit.

See the existing implementations in `qs.gatesets` for examples of how to write a gateset.