from .gates import *
from .assemblers import flatten_intermediate
import numpy as np


# Commonly used functions for generating gatesets
def linear_topology(double_gate, single_gate, n, d, identity_gate=None, single_alt=None, double_weight=1, single_weight=0, skip_index=None):
    weight = double_weight + 2*single_weight
    if not identity_gate:
        identity_gate = IdentityGate(d=d)
    if single_alt is None:
        return [(KroneckerGate(*[identity_gate]*i, ProductGate(double_gate, KroneckerGate(single_gate, single_gate)), *[identity_gate]*(n-i-2)), weight) for i in range(0, n-1) if skip_index is None or i != skip_index]
    else:
        return [(KroneckerGate(*[identity_gate]*i, ProductGate(double_gate, KroneckerGate(single_alt, single_gate)), *[identity_gate]*(n-i-2)), weight) for i in range(0, n-1) if skip_index is None or i != skip_index]

def fill_row(gate, n):
    return KroneckerGate(*[gate]*n)


def find_last_3_cnots_linear(circuit):
    # this function finds the last 3 CNOTs in the circuit and returns the index if they are all in a row or None otherwise
    # this function is written specifically for linear topology-based gatesets.  Other gatesets should use find_last_3_cnots_arbitrary.
    # do not use either of these functions with gatesets not intended specifically for it

    contents = circuit._subgates
    if len(contents) < 4:
        return None
    indices = [[type(sub) for sub in gate._subgates].index(ProductGate) for gate in contents[-1:-4:-1]]

    if indices[0] == indices[1] and indices[1] == indices[2]:
        return indices[0]
    else:
        return None

class Gateset():
    def __init__(self):
        self.d = 0
        raise NotImplementedError("Gatesets must implemented their own initializers and must set self.d to reflect the size of the ququdits implemented in the gateset")
    # The compiler takes a gateset class as one of its arguments.  The gateset class represents what the hardware can do.
    # All gatesets must set the property d, which represents the size of the ququdits represented (eg 2 for qubits, 3 for qutrits)

    # qudits is the number of qudits used in a circuit (usually calculated in the compiler as log(n) / log(d)

    # The first layer in the compilation.  Generally a layer of parameterized single-qubit gates
    def initial_layer(self, qudits):
        return None # NOTE: Returns A SINGLE gate

    # the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    def search_layers(self, qudits):
        return [] # NOTES: Returns a LIST of tuples of (gate, weight)

    def branching_factor(self, qudits):
        # returns an integer indicating the expected branching factor

        # this implemenation is a backwards compatibility implementation and should not be relied on
        return len(self.search_layers(qudits))

    def successors(self, circ, qudits=None):
        # NOTE: Returns a LIST of tuples of (gate, weight)
        # NOTE: it is safe to assume that the circuit passed in here was produced by the functions of this class
        
        # this implementation is a backwards compatibility implementation and should not be relied on
        qudits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        return [(circ.appending(t[0]), t[1]) for t in self.search_layers(qudits)]

    def __eq__(self, other):
        if self is other:
            return True
        if self.__module__ == Gateset.__module__:
            if type(self) == type(other):
                if self.__dict__ == other.__dict__:
                    return True
        return False

class ZXZXZCNOTLinear(Gateset):
    def __init__(self):
        self.single_gate = ZXZXZQubitGate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_gate, n, self.d)

class U3CNOTLinear(Gateset):
    def __init__(self):
        self.single_gate = U3Gate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_gate, n, self.d)

class QubitCNOTLinear(Gateset):
    def __init__(self):
        self.single_gate = U3Gate()
        self.single_alt  = XZXZGate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_gate, n, self.d, single_alt=self.single_alt)

    def branching_factor(self, qudits):
        return qudits-1

    def successors(self, circ, qudits=None):
        if qudits is None:
            qudits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        skip_index = find_last_3_cnots_linear(circ)
        return [(circ.appending(layer[0]), layer[1]) for layer in linear_topology(self.cnot, self.single_gate, qudits, self.d, single_alt=self.single_alt, skip_index=skip_index)]

class QubitCNOTRing(Gateset):
    def __init__(self):
        self.single_gate = U3Gate()
        self.single_alt  = XZXZGate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)

    def search_layers(self, n):
        I = IdentityGate(2)
        gates = linear_topology(self.cnot, self.single_gate, n, self.d, identity_gate=I, single_alt=self.single_alt)
        if n == 2:
            return gates
        finisher = (ProductGate(NonadjacentCNOTGate(n, n-1, 0), KroneckerGate(self.single_gate, *[I]*(n-2), self.single_alt)), 1)
        return gates + [finisher]


class QubitCNOTAdjacencyList(Gateset):
    def __init__(self, adjacency):
        self.single_gate = U3Gate()
        self.single_alt = XZXZGate()
        self.I = IdentityGate()
        self.adjacency = adjacency # a list of tuples of control-target.  It is not recommended to add bidirectional links, because that difference can be handled more efficiently via single qubit gates.
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)

    def search_layers(self, n):
        if n == 2:
            return [(ProductGate(CNOTGate(), KroneckerGate(self.single_gate, self.single_gate)), 1)] # prevents the creation of an extra cnot placement in the 2 qubit case

        gates = []
        for pair in self.adjacency:
            if pair[0] >= n or pair[1] >= n:
                continue
            cnot = NonadjacentCNOTGate(n, pair[0], pair[1])
            single_gates = [self.single_gate if i == pair[0] else self.single_alt if i == pair[1] else self.I for i in range(0, n)]
            gates.append((ProductGate(cnot, KroneckerGate(*single_gates)), 1)) 
        return gates

class QutritCPIPhaseLinear(Gateset):
    def __init__(self):
        self.single_gate = SingleQutritGate()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(CPIPhaseGate(), self.single_gate, n, self.d)

class QutritCNOTLinear(Gateset):
    def __init__(self):
        self.single_gate = SingleQutritGate()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(UpgradedConstantGate(CNOTGate()), self.single_gate, n, self.d)



# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit

