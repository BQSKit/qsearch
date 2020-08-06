from .circuits import *
from .assembler import flatten_intermediate
import numpy as np


#TODO: rename all the n's in here to "dits" as appropriate

# Commonly used functions for generating gatesets
def linear_topology(double_step, single_step, n, d, identity_step=None, single_alt=None, double_weight=1, single_weight=0, skip_index=None):
    weight = double_weight + 2*single_weight
    if not identity_step:
        identity_step = IdentityStep(d)
    if single_alt is None:
        return [(KroneckerStep(*[identity_step]*i, ProductStep(double_step, KroneckerStep(single_step, single_step)), *[identity_step]*(n-i-2)), weight) for i in range(0, n-1) if skip_index is None or i != skip_index]
    else:
        return [(KroneckerStep(*[identity_step]*i, ProductStep(double_step, KroneckerStep(single_alt, single_step)), *[identity_step]*(n-i-2)), weight) for i in range(0, n-1) if skip_index is None or i != skip_index]

def fill_row(step, n):
    return KroneckerStep(*[step]*n)


def find_last_3_cnots(circuit):
    # for CNOT-based circuit, this function will return the index of the last 3 CNOTs if they are all in a row, or None otherwise
    il = [tup for tup in flatten_intermediate(circuit.assemble([0]*circuit.num_inputs)) if tup[1] == "CNOT"]
    if len(il) < 3:
        return None
    il = il[-1:-4:-1]
    if il[0] == il[1] and il[1] == il[2]:
        return il[0]
    else:
        return None


class Gateset():
    def __init__(self):
        self.d = 0
        raise NotImplementedError("Gatesets must implemented their own initializers and must set self.d to reflect the size of the qudits implemented in the gateset")
    # The compiler takes a gateset class as one of its arguments.  The gateset class represents what the hardware can do.
    # All gatesets must set the property d, which represents the size of the qudits represented (eg 2 for qubits, 3 for qutrits)

    # dits is the number of qudits used in a circuit (usually calculated in the compiler as log(n) / log(d)

    # The first layer in the compilation.  Generally a layer of parameterized single-qubit gates
    def initial_layer(self, dits):
        return None # NOTE: Returns A SINGLE gate

    # the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    def search_layers(self, dits):
        return [] # NOTES: Returns a LIST of tuples of (gate, weight)

    def branching_factor(self, dits):
        # returns an integer indicating the expected branching factor

        # this implemenation is a backwards compatibility implementation and should not be relied on
        return len(self.search_layers(dits))

    def successors(self, circ, dits=None):
        # NOTE: Returns a LIST of tuples of (gate, weight)
        # NOTE: it is safe to assume that the circuit passed in here was produced by the functions of this class
        
        # this implementation is a backwards compatibility implementation and should not be relied on
        dits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        return [(circ.appending(t[0]), t[1]) for t in self.search_layers(dits)]

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
        self.single_step = ZXZXZQubitStep()
        self.cnot = CNOTStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_step, n, self.d)

class QiskitU3Linear(Gateset):
    def __init__(self):
        self.single_step = QiskitU3QubitStep()
        self.cnot = CNOTStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_step, n, self.d)

class QubitCNOTLinear(Gateset):
    def __init__(self):
        self.single_step = QiskitU3QubitStep()
        self.single_alt  = XZXZPartialQubitStep()
        self.cnot = CNOTStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_step, n, self.d, single_alt=self.single_alt)

    def branching_factor(self, dits):
        return dits-1

    def successors(self, circ, dits=None):
        if dits is None:
            dits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        skip_index = find_last_3_cnots(circ)
        return [(circ.appending(layer[0]), layer[1]) for layer in linear_topology(self.cnot, self.single_step, dits, self.d, single_alt=self.single_alt, skip_index=skip_index)]

class QubitCRZLinear(Gateset):
    def __init__(self):
        self.single_step = QiskitU3QubitStep()
        self.single_alt  = XZXZPartialQubitStep()
        self.crz = CRZStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)

    def search_layers(self, n):
        return linear_topology(self.crz, self.single_step, n, self.d, single_alt=self.single_alt)


class QubitCNOTRing(Gateset):
    def __init__(self):
        self.single_step = QiskitU3QubitStep()
        self.single_alt  = XZXZPartialQubitStep()
        self.cnot = CNOTStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)

    def search_layers(self, n):
        I = IdentityStep(2)
        steps = linear_topology(self.cnot, self.single_step, n, self.d, identity_step=I, single_alt=self.single_alt)
        if n == 2:
            return steps
        finisher = (ProductStep(NonadjacentCNOTStep(n, n-1, 0), KroneckerStep(self.single_step, *[I]*(n-2), self.single_alt)), 1)
        return steps + [finisher]


class QubitCNOTAdjacencyList(Gateset):
    def __init__(self, adjacency):
        self.single_step = QiskitU3QubitStep()
        self.single_alt = XZXZPartialQubitStep()
        self.I = IdentityStep(2)
        self.adjacency = adjacency # a list of tuples of control-target.  It is not recommended to add bidirectional links, because that difference can be handled more efficiently via single qubit gates.
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)

    def search_layers(self, n):
        if n == 2:
            return [(ProductStep(CNOTStep(), KroneckerStep(self.single_step, self.single_step)), 1)] # prevents the creation of an extra cnot placement in the 2 qubit case

        steps = []
        for pair in self.adjacency:
            if pair[0] >= n or pair[1] >= n:
                continue
            cnot = NonadjacentCNOTStep(n, pair[0], pair[1])
            single_steps = [self.single_step if i == pair[0] else self.single_alt if i == pair[1] else self.I for i in range(0, n)]
            steps.append((ProductStep(cnot, KroneckerStep(*single_steps)), 1)) 
        return steps

class QubitCRZRing(Gateset):
    def __init__(self):
        self.single_step = QiskitU3QubitStep()
        self.single_alt  = XZXZPartialQubitStep()
        self.crz = CRZStep()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_step, n)

    def search_layers(self, n):
        I = IdentityStep(2)
        steps = linear_topology(self.crz, self.single_step, n, self.d, identity_step=I, single_alt=self.single_alt)
        if n == 2:
            return steps
        finisher = (ProductStep(NonadjacentCRZStep(n, n-1, 0), KroneckerStep(self.single_step, *[I]*(n-2), self.single_alt)), 1)
        return steps + [finisher]

class QutritCPIPhaseLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQutritStep()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(CPIPhaseStep(), self.single_step, n, self.d)

class QutritCNOTLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQutritStep()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(UpgradedConstantStep(CNOTStep()), self.single_step, n, self.d)



# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit

