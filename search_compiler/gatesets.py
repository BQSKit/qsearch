from .logging import logprint
from .circuits import *

try:
    from search_compiler_rs import QubitCNOTLinearNative
except ImportError:
    pass

#TODO: rename all the n's in here to "dits" as appropriate

# Commonly used functions for generating gatesets
def linear_topology(double_step, single_step, n, d, identity_step=None, single_alt=None):
    if not identity_step:
        identity_step = IdentityStep(d)
    if single_alt is None:
        return [KroneckerStep(*[identity_step]*i, ProductStep(double_step, KroneckerStep(single_step, single_step)), *[identity_step]*(n-i-2)) for i in range(0, n-1)]
    else:
        return [KroneckerStep(*[identity_step]*i, ProductStep(double_step, KroneckerStep(single_alt, single_step)), *[identity_step]*(n-i-2)) for i in range(0, n-1)]

def fill_row(step, n):
    return KroneckerStep(*[step]*n)


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
        return [] # NOTES: Returns a LIST of gates

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
        finisher = ProductStep(NonadjacentCNOTStep(n, n-1, 0), KroneckerStep(self.single_step, *[I]*(n-2), self.single_alt))
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
            return [ProductStep(CNOTStep(), KroneckerStep(self.single_step, self.single_step))] # prevents the creation of an extra cnot placement in the 2 qubit case

        steps = []
        for pair in self.adjacency:
            if pair[0] >= n or pair[1] >= n:
                continue
            cnot = NonadjacentCNOTStep(n, pair[0], pair[1])
            single_steps = [self.single_step if i == pair[0] else self.single_alt if i == pair[1] else self.I for i in range(0, n)]
            steps.append(ProductStep(cnot, KroneckerStep(*single_steps))) 
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
        finisher = ProductStep(NonadjacentCRZStep(n, n-1, 0), KroneckerStep(self.single_step, *[I]*(n-2), self.single_alt))
        return steps + [finisher]

class QutritCPIPhaseLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQutritStep()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n):
        return linear_topology(CPIPhaseStep(), self.single_step, n, self.d)


# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit

