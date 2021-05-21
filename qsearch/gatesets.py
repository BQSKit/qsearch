"""
This module defines the Gateset class, which represents the allowed gates and topology for a specific quantum computer.

Several Implementations of Gateset are also defined here.
Several aliases are also defined, for the most common use cases.

Attributes:
    ZXZXZCNOTLinear : A Gateset that uses CNOT and the ZXZXZ single qubit parameterization with the linear topology.
    U3CNOTLinear : A Gateset that uses CNOT and the U3 single qubit parameterization with the linear topology.
    QubitCNOTLinear : A Gateset that uses CNOT and the U3 single qubit parameterization with the linear topology, except it uses an XZXZ instead of a U3 after the control qubit of each CNOT.  This results in a gateset that covers the same search space as U3CNOTLinear, but with fewer redundant parameters, and therefore faster runtime.
    QubitCNOTRing : Uses U3 and XZXZ like QubitCNOTLinear, but includes a NonadjacentCNOTGate to add a link from the last qubit to the 0th.
    QubitCNOTAdjacencyList : Similar to QubitCNOTLinear and QubitCNOTRing, but takes in an adjacency list which uses NonadjacentCNOTGate to define work with a custom topology.
    QutritCPIPhaseLinear : A qutrit gateset that uses the CPIPhase gate as its two-qutrit gate, with a linear topology.
    QutritCNOTLinear : A qutrit gateset that uses an upgraded version of the CNOT gate as its two-qutrit gate, with a linear topology.

    DefaultQubit : The default Gateset for working with qubits.  Currently is equivalent to QubitCNOTLinear.
    DefaultQutrit : The default Gateset for working with qutrits.  Currently is equivalent to QutritCPIPhaseLinear.
    Default : The overall default Gateset, which is equivalent to DefaultQubit.
"""
from .gates import *
from .assemblers import flatten_intermediate
import numpy as np



class Gateset():
    """This class defines the supported gates and topology for a specific quantum hardware."""
    def __init__(self):
        """Gatesets must set the value of d in their initializer, which represents the size of qudits that are supported (e.g. 2 for qubits or 3 for qutrits)."""
        self.d = 0
        raise NotImplementedError("Gatesets must implemented their own initializers and must set self.d to reflect the size of the qudits implemented in the gateset")

    def initial_layer(self, qudits):
        """
        The initial layer in the compilation.  Usually a layer of parameterized single-qudit gates.

        Args:
            qudits : The number of qudits in this circuit.

        Returns:
            qsearch.gates.Gate : A single Gate representing an initial layer for the circuit
        """
        return None # NOTE: Returns A SINGLE gate

    # the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    def search_layers(self, qudits):
        """
        A set of possible multi-qubit gates for searching.  Usually this is a two-qudit gate followed by two single-qudit gates, for every allowed placement of the two-qudit gate.  This defines the branching factor of the search tree.

        Args:
            qudits : The number of qudits in this circuit
        
        Returns:
            list : A list of tuples of (gate,weight) where Gate is the Gate representing that possible placement of the two-qudit gate, and weight is the weight or cost of adding that gate in that placement to the final circuit.
        """
        return [] # NOTES: Returns a LIST of tuples of (gate, weight)

    def branching_factor(self, qudits):
        """
        Returns an integer indicating the expected branching factor.  Usually this is automatically determined from search_layers, but it may need to be overridden if successors is overridden.

        Args:
            qudits : The number of qudits in this circuit

        Returns:
            int : An integer indicating the expecte branching factor
        """
        # This is the default implementation, for Gatesets that rely on search_layers
        return len(self.search_layers(qudits))

    def successors(self, circ, qudits=None):
        """
        Returns a list of Gates that are successors in the search tree to the input Gate, circ, representing a current ansatz circuit.

        Args:
            circ : The curret ansatz Gate.
            qudits : The number of qudits in this circuit.

        Returns:
            list : A list of tuples of (gate, weight) where gate is a Gate that is a successor to circ, and weight is the cost or weight of moving to gate from circ.
        """

        # NOTE: Returns a LIST of tuples of (gate, weight)
        # NOTE: it is safe to assume that the circuit passed in here was produced by the functions of this class
        
        # This is the default implementation, for Gatesets that rely on search_layers
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
    """A Gateset for working with CNOT and single-qubit gates parameterized with ZXZXZGate on the linear topology."""
    def __init__(self):
        self.single_gate = ZXZXZGate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_gate, n, self.d)

class U3CNOTLinear(Gateset):
    """A Gateset for working with CNOT and single-qubit gates parameterized with U3Gate on the linear topology."""
    def __init__(self):
        self.single_gate = U3Gate()
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.cnot, self.single_gate, n, self.d)

class QubitCNOTLinear(Gateset):
    """A Gateset for working with CNOT and single-qubit gates parameterized with U3Gate and XZXZGate on the linear topology.  This Gateset covers the same search space but uses fewer parameters than ZXZXZCNOTLinear and U3CNOTLinear.
    
       Args:
           single_gate: A qsearch.gates.Gate object used as the single-qubit gate placed after the target side of a CNOT.
           single_alt: A qsearch.gates.Gate object used as the single-qubit gate placed after the control side of a CNOT.
    """
    def __init__(self, single_gate=U3Gate(), single_alt=XZXZGate()):
        self.single_gate = single_gate
        self.single_alt  = single_alt
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
    """A Gateset for working with CNOT and single-qubit gates parameterized with U3Gate and XZXZGate on the ring topology.
       Args:
           single_gate: A qsearch.gates.Gate object used as the single-qubit gate placed after the target side of a CNOT.
           single_alt: A qsearch.gates.Gate object used as the single-qubit gate placed after the control side of a CNOT.
    """
    def __init__(self, single_gate=U3Gate(), single_alt=XZXZGate()):
        self.single_gate = single_gate
        self.single_alt  = single_alt
        self.cnot = CNOTGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)

    def search_layers(self, n):
        I = IdentityGate()
        gates = linear_topology(self.cnot, self.single_gate, n, self.d, identity_gate=I, single_alt=self.single_alt)
        if n == 2:
            return gates
        finisher = (ProductGate(NonadjacentCNOTGate(n, n-1, 0), KroneckerGate(self.single_gate, *[I]*(n-2), self.single_alt)), 1)
        return gates + [finisher]

class QubitCZLinear(Gateset):
    """A Gateset for working with CZ and single-qubit gates parameterized with U3Gate and XZXZGate on the linear topology."""
    def __init__(self):
        self.single_gate = U3Gate()
        self.single_alt  = XZXZGate()
        self.two_gate = CZGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.two_gate, self.single_gate, n, self.d, single_alt=self.single_alt)

    def branching_factor(self, qudits):
        return qudits-1

    def successors(self, circ, qudits=None):
        if qudits is None:
            qudits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        skip_index = find_last_3_cnots_linear(circ)
        return [(circ.appending(layer[0]), layer[1]) for layer in linear_topology(self.two_gate, self.single_gate, qudits, self.d, single_alt=self.single_alt, skip_index=skip_index)]

class QubitISwapLinear(Gateset):
    """A Gateset for working with ISwap and single-qubit gates parameterized with U3Gate and XZXZGate on the linear topology."""
    def __init__(self):
        self.single_gate = U3Gate()
        self.single_alt  = XZXZGate()
        self.two_gate = ISwapGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.two_gate, self.single_gate, n, self.d, single_alt=self.single_alt)

    def branching_factor(self, qudits):
        return qudits-1

    def successors(self, circ, qudits=None):
        if qudits is None:
            qudits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        skip_index = find_last_3_cnots_linear(circ)
        return [(circ.appending(layer[0]), layer[1]) for layer in linear_topology(self.two_gate, self.single_gate, qudits, self.d, single_alt=self.single_alt, skip_index=skip_index)]

class QubitXXLinear(Gateset):
    """A Gateset for working with ISwap and single-qubit gates parameterized with U3Gate and XZXZGate on the linear topology."""
    def __init__(self):
        self.single_gate = U3Gate()
        self.single_alt  = XZXZGate()
        self.two_gate = XXGate()
        self.d = 2

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(self.two_gate, self.single_gate, n, self.d, single_alt=self.single_alt)

    def branching_factor(self, qudits):
        return qudits-1

    def successors(self, circ, qudits=None):
        if qudits is None:
            qudits = int(np.log(circ.matrix([0]*circ.num_inputs).shape[0])/np.log(self.d))
        skip_index = find_last_3_cnots_linear(circ)
        return [(circ.appending(layer[0]), layer[1]) for layer in linear_topology(self.two_gate, self.single_gate, qudits, self.d, single_alt=self.single_alt, skip_index=skip_index)]

class QubitCNOTAdjacencyList(Gateset):
    """A Gateset for working with CNOT and single-qubit gates parameterized with U3Gate and XZXZGate on a custom topology, specified in the initializer."""
    def __init__(self, adjacency, single_gate=U3Gate(), single_alt=XZXZGate()):
        """
        Allows the specification of a custom topology through an adjacency list.

        For example, this is how you would specifiy the ring topology for 3 qubits:
        `[(0,1), (1,2), (2,1)]`

        It is not recommended to add bi-directional links, because with the arbitrary parameterized single qubit gates everywhere, such links would be redundant.

        Args:
            adjacency : A list of tuples specifying which CNOT placements are allowed.  The tuples must be in the form of (control, target).
            single_gate: A qsearch.gates.Gate object used as the single-qubit gate placed after the target side of a CNOT.
            single_alt: A qsearch.gates.Gate object used as the single-qubit gate placed after the control side of a CNOT.
        """
        self.single_gate = single_gate
        self.single_alt = single_alt
        self.I = IdentityGate()
        self.adjacency = adjacency
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
    """A Gateset for working with CPIPhase and single-qutrit gates on the linear topology."""
    def __init__(self):
        self.single_gate = SingleQutritGate()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(CPIPhaseGate(), self.single_gate, n, self.d)

class QutritCNOTLinear(Gateset):
    """A hybrid Gateset for working with CNOT and single-qutrit gates on the linear topology."""
    def __init__(self):
        self.single_gate = SingleQutritGate()
        self.d = 3

    def initial_layer(self, n):
        return fill_row(self.single_gate, n)
    
    def search_layers(self, n):
        return linear_topology(UpgradedConstantGate(CNOTGate()), self.single_gate, n, self.d)

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


# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit

