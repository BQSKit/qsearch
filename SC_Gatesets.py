import SC_Utils as util
import SC_Sample_Gates as gates
from SC_Logging import logprint
from SC_Circuits import *

# Commonly used functions for generating gatesets
def linear_topology(double_step, n, d):
    identity_step = IdentityStep(d)
    return [KroneckerStep(*[identity_step]*i, double_step, *[identity_step]*(n-i-2)) for i in range(0,n-1)]

def fill_row(step, n):
    return KroneckerStep(*[step]*n)


class Gateset():
    # the compiler takes a gateset class as one of its arguments.  The gateset class generate a gateset to compile a specific circuit with based on the circuit size.

    # n is the number of qudits used in the gate
    # d is the number of states in the qudit (ie 2 for a qubit, 3 for a qutrit)

    # the layer that should be put between search layers.  Generally a layer of parameterized one-qudit gates.
    def pad_layer(self, n, d):
        return None # NOTE: Returns A SINGLE gate

    # the set of possible multi-qubit gates for searching.  May or may not be parameterized.
    def search_layers(self, n, d):
        return [] # NOTES: Returns a LIST of gates



class QubitCNOTLinear(Gateset):
    def pad_layer(self, n, d):
        return fill_row(SingleQubitStep(), n)
    
    def search_layers(self, n, d):
        return linear_topology(CNOTStep(), n, d)


class QubitCRZLinear(Gateset):
    def pad_layer(self, n, d):
        return fill_row(SingleQubitStep(), n)

    def search_layers(self, n, d):
        return linear_topology(CRZStep(), n, d)


class QubitCNOTRing(Gateset):
    def pad_layer(self, n, d):
        return fill_row(SingleQubitStep(), n)

    def search_layers(self, n, d):
        return [UStep(gates.arbitrary_cnot(n, i, (i+1)%n), name="CNOT q{} q{}".format(i, (i+1)%n), dits=n) for i in range(0, n)]


class QutritCPIPhaseLinear(Gateset):
    def pad_layer(self, n, d):
        return fill_row(SingleQutritStep(), n)
    
    def search_layers(self, n, d):
        return linear_topology(CPIPhaseStep(), n, d)


# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit
