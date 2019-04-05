import SC_Utils as util
import SC_Sample_Gates as gates
from SC_Logging import logprint
from SC_Circuits import *

# Commonly used functions for generating gatesets
def linear_topology(double_step, single_step, n, d, identity_step=None):
    if not identity_step:
        identity_step = IdentityStep(d)
    return [ProductStep(KroneckerStep(*[identity_step]*i, double_step, *[identity_step]*(n-i-2)),
                        KroneckerStep(*[identity_step]*i, single_step, single_step, *[identity_step]*(n-i-2))
            ) for i in range(0,n-1)]

def fill_row(step, n):
    return KroneckerStep(*[step]*n)


class Gateset():
    # the compiler takes a gateset class as one of its arguments.  The gateset class generate a gateset to compile a specific circuit with based on the circuit size.

    # n is the number of qudits used in the gate
    # d is the number of states in the qudit (ie 2 for a qubit, 3 for a qutrit)

    # The first layer in the compilation.  Generally a layer of parameterized single-qubit gates
    def initial_layer(self, n, d):
        return None # NOTE: Returns A SINGLE gate

    # the set of possible multi-qubit gates for searching.  Generally a two-qubit gate with single qubit gates after it.
    def search_layers(self, n, d):
        return [] # NOTES: Returns a LIST of gates


class QubitCNOTLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQubitStep()
        self.cnot = CNOTStep()

    def initial_layer(self, n, d):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n, d):
        return linear_topology(self.cnot, self.single_step, n, d)


class QubitCRZLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQubitStep()

    def initial_layer(self, n, d):
        return fill_row(self.single_step, n)

    def search_layers(self, n, d):
        return linear_topology(CRZStep(), self.single_step, n, d)


class QubitCNOTRing(Gateset):
    def __init__(self):
        self.single_step = SingleQubitStep()
        self.I = IdentityStep(2)

    def initial_layer(self, n, d):
        return fill_row(self.single_step, n)

    def search_layers(self, n, d):
        if n == 2:
            return [ProductStep(CNOTStep(), KroneckerStep(self.single_step, self.single_step))] # prevents the creation of an extra cnot placement in the 2 qubit case

        steps = []
        for i in range(0, n):
            cnot = UStep(gates.arbitrary_cnot(n, i, (i+1)%n), name="CNOT q{} q{}".format(i, (i+1)%n), dits=n)
            single_steps = []
            if i+1 == n:
                single_steps.append(self.single_step)
                single_steps.extend([self.I]*(n-2))
                single_steps.append(self.single_step)
            else:
                single_steps.extend([self.I]*i)
                single_steps.extend([self.single_step, self.single_step])
                single_steps.extend([self.I]*(n-i-2))
            
            steps.append(ProductStep(cnot, KroneckerStep(*single_steps))) 
        return steps


class QutritCPIPhaseLinear(Gateset):
    def __init__(self):
        self.single_step = SingleQutritStep()

    def initial_layer(self, n, d):
        return fill_row(self.single_step, n)
    
    def search_layers(self, n, d):
        return linear_topology(CPIPhaseStep(), self.single_Step, n, d)


# commonly used defaults
DefaultQubit = QubitCNOTLinear
DefaultQutrit = QutritCPIPhaseLinear
Default = DefaultQubit
