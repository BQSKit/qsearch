import numpy as np
from . import circuits as circuits
from . import utils as util

gate_X = np.array([[0,1],[1,0]], dtype='complex128')
gate_H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],[1/np.sqrt(2), -1/np.sqrt(2)]], dtype='complex128')
gate_CX = util.cnot

step_X = circuits.UStep(gate_X, name="X")
step_H = circuits.UStep(gate_H, name="H")
step_CX = circuits.CNOTStep()
step_I = circuits.IdentityStep(2)


class RZStep(circuits.QuantumStep):
    def __init__(self):
        self._num_inputs = 1
        self._dits = 1

    def matrix(self, v):
        x = v[0] * np.pi * 2
        return np.array([[np.exp(-1j*x/2), 0],[0, np.exp(1j*x/2)]], dtype='complex128')

    def path(self, v):
        return ["RZ", list(v)]

    def assemble(self, v, i=0):
        return [("gate", "Z", (v[0]*np.pi*2,), (i,))]

    def __repr__(self):
        return "RZStep()"

class RXStep(circuits.QuantumStep):
    def __init__(self):
        self._num_inputs = 1
        self._dits = 1

    def matrix(self, v):
        x = v[0] * np.pi * 2
        return np.array([[np.cos(x/2), -1j*np.sin(x/2)],[-1j*np.sin(x/2), np.cos(x/2)]], dtype='complex128')

    def path(self, v):
        return ["RX", list(v)]

    def assemble(self, v, i=0):
        return [("gate", "X", (v[0]*np.pi*2,), (i,))]

    def __repr__(self):
        return "RXStep()"

class RYStep(circuits.QuantumStep):
    def __init__(self):
        self._num_inputs = 1
        self._dits = 1

    def matrix(self, v):
        x = v[0] * np.pi*2
        return np.array([[np.cos(x/2), -np.sin(x/2)],[np.sin(x/2), np.cos(x/2)]], dtype='complex128')

    def path(self, v):
        return ["RY", list(v)]

    def assemble(self, v, i=0):
        return [("gate", "Y", (v[0]*np.pi*2,), (i,))]

    def __repr__(self):
        return "RYStep()"


step_RX = RXStep()
step_RY = RYStep()
step_RZ = RZStep()
    
