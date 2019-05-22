import numpy as np
from . import circuits as circuits
from . import utils as util

gate_X = np.matrix([[0,1],[1,0]], dtype='complex128')
gate_H = np.matrix([[1/np.sqrt(2), 1/np.sqrt(2)],[1/np.sqrt(2), -1/np.sqrt(2)]], dtype='complex128')
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
        return np.matrix([[np.exp(-1j*v[0]/2), 0],[0, np.exp(1j*v[0]/2)]], dtype='complex128')

    def path(self, v):
        return ["RZ", list(v)]

    def assemble(self, v, i=0):
        return "RZ({}) q{}".format(*v, i)

    def __repr__(self):
        return "RZStep()"

class RXStep(circuits.QuantumStep):
    def __init__(self):
        self._num_inputs = 1
        self._dits = 1

    def matrix(self, v):
        return np.matrix([[np.cos(v[0]/2), -1j*np.sin(v[0]/2)],[-1j*np.sin(v[0]/2), np.cos(v[0]/2)]], dtype='complex128')

    def path(self, v):
        return ["RX", list(v)]

    def assemble(self, v, i=0):
        return "RX({}) q{}".format(*v, i)

    def __repr__(self):
        return "RXStep()"

class RYStep(circuits.QuantumStep):
    def __init__(self):
        self._num_inputs = 1
        self._dits = 1

    def matrix(self, v):
        return np.matrix([[np.cos(v[0]/2), -np.sin(v[0]/2)],[np.sin(v[0]/2), np.cos(v[0]/2)]], dtype='complex128')

    def path(self, v):
        return ["RY", list(v)]

    def assemble(self, v, i=0):
        return "RY({}) q{}".format(*v, i)

    def __repr__(self):
        return "RYStep()"


step_RX = RXStep()
step_RY = RYStep()
step_RZ = RZStep()
    