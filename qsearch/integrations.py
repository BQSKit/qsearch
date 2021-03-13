try:
    from qiskit import QuantumCircuit
    import qiskit
except ImportError:
    qiskit = None
    raise ImportError("Cannot import qiskit, please run pip3 install qiskit before importing qiskit code.")

import numpy as np

from .gates import *


class QiskitImportError(Exception):
    """A class to represent issues importing code from qiskit"""


class QiskitGateConverter:
    def __init__(self, num_qubits):
        self.registers = []
        self.num_qubits = num_qubits
        self.parameters = []

    def convert(self, gate, qubits, cbits):
        """Abstraction to convert an arbitrary qiskit gate to a layer in a qsearch circuit"""
        if cbits != []:
            raise QiskitImportError("Classical operations are not supported in qsearch for now.")
        return getattr(self, f'convert_{gate.name}')(gate, qubits, cbits)
    
    def convert_cx(self, gate, qubits, cbits):
        for q in qubits:
            if q.register.name not in self.registers:
                if len(self.registers) == 0:
                    self.registers.append(q.register.name)
                else:
                    raise QiskitImportError("Qsearch does not support importing circuits with multiple quantum registers.")
        pair = [q.index for q in qubits]
        assert len(pair) == 2, "CNOT between more than 2 qubits?"
        return NonadjacentCNOTGate(self.num_qubits, pair[0], pair[1])

    def convert_u3(self, gate, qubits, cbits):
        assert len(qubits) == 1, "U3 on more than one qubit?"
        identity_gate = IdentityGate()
        self.parameters.extend(gate.params)
        index = qubits[0].index
        return KroneckerGate(*[identity_gate]*index, U3Gate(), *[identity_gate]*(self.num_qubits-index-1))

    def convert_u2(self, gate, qubits, cbits):
        assert len(qubits) == 1, "U2 on more than one qubit?"
        identity_gate = IdentityGate()
        self.parameters.extend(gate.params)
        index = qubits[0].index
        return KroneckerGate(*[identity_gate]*index, U2Gate(), *[identity_gate]*(self.num_qubits-index-1))

    def convert_rx(self, gate, qubits, cbits):
        assert len(qubits) == 1, "X on more than one qubit?"
        identity_gate = IdentityGate()
        self.parameters.extend(gate.params)
        index = qubits[0].index
        return KroneckerGate(*[identity_gate]*index, XGate(), *[identity_gate]*(self.num_qubits-index-1))

    def convert_ry(self, gate, qubits, cbits):
        assert len(qubits) == 1, "Y on more than one qubit?"
        identity_gate = IdentityGate()
        self.parameters.extend(gate.params)
        index = qubits[0].index
        return KroneckerGate(*[identity_gate]*index, YGate(), *[identity_gate]*(self.num_qubits-index-1))

    def convert_rz(self, gate, qubits, cbits):
        assert len(qubits) == 1, "Z on more than one qubit?"
        identity_gate = IdentityGate()
        self.parameters.extend(gate.params)
        index = qubits[0].index
        return KroneckerGate(*[identity_gate]*index, ZGate(), *[identity_gate]*(self.num_qubits-index-1))


def qiskit_to_qsearch(circ, converter=None):
    """Convert qiskit code to qsearch *structure* and parameters"""
    converter = converter if converter is not None else QiskitGateConverter(circ.num_qubits)
    circuit = []
    for gate, qubits, cbits in circ.data:
        circuit.append(converter.convert(gate, qubits, cbits))
    return ProductGate(*circuit), np.array(converter.parameters)
