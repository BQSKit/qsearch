try:
    from qiskit import QuantumCircuit
    import qiskit
except ImportError:
    qiskit = None
    raise ImportError("Cannot import qiskit, please run pip3 install qiskit before importing qiskit code.")
from .circuits import *


class QiskitImportError(Exception):
    """A class to represent issues importing code from qiskit"""

def new_layer(instruction, pair, num_qubits):
    min_idx = min(pair)
    max_idx = max(pair)
    diff_idx = max_idx - min_idx
    xzxz = XZXZPartialQubitStep()
    u3 = QiskitU3QubitStep()
    cnot = NonadjacentCNOTStep(diff_idx + 1, pair[0], pair[1])
    if diff_idx > 1:
        between_u = IdentityStep(2 * (diff_idx - 1))
        u_layer = KroneckerStep(xzxz, between_u, u3)
    else:
        u_layer = KroneckerStep(xzxz, u3)
    p_layer = ProductStep(cnot, u_layer)
    up = [IdentityStep(2)]*min_idx
    down = [IdentityStep(2)]* (num_qubits - max_idx - 1)
    return KroneckerStep( *up, p_layer, *down)

def qiskit_to_qsearch(circ):
    """Convert qiskit code to qsearch *structure* but not parameters"""
    registers = []
    circuit = [KroneckerStep(*[QiskitU3QubitStep() for _ in range(circ.num_qubits)])]
    for gate, qubits, cbits in circ.data:
        if gate.name == "cx":
            if cbits != []:
                raise QiskitImportError("Classical operations are not supported in qsearch for now.")
            for q in qubits:
                if q.register.name not in registers:
                    if len(registers) == 0:
                        registers.append(q.register.name)
                    else:
                        raise QiskitImportError("Qsearch does not support importing circuits with multiple quantum registers.")
            qubit_numbers = [q.index for q in qubits]
            if len(qubit_numbers) != 2:
                raise QiskitImportError("Qsearch can only handle operations that happen between")
            circuit.append(new_layer(gate, qubit_numbers, circ.num_qubits))
    return ProductStep(*circuit)
