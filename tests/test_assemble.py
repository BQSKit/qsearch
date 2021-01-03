from qsearch.assemblers import ASSEMBLER_IBMOPENQASM, ASSEMBLER_OPENQASM, ASSEMBLER_QISKIT
from qsearch.gates import *
from qsearch import utils
import pytest
try:
    from qiskit import QuantumCircuit
    import qiskit
except Exception:
    qiskit = None

circ = ProductGate(KroneckerGate(U3Gate(), U3Gate()), KroneckerGate(ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())),), KroneckerGate(
    ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())),), KroneckerGate(ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())),))

parameters = np.array([5.00000000e-01, 6.68212734e-01, 6.68212734e-01, 2.25848361e+00,
                1.68291228e+00, 2.75796050e+00, 1.25000000e+00, 1.06173194e+00,
                2.24071513e+00, 4.68541330e-04, 2.62313007e-01, 6.38346363e-01,
                1.58982244e+00, 1.54075164e+00, 2.78047447e+00, 1.02950391e+00,
                1.36762044e-01, 2.25000000e+00, 2.50000000e-01, 2.83496183e+00,
                5.00000000e-01])

cdict = {"structure":circ, "parameters":parameters}

def unitary_from_openqasm(qasm_str):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    backend = qiskit.BasicAer.get_backend('unitary_simulator')
    job = qiskit.execute(qc, backend)
    U = job.result().get_unitary()
    BE = utils.endian_reverse(U)
    return BE

@pytest.mark.skipif(qiskit is None, reason="Qiskit not installed")
def test_openqasm_assemble_roundtrip():
    s = ASSEMBLER_IBMOPENQASM.assemble(cdict)
    qc = QuantumCircuit.from_qasm_str(s)
    backend = qiskit.BasicAer.get_backend('unitary_simulator')
    job = qiskit.execute(qc, backend)
    U = job.result().get_unitary()
    BE = utils.endian_reverse(U)
    assert utils.matrix_distance_squared(BE, circ.matrix(parameters)) < 1e-10

ASSEMBLER_GATES = (
    XGate(),
    YGate(),
    ZGate(),
    U3Gate(),
    U2Gate(),
    U1Gate(),
    CNOTGate(),
    CZGate(),
    XXGate(),
    ISwapGate(),
)

@pytest.mark.skipif(qiskit is None, reason="Qiskit not installed")
@pytest.mark.parametrize("gate", ASSEMBLER_GATES, ids=lambda gate: repr(gate))
def test_assemble_ibmopenqasm(gate):
    parameters = np.random.uniform(size=(max(gate.num_inputs, 1),))
    cdict = {"structure": gate, "parameters": parameters}
    s = ASSEMBLER_IBMOPENQASM.assemble(cdict)
    U = unitary_from_openqasm(s)
    assert utils.matrix_distance_squared(U, gate.matrix(parameters)) < 1e-10
