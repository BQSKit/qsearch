import numpy as np

try:
    import qiskit
    from qiskit import Aer, execute
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
except Exception:
    qiskit = None

import qsearch
from qsearch import Project, defaults, leap_compiler, solvers, utils, post_processing, multistart_solvers, parallelizers

import pytest

STATEVECTOR_SIMULATOR = Aer.get_backend('statevector_simulator')


def simulate_statevector(qc):
    result = execute(qc, STATEVECTOR_SIMULATOR).result()
    return np.array(result.get_statevector(qc))



def qsearch_stateprep(project, target_state):
    stateprep_options = qsearch.Options(target_state=target_state,smart_defaults=qsearch.defaults.stateprep_smart_defaults)
    project.add_compilation("stateprep",stateprep_options.target,options=stateprep_options)
    project.run()
    qiskit_code = project.assemble("stateprep", assembler=qsearch.assemblers.ASSEMBLER_QISKIT)
    locals = {}
    exec(qiskit_code, globals(), locals)
    return locals['qc']
    
STATES = map(np.array, (
    [1/np.sqrt(2),-1/np.sqrt(2)],
    [0.5,0.5,0.5,0,0,0,0,0.5],
    [0,0.5,0.5j,0,-0.5,0,0,0,-0.5j,0,0,0,0,0,0,0]
))

@pytest.mark.skipif(qiskit is None, reason="Qiskit is not installed")
@pytest.mark.parametrize("target_state", STATES, ids=lambda state: repr(state[:5]))
def test_stateprep_roundtrip(project, target_state):
    qc = qsearch_stateprep(project, utils.endian_reverse(target_state))
    output_state = simulate_statevector(qc)
    num_qubits = int(np.log2(target_state.shape[0]))
    qc2 = QuantumCircuit(num_qubits)
    qc2.initialize(target_state, list(range(num_qubits)))
    for i in range(num_qubits):
        qc2.id(i)
    output_state_ibm = simulate_statevector(qc2)
    assert np.isclose(np.abs(np.vdot(output_state,output_state_ibm)),1)
