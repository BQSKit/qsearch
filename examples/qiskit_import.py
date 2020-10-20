import qiskit
from qsearch.qiskit import qiskit_to_qsearch
from qsearch import multistart_solvers, utils, Options, defaults, unitaries
from qsearch.assemblers import ASSEMBLER_IBMOPENQASM
from qsrs import native_from_object
import matplotlib.pyplot as plt

import os.path

import qsearch
import numpy as np
from qsearch.gates import *

def compare_gradient(gate):
    totaldiff = [0] * gate.num_inputs
    eps = 1e-5
    repeats = 100
    for _ in range(repeats):
        v = np.random.rand(gate.num_inputs) * 2 * np.pi
        M, Js = gate.mat_jac(v)
        for i in range(gate.num_inputs):
            v2 = np.copy(v)
            v2[i] = v[i] + eps
            U1 = gate.matrix(v2)
            v2[i] = v[i] - eps
            U2 = gate.matrix(v2)
            
            FD = (U1 - U2) / (2*eps)

            diffs = np.sum(np.abs(FD - Js[i]))
            totaldiff[i] += diffs

    for i in range(gate.num_inputs):
        assert totaldiff[i] < eps


if __name__ == '__main__':
    # use the multistart solver for more accurate results, this uses 24 starting points
    solv = multistart_solvers.MultiStart_Solver(24)


    backend = qiskit.BasicAer.get_backend('unitary_simulator')
    # load a qft5 solution
    qc1 = qiskit.QuantumCircuit.from_qasm_file(f'{os.path.dirname(__file__)}/qft5.qasm')
    print("Loaded qft5 circuit!")
    # generate a unitary from the Qiskit circuit
    job = qiskit.execute(qc1, backend)
    U2 = job.result().get_unitary()
    U2 = utils.endian_reverse(U2) # switch from Qiskit endianess qsearch endianess
    # tell the optimizer what we are solving for
    opts = Options()
    opts.target = unitaries.qft(32)

    #qc1.draw(output='mpl')
    #plt.show()
    circ, vec = qiskit_to_qsearch(qc1)
    circ.validate_structure()
    compare_gradient(circ)
    print("Passed checks!")
    for _ in range(100):
        U1, vec = solv.solve_for_unitary(circ, opts)
        dist = utils.matrix_distance_squared(U1, unitaries.qft(32))
        if dist < 1e-10:
            break
    res = {'structure': circ, 'parameters': vec}
    print(dist)
    qasm = ASSEMBLER_IBMOPENQASM.assemble(res, opts)
    qc2 = qiskit.QuantumCircuit.from_qasm_str(qasm)
    qc2.draw(output='mpl')
    plt.show()