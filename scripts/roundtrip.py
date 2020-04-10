##################################################
#               ROUND TRIP TEST                  #
# This is a variant of the benchmarks script     #
# includes a check that the original matrix is   #
# reproduced after assembling to Qiskit, using   #
# the tools provided in Qiskit to produce a      #
# matrix, and comparing that to the original     #
# target, after accounting for endianness.       #
#                                                #
# It tests many parts of the code using Qiskit   #
# as a third party to compare to, and            #
# demonstrates the capability to take matrices   #
# from sources like Qiskit as well as export     #
# circuits to third parties like Qiskit.         #
#                                                #
# Note that Qiskit is required for this script.  #
##################################################

import search_compiler as sc
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from search_compiler import unitaries, advanced_unitaries, utils, gatesets

# create a new project
project = sc.Project("round-trip")

# add some gates to compile
project.add_compilation("qft2", unitaries.qft(4))
project.add_compilation("qft3", unitaries.qft(8))
project.add_compilation("fredkin", unitaries.fredkin)
project.add_compilation("toffoli", unitaries.toffoli)
project.add_compilation("peres", unitaries.peres)
project.add_compilation("logical or", unitaries.logical_or)

project.add_compilation("miro", advanced_unitaries.mirogate)
project.add_compilation("hhl", advanced_unitaries.HHL)

# 4 qubit benchmarks (WARNING: These may take days to run.)
#project.add_compilation("qft4", gates.qft(16))
#project.add_compilation("full adder", gates.full_adder)
#project.add_compilation("ethelyne", advanced_gates.ethelyne)

# run the project
project.run()

# prepare a Qiskit backend to generate unitaries
backend = qiskit.BasicAer.get_backend('unitary_simulator')

for compilation in project.compilations():
    # get the original target unitary and final 
    U1 = project.get_target(compilation)

    # generate and run Qiskit code to create a Qiskit version of the circuit
    qiskit_code = project.assemble(compilation, sc.assembler.ASSEMBLY_QISKIT)
    exec(qiskit_code)

    # generate a unitary from the Qiskit circuit
    job = qiskit.execute(qc, backend)
    U2 = job.result().get_unitary()
    U2 = sc.utils.endian_reverse(U2) # switch from Qiskit endianess search_compiler endianess

    # Compare the two unitaries and print the result.  The values should be close to 0.
    print("Match for {}: {}".format(compilation, sc.utils.matrix_distance_squared(U1, U2)))

