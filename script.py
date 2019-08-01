import numpy as np

import search_compiler as sc
from search_compiler import  gates

#from qasm_parser import parse_qasm

project = sc.Project("paper-benchmarks")
project.add_compilation("qft2", gates.qft(4))
project.add_compilation("qft3", gates.qft(8))
project.add_compilation("fredkin", gates.fredkin)
project.add_compilation("toffoli", gates.toffoli)
project.add_compilation("peres", gates.peres)
project.add_compilation("logical or", gates.logical_or)

theta = np.pi/3
c = np.cos(theta/2)
s = -1j*np.sin(theta/2)
mirogate = np.matrix([
    [c,0,0,0,0,0,0,s],
    [0,c,0,0,0,0,s,0],
    [0,0,c,0,0,s,0,0],
    [0,0,0,c,s,0,0,0],
    [0,0,0,s,c,0,0,0],
    [0,0,s,0,0,c,0,0],
    [0,s,0,0,0,0,c,0],
    [s,0,0,0,0,0,0,c]
    ], dtype='complex128')

from search_compiler.circuits import *
import scipy as sp
def hadamard(theta=0):
    return np.matrix([[np.cos(2*theta), np.sin(2*theta)],[np.sin(2*theta), -np.cos(2*theta)]])
H = UStep(hadamard(), "H")
RCH8 = CUStep(hadamard(np.pi/8), "H8", flipped=True)
RCH16 = CUStep(hadamard(np.pi/16), "H16", flipped=True)
RCY = CUStep(np.matrix([[0,-1j],[1j,0]]), "CY", flipped=True)
RCNOT = UStep(np.matrix([[0,1,0,0],
                        [1,0,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]), "RCNOT")
SWAP = UStep(np.matrix([[1,0,0,0],
                        [0,0,1,0],
                        [0,1,0,0],
                        [0,0,0,1]]))

# input parameters
t0 = 2*np.pi
A = np.matrix([[1.5, 0.5],[0.5, 1.5]])
AU = sp.linalg.expm(1j * t0 * A / 2)

CAU = CUStep(AU, "CA")
CSH = CUStep(np.matrix([[1,0],[0,-1j]]), "CSH")

X = UStep(np.matrix([[0,1],[1,0]]), "X")
I = IdentityStep(2)

circuit = ProductStep()
circuit = circuit.appending(KroneckerStep(I,I,H))
circuit = circuit.appending(KroneckerStep(I,CSH))
circuit = circuit.appending(KroneckerStep(I,H,I))
circuit = circuit.appending(KroneckerStep(I,I,H))
circuit = circuit.appending(KroneckerStep(RCY, I))
circuit = circuit.appending(KroneckerStep(SWAP,I))
circuit = circuit.appending(KroneckerStep(I,RCY))
circuit = circuit.appending(KroneckerStep(SWAP,I))




HHL = circuit.matrix([])

project.add_compilation("miro", mirogate)
project.add_compilation("hhl", HHL)

# 4 qubit stuff here
project.add_compilation("qft4", gates.qft(16))
project.add_compilation("full adder", gates.full_adder)


# compiler configuration here
project["beams"] = -1

project.run()

