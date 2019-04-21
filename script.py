import numpy as np
import os
import shutil
from timeit import default_timer as timer

import search_compiler as sc
from search_compiler import sample_gates as gates
import search_compiler.logging as logging

from qasm_parser import parse_qasm

Compiler_Class = sc.SearchCompiler

def run_compilation(target, name, gateset=sc.gatesets.QubitCNOTLinear(), assemble=False, debug=True):
    directory = "compilations-crz-ring/{}".format(name)
    force = False
    if debug:
        directory = "compilations-DEBUG/{}".format(name)
        force = True
    if os.path.exists(directory):
        if force:
            shutil.rmtree(directory)
        else:
            print("ERROR: File already exists at {}".format(directory))
            return

    os.makedirs(directory)
    logging.output_file = "{}/{}-pylog.txt".format(directory,name)
    logging.logprint("Circuit {} is of size {}".format(name, np.shape(target)))

    compiler = Compiler_Class(threshold=1e-10, gateset=gateset)
    start = timer()
    result, structure, vector = compiler.compile(target, 32)
    end = timer()
    logging.logprint("{} compilation complete!  Duration: {}\n".format(name, end-start))
    
    with open("{}/{}-target.txt".format(directory, name), "w") as outtarget:
        outtarget.write(repr(target))
        print("Recorded target")
    with open("{}/{}-final.txt".format(directory, name),"w") as outmat:
        outmat.write(repr(result))
        print("recorded result")
    with open("{}/{}-structure.txt".format(directory, name),"w") as outpath:
        outpath.write(repr(structure))
        print("recorded structure")
    with open("{}/{}-vector.txt".format(directory, name),"w") as outpath:
        outpath.write(repr(vector))
        print("recore path")
    if assemble:
        with open("{}/{}-qasm.txt".format(directory, name),"w") as outpath:
            outpath.write(structure.assemble(vector))


# add things to do down here
#run_compilation(gates.qft(4), "qft2")
#run_compilation(gates.toffoli, "toffoli")
#run_compilation(gates.fredkin, "fredkin")
#run_compilation(gates.peres, "peres")
run_compilation(gates.qft(8), "qft3")
#run_compilation(gates.qft(8), "qft3-line", gateset=gatesets.QubitCNOTLinear())
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

#run_compilation(mirogate, "miro")


# HHL
from SC_Circuits import *
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
#run_compilation(HHL, "HHL")

