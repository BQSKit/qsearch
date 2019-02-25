import numpy as np
import os
import shutil

from qasm_parser import parse_qasm
import SC_Search_Compiler as comp
import SC_Utils as util
import SC_Sample_Gates as gates
from SC_Logging import logprint
import SC_Logging as logger

def run_compilation(target, name, force=False):
    directory = "compilations-DEBUG/{}".format(name)
    if os.path.exists(directory) and force:
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.output_file = "{}/{}-pylog.txt".format(directory,name)
    logprint("Circuit {} is of size {}".format(name, np.shape(target)))

    compiler = comp.Search_Compiler(threshold=1e-10)
    result, structure, vector = compiler.compile(target, 32)
    logprint("{} compilation complete!\n".format(name))
    
    with open("{}/{}-target.txt".format(directory, name), "w") as outtarget:
        outtarget.write(repr(target))
    with open("{}/{}-final.txt".format(directory, name),"w") as outmat:
        outmat.write(repr(result))
    with open("{}/{}-structure.txt".format(directory, name),"w") as outpath:
        outpath.write(repr(structure))
    with open("{}/{}-vector.txt".format(directory, name),"w") as outpath:
        outpath.write(repr(vector))


# add things to do down here
#run_compilation(gates.qft(4), "qft2")
#run_compilation(gates.toffoli, "toffoli")
#run_compilation(gates.fredkin, "fredkin")
#run_compilation(gates.peres, "peres")
#run_compilation(gates.qft(8), "qft3")
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

run_compilation(gates.qft(4), "test", force=True)

