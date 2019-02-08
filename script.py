import numpy as np
import os

from qasm_parser import parse_qasm
import CMA_Search_Compiler as comp
import CMA_Utils as util
import SampleGates as gates
from CMA_Logging import logprint
import CMA_Logging as logger

def run_compilation(target, name):
    directory = "compilations/{}".format(name)
    os.makedirs(directory)
    logger.output_file = "{}/{}-pylog.txt".format(directory,name)
    logprint("Circuit {} is of size {}".format(name, np.shape(target)))

    compiler = comp.CMA_Search_Compiler(threshold=1e-10)
    result, path = compiler.compile(target, 32)
    logprint("{} compilation complete!\n".format(name))
    
    with open("{}/{}-target.txt".format(directory, name), "w") as outtarget:
        outtarget.write(repr(target))
    with open("{}/{}-final.txt".format(directory, name),"w") as outmat:
        outmat.write(repr(result))
    with open("{}/{}-path.txt".format(directory, name),"w") as outpath:
        outpath.write(repr(path))


# add things to do down here
run_compilation(gates.qft(4), "qft2")
run_compilation(gates.toffoli, "toffoli")
run_compilation(gates.fredkin, "fredkin")
run_compilation(gates.peres, "peres")
run_compilation(gates.qft(8), "qft3")

