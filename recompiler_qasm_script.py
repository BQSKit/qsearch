import numpy as np

from qasm_parser import parse_qasm
import CMA_Search_Compiler as comp
import CMA_Utils as util

target, circuit = parse_qasm("ethylene-4.qasm")
print("Parsed the input file")
print("Circuit is of size {}".format(np.shape(target)))

compiler = comp.CMA_Search_Compiler(threshold=1e-10)
result, path = compiler.compile(target, 72)
print("complete!")

with open("final-matrix.txt","w") as outmat:
    outmat.write(str(result))
with open("final-path.txt","w") as outpath:
    outpath.write(str(path))
