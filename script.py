import numpy as np

import search_compiler as sc
from search_compiler import gates, advanced_gates

#from qasm_parser import parse_qasm

project = sc.Project("paper-benchmarks")
project.add_compilation("qft2", gates.qft(4))
project.add_compilation("qft3", gates.qft(8))
project.add_compilation("fredkin", gates.fredkin)
project.add_compilation("toffoli", gates.toffoli)
project.add_compilation("peres", gates.peres)
project.add_compilation("logical or", gates.logical_or)

project.add_compilation("miro", advanced_gates.mirogate)
project.add_compilation("hhl", advanced_gates.HHL)

# 4 qubit stuff here
project.add_compilation("qft4", gates.qft(16))
project.add_compilation("full adder", gates.full_adder)
project.add_compilation("ethelyne", advanced_gates.ethelyne)


# compiler configuration here
project["beams"] = -1

project.run()

