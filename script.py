import numpy as np

import search_compiler as sc
from search_compiler import sample_gates as gates

from qasm_parser import parse_qasm

project = sc.Project("newscript-test")
project.add_compilation("qft2", gates.qft(4), handle_existing="ignore")
project.add_compilation("qft3", gates.qft(8), handle_existing="ignore")
project.add_compilation("qft4", gates.qft(16), handle_existing="ignore")

project.status()
project.run()
project.status()

