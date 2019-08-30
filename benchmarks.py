import search_compiler as sc
from search_compiler import gates, advanced_gates

project = sc.Project("benchmarks")
project.add_compilation("qft2", gates.qft(4))
project.add_compilation("qft3", gates.qft(8))
project.add_compilation("fredkin", gates.fredkin)
project.add_compilation("toffoli", gates.toffoli)
project.add_compilation("peres", gates.peres)
project.add_compilation("logical or", gates.logical_or)

project.add_compilation("miro", advanced_gates.mirogate)
project.add_compilation("hhl", advanced_gates.HHL)

# 4 qubit benchmarks (WARNING: These may take days to run.)
#project.add_compilation("qft4", gates.qft(16))
#project.add_compilation("full adder", gates.full_adder)
#project.add_compilation("ethelyne", advanced_gates.ethelyne)


# compiler configuration
project["beams"] = -1 # tell the compiler to use all available processors by using beams (this is usually a speedup, but not always)

project.run()

