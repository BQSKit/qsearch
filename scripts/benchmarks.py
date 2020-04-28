import search_compiler as sc
from search_compiler import unitaries, advanced_unitaries
import time
project = sc.Project("benchmarks")
project.add_compilation("qft2", unitaries.qft(4))
project.add_compilation("qft3", unitaries.qft(8))
project.add_compilation("fredkin", unitaries.fredkin)
project.add_compilation("toffoli", unitaries.toffoli)
project.add_compilation("peres", unitaries.peres)
project.add_compilation("or", unitaries.logical_or)

project.add_compilation("miro", advanced_unitaries.mirogate)
project.add_compilation("hhl", advanced_unitaries.HHL)

# 4 qubit benchmarks (WARNING: These may take days to run.)
#project.add_compilation("qft4", unitaries.qft(16))
#project.add_compilation("full adder", unitaries.full_adder)
#project.add_compilation("ethylene", advanced_unitaries.ethylene)


# compiler configuration example
#project["gateset"] = sc.gatesets.QubitCNOTRing() # use this to synthesize for the ring topology instead of the default line topology
project["solver"]  = sc.solver.BFGS_Jac_SolverNative()

times = {}
for compilation in project.compilations():
    times[compilation] = 0
for _ in range(10):
    project.reset()
    project.run()
    for compilation in project.compilations():
        times[compilation] += project.get_time(compilation)
for compilation in project.compilations():
    print(f'Compilation {compilation} took {times[compilation]/10}s on average.')

