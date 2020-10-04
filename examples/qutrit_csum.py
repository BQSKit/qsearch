import qsearch
from qsearch import gatesets, circuits, solvers, multistart_solverss, parallelizers

qttof = qsearch.utils.upgrade_dits(qsearch.unitaries.toffoli)

p1 = qsearch.Project("qutrit-cpp")
p1["solver"] = solvers.BFGS_Jac_Solver()
p1["gateset"] = gatesets.QutritCPIPhaseLinear()
p1["verbosity"] = 2

p1.add_compilation("csum", circuits.CSUMStep().matrix([]))
#p1.add_compilation("toffoli", qttof, depth=8)

p1.run()

p2 = qsearch.Project("qutrit-cnot")
p2["gateset"] = gatesets.QutritCNOTLinear()
p2["solver"] = multistart_solverss.MultiStart_Solver(12, "least_squares")
p2["parallelizer"] = parallelizers.ProcessPoolParallelizer
p2["verbosity"] = 2
p2.add_compilation("csum", circuits.CSUMStep().matrix([]))
#p2.add_compilation("toffoli", qttof, depth=8)
p2.add_compilation("bswap", qsearch.utils.upgrade_dits(qsearch.unitaries.swap))
p2.add_compilation("tswap", qsearch.unitaries.general_swap(3))
p2.add_compilation("qft2t", qsearch.unitaries.qft(9))
p2.add_compilation("qft3t", qsearch.unitaries.qft(27))
p2.add_compilation("qft3b", qsearch.utils.upgrade_dits(qsearch.unitaries.qft(8)))

p2.run()
