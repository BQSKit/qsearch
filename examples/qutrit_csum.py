import qsearch
from qsearch import gatesets, circuits, solver, multistart_solver, parallelizer

qttof = qsearch.utils.upgrade_dits(qsearch.unitaries.toffoli)

p1 = qsearch.Project("qutrit-cpp")
p1["solver"] = solver.BFGS_Jac_Solver()
p1["gateset"] = gatesets.QutritCPIPhaseLinear()
p1["verbosity"] = 2

p1.add_compilation("csum", circuits.CSUMStep().matrix([]))
#p1.add_compilation("toffoli", qttof, depth=8)

p1.run()

p2 = qsearch.Project("qutrit-cnot")
p2["gateset"] = gatesets.QutritCNOTLinear()
p2["solver"] = multistart_solver.MultiStart_Solver(12, "least_squares")
p2["parallelizer"] = parallelizer.ProcessPoolParallelizer
p2["verbosity"] = 2
p2.add_compilation("csum", circuits.CSUMStep().matrix([]))
#p2.add_compilation("toffoli", qttof, depth=8)

p2.run()
