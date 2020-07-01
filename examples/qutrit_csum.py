import qsearch
from qsearch import gatesets, circuits, solver

p = qsearch.Project("qutrit-csum")
p["gateset"] = gatesets.QutritCPIPhaseLinear()
p["verbosity"] = 2

p.add_compilation("cob-csum", circuits.CSUMStep().matrix([]), solver=solver.COBYLA_Solver())
p.add_compilation("jac-csum", circuits.CSUMStep().matrix([]), solver=solver.BFGS_Jac_Solver())
p.add_compilation("lss-csum", circuits.CSUMStep().matrix([]), solver=solver.LeastSquares_Jac_Solver())

p.run()
