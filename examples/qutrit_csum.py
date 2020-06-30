import qsearch
from qsearch import gatesets, circuits, solver

p = qsearch.Project("qutrit-csum")
p["gateset"] = gatesets.QutritCPIPhaseLinear()
p["verbosity"] = 2

p.add_compilation("csum", circuits.CSUMStep().matrix([]))
p.run()
