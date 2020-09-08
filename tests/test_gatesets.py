from qsearch import gatesets, unitaries, advanced_unitaries

def test_qubit_cnot_linear(project, check_project):
    project['gateset'] = gatesets.QubitCNOTLinear()
    project.add_compilation('fredkin', unitaries.fredkin)
    project.run()
    check_project(project)

def test_zxzxzcnotlinear(project, check_project):
    project['gateset'] = gatesets.ZXZXZCNOTLinear()
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    check_project(project)

def test_qiskitu3linear(project, check_project):
    project['gateset'] = gatesets.QiskitU3Linear()
    project.add_compilation('hhl', advanced_unitaries.HHL)
    project.run()
    check_project(project)

def test_cnotring(project, check_project):
    project['gateset'] = gatesets.QubitCNOTRing()
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    check_project(project)

def test_adjacency_list(project, check_project):
    project['gateset'] = gatesets.QubitCNOTAdjacencyList([(0,1), (1,2), (2,3), (2,0)])
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    check_project(project)
