from qsearch import unitaries, gatesets, solver, utils
from qsearch.circuits import CNOTStep

def test_smart_defaults(project, check_project):
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    assert isinstance(project.options.solver, solver.LeastSquares_Jac_Solver)
    assert isinstance(project.options.gateset, gatesets.QubitCNOTLinear)
    assert project.options.eval_func is utils.matrix_distance_squared
    assert project.options.error_func is utils.matrix_distance_squared
    assert project.options.error_jac is utils.matrix_residuals_jac

