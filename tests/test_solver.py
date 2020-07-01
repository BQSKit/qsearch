from qsearch import Project, solver, unitaries, utils, multistart_solver, parallelizer
import scipy as sp
import os


qft3 = unitaries.qft(8)

def test_cobyla(project):
    project.add_compilation('qft2', unitaries.qft(4))
    project['solver'] = solver.COBYLA_Solver()
    project.run()

def test_bfgs_jac(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = solver.BFGS_Jac_Solver()
    project.run()

def test_least_squares_jac(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = solver.LeastSquares_Jac_Solver()
    project['error_func'] = utils.matrix_residuals
    project['error_jac'] = utils.matrix_residuals_jac
    project.run()

def test_multistart_least_squares(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = multistart_solver.MultiStart_Solver(4, 'least_squares')
    project['error_func'] = utils.matrix_residuals
    project['error_jac'] = utils.matrix_residuals_jac
    project['parallelizer'] = parallelizer.ProcessPoolParallelizer
    project.run()

def test_multistart_bfgs(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = multistart_solver.MultiStart_Solver(4, 'BFGS')
    project['parallelizer'] = parallelizer.ProcessPoolParallelizer
    project.run()