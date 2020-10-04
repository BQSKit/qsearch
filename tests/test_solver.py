from qsearch import Project, solvers, unitaries, utils, multistart_solvers, parallelizers, compiler, options
import scipy as sp
import os
try:
    from qsrs import BFGS_Jac_SolverNative, LeastSquares_Jac_SolverNative
except ImportError:
    BFGS_Jac_SolverNative = None
    LeastSquares_Jac_SolverNative = None

import pytest
import tempfile
import os

qft3 = unitaries.qft(8)

def test_cobyla(project):
    project.add_compilation('qft2', unitaries.qft(4))
    project['solver'] = solvers.COBYLA_Solver()
    project.run()

def test_bfgs_jac(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = solvers.BFGS_Jac_Solver()
    project.run()

def test_least_squares_jac(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = solvers.LeastSquares_Jac_Solver()
    project['error_func'] = utils.matrix_residuals
    project['error_jac'] = utils.matrix_residuals_jac
    project.run()

def test_multistart_least_squares(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = multistart_solvers.MultiStart_Solver(2)
    project['inner_solver'] = solvers.LeastSquares_Jac_Solver()
    project['parallelizer'] = parallelizers.ProcessPoolParallelizer
    project['error_func'] = utils.matrix_residuals
    project['error_jac'] = utils.matrix_residuals_jac
    project.run()

def test_multistart_bfgs(project):
    project.add_compilation('qft3', qft3)
    project['solver'] = multistart_solvers.MultiStart_Solver(2)
    project['inner_solver'] = solvers.BFGS_Jac_Solver()
    project['parallelizer'] = parallelizers.ProcessPoolParallelizer
    project.run()

def compile(U, solver):
    with tempfile.TemporaryDirectory() as dir:
        opts = options.Options()
        opts.target = U
        opts.error_func = utils.matrix_distance_squared
        opts.error_jac = utils.matrix_distance_squared_jac
        opts.solver = solver
        opts.log_file = os.path.join(dir, 'test.log')
        comp = compiler.SearchCompiler()
        res = comp.compile(opts)
    return res

@pytest.mark.skipif(BFGS_Jac_SolverNative is None, reason="The rustopt feature has not been enabled")
def test_rust_solver_qft3():
    U = unitaries.qft(8)
    res = compile(U, BFGS_Jac_SolverNative())
    circ = res['structure']
    v = res['vector']
    assert utils.matrix_distance_squared(U, circ.matrix(v)) < 1e-10

@pytest.mark.skipif(LeastSquares_Jac_SolverNative is None, reason="The rustopt feature has not been enabled")
def test_rust_solver_qft3():
    U = unitaries.qft(8)
    res = compile(U, LeastSquares_Jac_SolverNative())
    circ = res['structure']
    v = res['vector']
    assert utils.matrix_distance_squared(U, circ.matrix(v)) < 1e-10
