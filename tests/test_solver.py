from qsearch import Project, solver, unitaries, utils
import scipy as sp
import os


qft3 = unitaries.qft(8)

def test_cobyla():
    if not os.path.isdir('.test'):
        os.mkdir('.test')
    p = Project('.test/cobyla-solver')
    p.clear()
    p.add_compilation('qft2', unitaries.qft(4))
    p['solver'] = solver.COBYLA_Solver()
    p.run()

def test_bfgs_jac():
    if not os.path.isdir('.test'):
        os.mkdir('.test')
    p = Project('.test/bfgs-jac-solver')
    p.clear()
    p.add_compilation('qft3', qft3)
    p['solver'] = solver.BFGS_Jac_Solver()
    p.run()

def test_least_squares_jac():
    if not os.path.isdir('.test'):
        os.mkdir('.test')
    p = Project('.test/lm-jac-solver')
    p.clear()
    p.add_compilation('qft3', qft3)
    p['solver'] = solver.LeastSquares_Jac_Solver()
    p['error_func'] = utils.matrix_residuals
    p['error_jac'] = utils.matrix_residuals_jac
    p.run()

if __name__ == '__main__':
    test_cobyla()
    test_bfgs_jac()
    test_least_squares_jac()
