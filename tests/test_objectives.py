from qsearch import utils, unitaries, Project, solvers
import qsrs
import numpy as np

qft_py = unitaries.qft(8)
qft_rs = qsrs.qft(8)

def test_qft():
    assert utils.matrix_distance_squared(qft_py, qft_rs) < 1e-14

def test_matrix_distance_squared():
    assert qsrs.matrix_distance_squared(qft_py, qft_rs) < 1e-14

def test_matrix_distance_squared_jac(tmpdir):
    p = Project(str(tmpdir))
    p.add_compilation('qft2', qft_py)
    p.run()
    res = p.get_result('qft2')
    c = res['structure']
    v = res['vector']
    u, jacs = c.mat_jac(v)
    f_rs, jacs_rs = qsrs.matrix_distance_squared_jac(qft_py, u, jacs)
    assert f_rs < 1e-14
    f_py, jacs_py = utils.matrix_distance_squared_jac(qft_py, u, jacs)
    for (j_rs, j_py) in zip(jacs_rs, jacs_py):
        assert np.allclose(j_rs, j_py), print(j_py) or print(j_rs)

def test_matrix_residuals():
    I = np.eye(qft_py.shape[0])
    np.allclose(qsrs.matrix_residuals(qft_py, qft_rs, I), utils.matrix_residuals(qft_py, qft_rs, I))

def test_matrix_residuals_jac(tmpdir):
    p = Project(str(tmpdir))
    p.add_compilation('qft2', qft_py)
    p.run()
    res = p.get_result('qft2')
    c = res['structure']
    v = res['vector']
    u, jacs = c.mat_jac(v)
    j_rs = qsrs.matrix_residuals_jac(qft_py, u, jacs)
    j_py = utils.matrix_residuals_jac(qft_py, u, jacs)
    assert np.allclose(j_rs, j_py), print(j_py) or print(j_rs)

def test_project(tmpdir):
    p = Project(str(tmpdir))
    p.add_compilation('qft2', qft_py)
    p['solver'] = solvers.LeastSquares_Jac_Solver()
    p['residuals_func'] = qsrs.matrix_residuals
    p['residuals_jac'] = qsrs.matrix_residuals_jac
    p.run()
