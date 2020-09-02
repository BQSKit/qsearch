from qsearch import unitaries, gatesets, solver, utils, compiler, backend, Options
from qsearch.circuits import CNOTStep, QuantumStep
import numpy as np
try:
    from qsrs import LeastSquares_Jac_SolverNative
except ImportError:
    LeastSquares_Jac_SolverNative = None

class NoJacQiskitU3QubitStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 3
        self.dits = 1

    def matrix(self, v):
        ct = np.cos(v[0] * np.pi)
        st = np.sin(v[0] * np.pi)
        cp = np.cos(v[1] * np.pi * 2)
        sp = np.sin(v[1] * np.pi * 2)
        cl = np.cos(v[2] * np.pi * 2)
        sl = np.sin(v[2] * np.pi * 2)
        return np.array([[ct, -st * (cl + 1j * sl)], [st * (cp + 1j * sp), ct * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128')

class NoJacCNOTLinear(gatesets.QubitCNOTLinear):
    def __init__(self):
        gatesets.QubitCNOTLinear.__init__(self)
        self.single_step = NoJacQiskitU3QubitStep()

def test_smart_defaults():
    options = Options()
    options.target = unitaries.qft(4)
    options.log_file=None
    c = compiler.SearchCompiler(options=options)
    res = c.compile()
    if LeastSquares_Jac_SolverNative is not None:
        assert isinstance(c.options.solver, LeastSquares_Jac_SolverNative)
    else:
        assert isinstance(c.options.solver, solver.LeastSquares_Jac_Solver)
    assert isinstance(c.options.backend, backend.SmartDefaultBackend)
    assert not isinstance(c.options.backend.prepare_circuit(c.options.gateset.initial_layer(2), c.options), QuantumStep)
    assert isinstance(c.options.gateset, gatesets.QubitCNOTLinear)
    assert c.options.eval_func is utils.matrix_distance_squared
    assert c.options.error_func is utils.matrix_distance_squared
    assert c.options.error_jac is utils.matrix_distance_squared_jac
    assert c.options.error_residuals is utils.matrix_residuals
    assert c.options.error_residuals_jac is utils.matrix_residuals_jac

def test_no_grad_gateset():
    options = Options()
    options.target = unitaries.qft(4)
    options.log_file = None
    options.gateset = NoJacCNOTLinear()
    c = compiler.SearchCompiler(options=options)
    res = c.compile()
    assert isinstance(c.options.solver, solver.COBYLA_Solver)
    assert isinstance(c.options.backend, backend.SmartDefaultBackend)
    assert isinstance(c.options.backend.prepare_circuit(c.options.gateset.initial_layer(2), c.options), QuantumStep)
    assert isinstance(c.options.gateset, NoJacCNOTLinear)
    assert c.options.eval_func is utils.matrix_distance_squared
    assert c.options.error_func is utils.matrix_distance_squared
    assert c.options.error_jac is utils.matrix_distance_squared_jac
    assert c.options.error_residuals is utils.matrix_residuals
    assert c.options.error_residuals_jac is utils.matrix_residuals_jac
