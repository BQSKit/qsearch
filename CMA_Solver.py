import numpy as np
import cma

import CMA_Utils as util

class QuantumStep:
    def __init__(self):
        raise NotImplementedError("Subclasses of QuantumStep should declare their own initializers.")
        # NOTE: QuantumStep initializers must set self._num_inputs
    
    def matrix(self, v):
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the matrix(v) method.")

    def path(self, v):
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the path(v) method.")

    def assemble(self, v, i=0):
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the assemble(v, i) method.")

    def copy(self):
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def solve_for_unitary(self, U, error_func=util.matrix_distance_squared, error_target=0.01):
        eval_func = lambda v: error_func(U, self.matrix(v))
        xopt, _ = cma.fmin2(eval_func, 'np.random.rand({})*np.pi'.format(self._num_inputs), np.pi/4, {'verb_disp':0, 'verb_log':0}, restarts=2)
        return (self.matrix(xopt), xopt)

    def __repr__(self):
        return "QuantumStep()"


class IdentityStep:  
    def __init__(self, n, dits=1):
        self._num_inputs=0
        self._I = np.matrix(np.eye(n), dtype='complex128')
        self._dits = dits

    def matrix(self, v):
        return self._I

    def path(self, v):
        return ["IDENTITY"]

    def assemble(self, v, i=0):
        return ""

    def __repr__(self):
        return "IdentityStep()"
    


class SingleQubitStep(QuantumStep):
    def __init__(self):
        self._num_inputs = 3
        self._dits = 1

    def matrix(self, v):
        return util.Q1_unitary(v)

    def path(self, v):
        return ["QUBIT", list(v)]

    def assemble(self, v, i=0):
        # once you are done with the basics, expand this into its several steps
        return "Z({}) q{}\nX(pi/2) q{}\nZ({}) q{}\nX(pi/2) q{}\nZ({}) q{}\n".format(v[0], i, i, v[1] + np.pi, i, i, v[2]-np.pi, i)
    
    def __repr__(self):
        return "SingleQubitStep()"


class SingleQutritStep(QuantumStep):
    def __init__(self):
        self._num_inputs = 8
        self._dits = 1

    def matrix(self, v):
        return util.qt_arb_rot(*v)

    def path(self, v):
        return ["QUTRIT", list(v)]

    def assemble(self, v, i=0):
        return "U({}, {}, {}, {}, {}, {}, {}, {}) q{}".format(*v, i)
    
    def __repr__(self):
        return "SingleQutritStep()"

class UStep(QuantumStep):
    def __init__(self, U, name=None, dits=1):
        self.name = name
        self._num_inputs = 0
        self._U = U
        self._dits = dits

    def matrix(self, v):
        return self._U

    def path(self, v):
        if self.name is None:
            return ["CUSTOM", self._U]
        else:
            return [name]

    def assemble(self, v, i=0):
        if self.name is None:
            return "UNKNOWN q{}".format(i)
        else:
            return "{} q{}".format(self.name, i)

    def __repr__(self):
        if self.name is None:
            return "UStep({})".format(repr(self._U))
        else:
            return "UStep({}, name={})".format(repr(self._U), repr(self.name))

class CUStep(QuantumStep):
    def __init__(self, U, name=None, flipped=False):
        self.name = name
        self.flipped = flipped
        self._num_inputs = 0
        self._U = U
        n = np.shape(U)[0]
        I = np.matrix(np.eye(n))
        top = np.pad(self._U if flipped else I, [(0,n),(0,n)], 'constant')
        bot = np.pad(I if flipped else self._U, [(n,0),(n,0)], 'constant')
        self._CU = np.matrix(top + bot)
        self._dits = 2

    def matrix(self, v):
        return self._CU

    def path(self, v):
        if self.name is None:
            return [("FLIPPED " if self.flipped else "") + "C-CUSTOM", self._U]
        else:
            return [self.name]

    def assemble(self, v, i=0):
        first = i+1 if self.flipped else i
        second = i if self.flipped else i+1
        if self.name is None:
            return "CONTROLLED-UNKNOWN q{} q{}".format(first, second)
        else:
            return "C{} q{} q{}".format(self.name, first, second)

    def __repr__(self):
        return "CUStep(" + str(repr(self._U)) + ("" if self.name is None else ", name={}".format(repr(self.name))) + ("flipped=True" if self.flipped else "") + ")"

class InvertStep(QuantumStep):
    def __init__(self, step):
        self._step = step
        self._num_inputs = step._num_inputs
        self._dits = step._dits

    def matrix(self, v):
        return self._step.matrix(v).H

    def path(self, v):
        return ["INVERTED", self._step.path(v)]

    def assemble(self, v, i=0):
        return "REVERSE {\n" + self._step.assemble(v, i) + "\n}"

    def __repr__(self):
        return "InvertStep({})".format(repr(self._step))


class CSUMStep(QuantumStep):
    _csum =  np.matrix([[1,0,0, 0,0,0, 0,0,0],
                        [0,1,0, 0,0,0, 0,0,0],
                        [0,0,1, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,1, 0,0,0],
                        [0,0,0, 1,0,0, 0,0,0],
                        [0,0,0, 0,1,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,1,0],
                        [0,0,0, 0,0,0, 1,0,0],
                        [0,0,0, 0,0,0, 0,0,1]
                       ], dtype='complex128')
    
    def __init__(self):
        self._num_inputs = 0
        self._dits = 2

    def matrix(self, v):
        return CSUMStep._csum

    def path(self, v):
        return ["CSUM"]

    def assemble(self, v, i=0):
        return "CSUM q{} q{}".format(i, i+1)

    def __repr__(self):
        return "CSUMStep()"

class CPIStep(QuantumStep):
    _cpi = np.matrix([[1,0,0, 0,0,0, 0,0,0],
                      [0,1,0, 0,0,0, 0,0,0],
                      [0,0,1, 0,0,0, 0,0,0],
                      [0,0,0, 0,1,0,0,0,0],
                      [0,0,0, 1,0,0, 0,0,0],
                      [0,0,0, 0,0,1, 0,0,0],
                      [0,0,0, 0,0,0, 1,0,0],
                      [0,0,0, 0,0,0, 0,1,0],
                      [0,0,0, 0,0,0, 0,0,1]
                     ], dtype='complex128')
    
    def __init__(self):
        self._num_inputs = 0
        self._dits = 2

    def matrix(self, v):
        return CPIStep._cpi

    def path(self, v):
        return ["CPI"]

    def assemble(self, v, i=0):
        return "CPI q{} q{}".format(i, i+1)

    def __repr__(self):
        return "CPIStep()"

class CPIPhaseStep(QuantumStep):
    def __init__(self):
        self._num_inputs = 0
        self._cpi = np.matrix([[1,0,0, 0,0,0, 0,0,0],
                               [0,1,0, 0,0,0, 0,0,0],
                               [0,0,1, 0,0,0, 0,0,0],
                               [0,0,0, 0,-1,0,0,0,0],
                               [0,0,0, 1,0,0, 0,0,0],
                               [0,0,0, 0,0,1, 0,0,0],
                               [0,0,0, 0,0,0, 1,0,0],
                               [0,0,0, 0,0,0, 0,1,0],
                               [0,0,0, 0,0,0, 0,0,1]
                              ], dtype='complex128')
        diag_mod = np.matrix(np.diag([1]*4 + [np.exp(2j * np.random.random()*np.pi) for _ in range(0,5)]))
        self._cpi = np.matmul(self._cpi, diag_mod)
        self._dits = 2

    def matrix(self, v):
        return self._cpi

    def path(self, v):
        return ["CPI+"]

    def assemble(self, v, i=0):
        return "CPI- q{} q{}".format(i, i+1)

    def __repr__(self):
        return "CPIPhaseStep()"

class CNOTStep(QuantumStep):
    _cnot = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]])
    def __init__(self):
        self._num_inputs = 0
        self._dits = 2

    def matrix(self, v):
        return CNOTStep._cnot

    def path(self, v):
        return ["CNOT"]

    def assemble(self, v, i=0):
        return "CNOT q{} q{}".format(i, i+1)

    def __repr__(self):
        return "CNOTStep()"

class CQubitStep(QuantumStep):
    _cnr = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0.5+0.5j,0.5-0.5j],
                       [0,0,0.5-0.5j,0.5+0.5j]])
    _I = np.matrix(np.eye(2))
    def __init__(self):
        self._num_inputs = 1
        self._dits = 2

    def matrix(self, v):
        U = np.dot(CQubitStep._cnr, np.kron(CQubitStep._I, util.rot_z(v[0]))) # TODO fix this line
        return np.dot(U, CQubitStep._cnr)

    def path(self, v):
        return ["CQ", v]

    def assemble(self, v, i=0):
        return "CNOTROOT q{} q{}\nZ({}) q{}\nCNOTROOT q{} q{}".format(i, i+1, v[0], i+1, i, i+1)

    def __repr__(self):
        return "CQubitStep()"

class CNOTRootStep(QuantumStep):
    _cnr = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0.5+0.5j,0.5-0.5j],
                       [0,0,0.5-0.5j,0.5+0.5j]])
    def __init__(self):
        self._num_inputs = 0
        self._dits = 2

    def matrix(self, v):
        return CNOTRootStep._cnr

    def path(self, v):
        return ["CNOTROOT"]

    def assemble(self, v, i=0):
        return "CNOTROOT q{} q{}".format(i, i+1)

    def __repr__(self):
        return "CNOTRootStep()"

class KroneckerStep(QuantumStep):
    def __init__(self, *substeps):
        self._num_inputs = sum([step._num_inputs for step in substeps])
        self._substeps = substeps
        self._dits = sum([step._dits for step in substeps])

    def matrix(self, v):
        matrices = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step._num_inputs])
            matrices.append(U)
            index += step._num_inputs
        U = matrices[0]
        for matrix in matrices[1:]:
            U = np.kron(U, matrix)
        return U

    def path(self, v):
        paths = ["KRON"]
        index = 0
        for step in self._substeps:
            p = step.path(v[index:index+step._num_inputs])
            paths.append(p)
            index += step._num_inputs
        return paths

    def assemble(self, v, i=0):
        outstr = ""
        index = 0
        for step in self._substeps:
            outstr += step.assemble(v[index:index+step._num_inputs], i) + "\n"
            index += step._num_inputs
            i += step._dits

        return outstr


    def appending(self, step):
        return KroneckerStep(*self._substeps, step)

    def __deepcopy__(self, memo):
        return KroneckerStep(self._substeps.__deepcopy__(memo))

    def __repr__(self):
        return "KroneckerStep({})".format(repr(self._substeps)[1:-1])

class ProductStep(QuantumStep):
    def __init__(self, *substeps):
        self._num_inputs = sum([step._num_inputs for step in substeps])
        self._substeps = substeps
        self._dits = 0 if len(substeps) == 0 else substeps[0]._dits

    def matrix(self, v):
        matrices = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step._num_inputs])
            matrices.append(U)
            index += step._num_inputs
        U = matrices[0]
        for matrix in matrices[1:]:
            U = np.matmul(U, matrix)
        return U

    def path(self, v):
        paths = ["PRODUCT"]
        index = 0
        for step in self._substeps:
            p = step.path(v[index:index+step._num_inputs])
            paths.append(p)
            index += step._num_inputs
        return paths

    def assemble(self, v, i=0):
        outstr = ""
        index = 0
        for step in self._substeps:
            outstr += step.assemble(v[index:index+step._num_inputs], i) + "\n"
            index += step._num_inputs

        return outstr

    def appending(self, *steps):
        return ProductStep(*self._substeps, *steps)

    def __deepcopy__(self, memo):
        return ProductStep(self._substeps.__deepcopy__(memo))

    def __repr__(self):
        return "ProductStep({})".format(repr(self._substeps)[1:-1])


def decode_path(path, d=2, args=None):
    if args is None:
        args = dict()
    if len(path) < 1:
        return (None,[])
    if path[0] == "KRON":
        k = []
        vf = []
        for item in path[1:]:
           (step, v) = decode_path(item, d, args)
           k.append(step)
           vf.extend(v)
        return (KroneckerStep(*k), vf)

    elif path[0] == "PRODUCT":
        p = []
        vf = []
        for item in path[1:]:
           (step, v) = decode_path(item, d, args)
           p.append(step)
           vf.extend(v)
        return (ProductStep(*p), vf)

    elif not path[0] in args:
        if path[0] == "IDENTITY":
            args[path[0]] = IdentityStep(d)
        elif path[0] == "QUBIT":
            args[path[0]] = SingleQubitStep()
        elif path[0] == "QUTRIT":
            args[path[0]] = SingleQutritStep()
        elif path[0] == "CNOT":
            args[path[0]] = CNOTStep()
        elif path[0] == "CQ":
            args[path[0]] = CQubitStep()
    return (args[path[0]], path[1] if len(path) > 1 else [])


