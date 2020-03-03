import numpy as np
from . import utils, graphics, unitaries
from hashlib import md5

try:
    from search_compiler_rs import native_from_object
except ImportError:
    native_from_object = None

class QuantumStep:
    def __init__(self):
        raise NotImplementedError("Subclasses of QuantumStep should declare their own initializers.")
        # NOTE: QuantumStep initializers must set self.num_inputs, self.dits
    
    def matrix(self, v):
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the matrix(v) method.")

    def assemble(self, v, i=0):
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the assemble(v, i) method.")

    def draw(self):
        gates = self._draw_assemble()
        labels = ["q{}".format(i) for i in range(0, self.dits)]
        return graphics.plot_quantum_circuit(gates, labels=labels, plot_labels=False)

    def jac(self, v):
        if self.num_inputs == 0:
            return [] # a circuit component with no inputs has no jacobian to return
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the jac(v) method in order to be used with gradient optimizers.")

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return int(md5(repr(self).encode()).hexdigest(), 16)

    def _draw_assemble(self, i=0):
        return []

    def copy(self):
        return self

    def _parts(self):
        return [self]

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return "QuantumStep()"


class IdentityStep(QuantumStep):
    def __init__(self, n=2, dits=1):
        self.num_inputs=0
        self._I = np.matrix(np.eye(n), dtype='complex128')
        self.dits = dits
        self._n = n

    def matrix(self, v):
        return self._I

    def assemble(self, v, i=0):
        return []

    def __repr__(self):
        return "IdentityStep({})".format(self._n)

class ZXZXZQubitStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 3
        self.dits = 1

        self._x90 = unitaries.rot_x(np.pi/2)
        self._rot_z = unitaries.rot_z(0)
        self._out = np.matrix(np.eye(2), dtype='complex128')
        self._buffer = np.matrix(np.eye(2), dtype = 'complex128')
        # need two buffers due to a bug in some implementations of numpy
       
    def matrix(self, v):
        utils.re_rot_z(v[0]*np.pi*2, self._rot_z)
        self._out = np.dot(self._x90, self._rot_z, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._out, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[2]*np.pi*2-np.pi, self._rot_z)
        return np.dot(self._rot_z, self._out)

    def jac(self, v):
        utils.re_rot_z_jac(v[0]*np.pi*2, self._rot_z, multiplier=np.pi*2)
        self._out = np.dot(self._x90, self._rot_z, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._out, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[2]*np.pi*2-np.pi, self._rot_z)
        J1 = np.dot(self._rot_z, self._out)

        utils.re_rot_z(v[0]*np.pi*2, self._rot_z)
        self._out = np.dot(self._x90, self._rot_z, out=self._out)
        utils.re_rot_z_jac(v[1]*np.pi*2 + np.pi, self._rot_z, multiplier=np.pi*2)
        self._buffer = np.dot(self._rot_z, self._out, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[2]*np.pi*2-np.pi, self._rot_z)
        J2 = np.dot(self._rot_z, self._out)

        utils.re_rot_z(v[0]*np.pi*2, self._rot_z)
        self._out = np.dot(self._x90, self._rot_z, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._out, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z_jac(v[2]*np.pi*2-np.pi, self._rot_z, multiplier=np.pi*2)
        J3 = np.dot(self._rot_z, self._out)
        return [J1, J2, J3]

    def assemble(self, v, i=0):
        # later use IBM's parameterization and convert to ZXZXZ in post processing
        out = []
        out.append(("gate", "Z", (v[0]*np.pi*2,), (i,)))
        out.append(("gate", "X", (np.pi/2,), (i,)))
        out.append(("gate", "Z", (v[1]*np.pi*2 + np.pi,), (i,)))
        out.append(("gate", "X", (np.pi/2,), (i,)))
        out.append(("gate", "Z", (v[2]*np.pi*2 + np.pi,), (i,)))
        return [("block", out)]

    def _draw_assemble(self, i=0):
        return [("U", "q{}".format(i))] 
    
    def __repr__(self):
        return "ZXZXZQubitStep()"

class XZXZPartialQubitStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 2
        self.dits = 1

        self._x90 = unitaries.rot_x(np.pi/2)
        self._rot_z = unitaries.rot_z(0)
        self._out = np.matrix(np.eye(2), dtype='complex128')
        self._buffer = np.matrix(np.eye(2), dtype = 'complex128')
        # need two buffers due to a bug in some implementations of numpy
        
    def matrix(self, v):
        utils.re_rot_z(v[0]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._x90, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2-np.pi, self._rot_z)
        return np.dot(self._rot_z, self._out)

    def jac(self, v):
        utils.re_rot_z_jac(v[0]*np.pi*2 + np.pi, self._rot_z, multiplier=np.pi*2)
        self._buffer = np.dot(self._rot_z, self._x90, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2-np.pi, self._rot_z)
        J1 = np.dot(self._rot_z, self._out)

        utils.re_rot_z(v[0]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._x90, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z_jac(v[1]*np.pi*2-np.pi, self._rot_z, multiplier=2*np.pi)
        J2 = np.dot(self._rot_z, self._out)
        return [J1, J2]

    def assemble(self, v, i=0):
        # later use IBM's parameterization and convert to ZXZXZ in post processing
        out = []
        out.append(("gate", "X", (np.pi/2,), (i,)))
        out.append(("gate", "Z", (v[0]*np.pi*2 + np.pi,), (i,)))
        out.append(("gate", "X", (np.pi/2,), (i,)))
        out.append(("gate", "Z", (v[1]*np.pi*2 + np.pi,), (i,)))
        return [("block", out)]

    def _draw_assemble(self, i=0):
        return [("U", "q{}".format(i))] 
    
    def __repr__(self):
        return "XZXZPartialQubitStep()"

class QiskitU3QubitStep(QuantumStep):
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
        return np.matrix([[ct, -st * (cl + 1j * sl)], [st * (cp + 1j * sp), ct * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128')

    def jac(self, v):
        ct = np.cos(v[0] * np.pi)
        st = np.sin(v[0] * np.pi)
        cp = np.cos(v[1] * np.pi * 2)
        sp = np.sin(v[1] * np.pi * 2)
        cl = np.cos(v[2] * np.pi * 2)
        sl = np.sin(v[2] * np.pi * 2)
        return [np.matrix([[-np.pi*st, -np.pi*ct * (cl + 1j * sl)], [np.pi*ct * (cp + 1j * sp), -np.pi*st * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128'),
                np.matrix([[0, 0], [st * 2*np.pi*(-sp + 1j * cp), ct * 2*np.pi*(cl * -sp - sl * cp + 1j * cl * cp + 1j * sl * -sp)]], dtype='complex128'),
                np.matrix([[0, -st * 2*np.pi*(-sl + 1j * cl)], [0, ct * 2*np.pi*(-sl * cp - cl * sp + 1j * -sl * sp + 1j * cl * cp)]], dtype='complex128')
                ]

    def assemble(self, v, i=0):
        return [("gate", "U3", (v[0]*np.pi*2, v[1]*np.pi*2, v[2]*np.pi*2), (i,))]

    def _draw_assemble(self, i=0):
        return [("U", "q{}".format(i))]

    def __repr__(self):
        return "QiskitU3QubitStep()"

class SingleQutritStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 8
        self.dits = 1

    def matrix(self, v):
        return utils.qt_arb_rot(*v)

    def assemble(self, v, i=0):
        return [("qutrit", tuple(v), (i,))]
    
    def __repr__(self):
        return "SingleQutritStep()"

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
        self.num_inputs = 0
        self.dits = 2

    def matrix(self, v):
        return CSUMStep._csum

    def assemble(self, v, i=0):
        return [("gate", "CSUM", (), (i, i+1))]

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
        self.num_inputs = 0
        self.dits = 2

    def matrix(self, v):
        return CPIStep._cpi

    def assemble(self, v, i=0):
        return [("gate", "CPI", (), (i, i+1))]

    def __repr__(self):
        return "CPIStep()"

class CPIPhaseStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 0
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
        self.dits = 2

    def matrix(self, v):
        return self._cpi

    def assemble(self, v, i=0):
        return [("gate", "CPI-", (), (i, i+1))]

    def __repr__(self):
        return "CPIPhaseStep()"

class CNOTStep(QuantumStep):
    _cnot = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype='complex128')
    def __init__(self):
        self.num_inputs = 0
        self.dits = 2

    def matrix(self, v):
        return CNOTStep._cnot

    def assemble(self, v, i=0):
        return [("gate", "CNOT", (), (i, i+1))]

    def _draw_assemble(self, i=0):
        return [("CNOT", "q{}".format(i+1), "q{}".format(i))]

    def __repr__(self):
        return "CNOTStep()"

class NonadjacentCNOTStep(QuantumStep):
    def __init__(self, dits, control, target):
        self.dits = dits
        self.num_inputs = 0
        self.control = control
        self.target = target
        self._U = unitaries.arbitrary_cnot(dits, control, target)

    def matrix(self, v):
        return self._U

    def assemble(self, v, i=0):
        return [("gate", "CNOT", (), (self.control, self.target))]

    def _draw_assemble(self, i=0):
        return [("CNOT", "q{}".format(self.target), "q{}".format(self.control))]

    def __repr__(self):
        return "NonadjacentCNOTStep({}, {}, {})".format(self.dits, self.control, self.target)

class UStep(QuantumStep):
    def __init__(self, U, d=2):
        self.d = d
        self.U = U
        self.dits = int(np.log(U.shape[0])/np.log(2))
        self.num_inputs = 0

    def matrix(self, v):
        return self.U

    def assemble(self, v, i=0):
        return [("gate", "CUSTOM", (), (i,))]

    def _draw_assemble(self, i=0):
        return [("?", "q{}".format(i))]

    def __repr__(self):
        if self.d == 2:
            return "UStep({})".format(repr(U))
        else:
            return "UStep({}, d={})".format(repr(U), self.d)

class CUStep(QuantumStep):
    def __init__(self, U, name=None, flipped=False):
        self.name = name
        self.flipped = flipped
        self.num_inputs = 0
        self._U = U
        n = np.shape(U)[0]
        I = np.matrix(np.eye(n))
        top = np.pad(self._U if flipped else I, [(0,n),(0,n)], 'constant')
        bot = np.pad(I if flipped else self._U, [(n,0),(n,0)], 'constant')
        self._CU = np.matrix(top + bot)
        self.dits = 2
        self.num_inputs = 0

    def matrix(self, v):
        return self._CU

    def _draw_assemble(self, i=0):
        raise NotImplementedError("Need to finish this")

    def assemble(self, v, i=0):
        return [("gate", "CUSTOM", (), (i,i+1))]

    def __repr__(self):
        return "CUStep(" + str(repr(self._U)) + ("" if self.name is None else ", name={}".format(repr(self.name))) + ("flipped=True" if self.flipped else "") + ")"

class CRZStep(QuantumStep):
    _cnr = unitaries.sqrt_cnot
    _I = np.matrix(np.eye(2), dtype='complex128')
    def __init__(self):
        self.num_inputs = 1
        self.dits = 2

    def matrix(self, v):
        U = np.dot(CRZStep._cnr, np.kron(CRZStep._I, unitaries.rot_z(v[0]*np.pi*2)))
        return np.dot(U, CRZStep._cnr)

    def assemble(self, v, i=0):
        return [("gate", "sqrt(CNOT)", (), (i, i+1)), ("gate", "Z", (v[0],), (i+1,)), ("gate", "sqrt(CNOT", (), (i, i+1))]

    def _draw_assemble(Self, i=0):
        return [("CRZ", "q{}".format(i+1), "q{}".format(i))]

    def __repr__(self):
        return "CQubitStep()"

class CNOTRootStep(QuantumStep):
    _cnr = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0.5+0.5j,0.5-0.5j],
                       [0,0,0.5-0.5j,0.5+0.5j]])
    def __init__(self):
        self.num_inputs = 0
        self.dits = 2

    def matrix(self, v):
        return CNOTRootStep._cnr

    def assemble(self, v, i=0):
        return [("gate", "sqrt(CNOT)", (), (i, i+1))]

    def __repr__(self):
        return "CNOTRootStep()"

class KroneckerStep(QuantumStep):
    def __init__(self, *substeps):
        self.num_inputs = sum([step.num_inputs for step in substeps])
        self._substeps = substeps
        self.dits = sum([step.dits for step in substeps])

    def matrix(self, v):
        matrices = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step.num_inputs])
            matrices.append(U)
            index += step.num_inputs
        U = matrices[0]
        for matrix in matrices[1:]:
            U = np.kron(U, matrix)
        return U

    def jac(self, v):
        matrices = []
        jacs = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step.num_inputs])
            matrices.append(U)
            index += step.num_inputs

        index = 0
        for i in range(len(self._substeps)):
            jacstep = self._substeps[i]
            if jacstep.num_inputs == 0:
                continue
            Js = jacstep.jac(v[index:index+jacstep.num_inputs])
            index += jacstep.num_inputs
            for J in Js:
                M = J if i == 0 else matrices[0]
                for k in range(1,len(matrices)):
                    M1 = J if i == k else matrices[k]
                    M = np.kron(M, M1)
                jacs.append(M)
        return jacs

    def assemble(self, v, i=0):
        out = []
        index = 0
        for step in self._substeps:
            out += step.assemble(v[index:index+step.num_inputs], i)
            index += step.num_inputs
            i += step.dits
        return [("block", out)]

    def appending(self, step):
        return KroneckerStep(*self._substeps, step)

    def _draw_assemble(self, i=0):
        endlist = []
        for step in self._substeps:
            endlist += step._draw_assemble(i)
            i += step.dits
        return endlist

    def _parts(self):
        return self._substeps

    def __deepcopy__(self, memo):
        return KroneckerStep(self._substeps.__deepcopy__(memo))

    def __repr__(self):
        return "KroneckerStep({})".format(repr(self._substeps)[1:-1])

class ProductStep(QuantumStep):
    def __init__(self, *substeps):
        self.num_inputs = sum([step.num_inputs for step in substeps])
        self._substeps = substeps
        self.dits = 0 if len(substeps) == 0 else substeps[0].dits

    def matrix(self, v):
        matrices = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step.num_inputs])
            matrices.append(U)
            index += step.num_inputs
        U = matrices[0]
        buffer1 = U.copy()
        buffer2 = U.copy()
        for matrix in matrices[1:]:
            U = np.matmul(matrix, U, out=buffer1)
            buffertmp = buffer2
            buffer2 = buffer1
            buffer1 = buffer2
        return U

    def jac(self, v):
        matrices = []
        jacs = []
        index = 0
        for step in self._substeps:
            U = step.matrix(v[index:index+step.num_inputs])
            matrices.append(U)
            index += step.num_inputs
        
        index = 0
        for i in range(len(self._substeps)):
            jacstep = self._substeps[i]
            if jacstep.num_inputs == 0:
                continue
            Js = jacstep.jac(v[index:index+jacstep.num_inputs])
            index += jacstep.num_inputs
            for J in Js:
                M = J if i == 0 else matrices[0]
                buffer1 = U.copy()
                buffer2 = U.copy()
                for k in range(1,len(matrices)):
                    M1 = J if i == k else matrices[k]
                    M = np.matmul(M1, M, out=buffer1)
                    buffertmp = buffer2
                    buffer2 = buffer1
                    buffer1 = buffer2
                jacs.append(M)

        return jacs

    def assemble(self, v, i=0):
        out = []
        index = 0
        for step in self._substeps:
            out += step.assemble(v[index:index+step.num_inputs], i)
            index += step.num_inputs
        return out

    def _optimize(self, I):
        return self # disabling optimization because its broken rn
        # for more detail: This code will make your circuits run faster, but will sometimes remove single qubit gates unintentionally, which is bad.  I am working on an improved version of this function, but it is a problem which will take some time to find a good algorithm for.
        steps = self._substeps
        for size in range(2, self.dits):
            latest = [None for _ in range(0, self.dits)]
            connected = [set() for _ in range(0, self.dits)]
            newsteps = []
            for step in steps:
                index = 0
                for part in step._parts():
                    nextindex = index + part.dits
                    if type(part) is IdentityStep:
                        index = nextindex
                        continue
                    part_connections = set(range(index, index + part.dits))
                    for qudit in part_connections:
                        part_connections = part_connections.union(connected[qudit])
                    if len(part_connections) >= size:
                        qudit = 0
                        newkron = []
                        while qudit < self.dits:
                            if qudit in part_connections:
                                newstep = latest[qudit]
                                if newstep == None:
                                    newstep = I
                                newkron.append(newstep)
                                for i in range(qudit, qudit+newstep.dits):
                                    latest[i] = None
                                    connected[i] = set()
                                qudit += newstep.dits
                            else:
                                newkron.append(I)
                                qudit += 1
                        newsteps.append(KroneckerStep(*newkron))
                        part_connections = set(range(index, index + part.dits))
                        for i in part_connections:
                            latest[i] = None
                            connected[i] = set()
                        for i in range(index, index + part.dits):
                            latest[i] = part
                            connected[i] = set(range(index, index + part.dits))
                    else:
                        qudit = min(part_connections)
                        if len(part_connections) > part.dits:
                            part = KroneckerStep(*[I]*(index - min(part_connections)), part, *[I]*(max(part_connections) - index + part.dits))
                        if type(part) is ProductStep:
                            part_pieces = part._substeps
                        else:
                            part_pieces = [part]
                        newq = latest[qudit]
                        if newq is None:
                            newq = I
                        qudit += newq.dits
                        while newq.dits < part.dits:
                            qtwo = latest[qudit]
                            if qtwo is None:
                                qtwo = I
                            qudit += qtwo.dits
                            if type(qtwo) is KroneckerStep:
                                qtwo = qtwo._substeps
                            else:
                                qtwo = [qtwo]
                            if type(newq) is KroneckerStep:
                                newq = newq.appending(*qtwo)
                            else:
                                newq = KroneckerStep(newq, *qtwo)
                        if type(newq) is IdentityStep:
                            newq = part
                        elif type(newq) is ProductStep:
                            newq = newq.appending(*part_pieces)
                        else:
                            newq = ProductStep(newq, *part_pieces)
                        for i in part_connections:
                            latest[i] = newq
                            connected[i] = part_connections
                    index = nextindex
            qudit = 0
            newkron = []
            while qudit < self.dits:
                if qudit in part_connections:
                    newstep = latest[qudit]
                    if newstep == None:
                        newstep = I
                    newkron.append(newstep)
                    for i in range(qudit, qudit+newstep.dits):
                        latest[i] = None
                        connected[i] = set()
                    qudit += newstep.dits
                else:
                    newkron.append(I)
                    qudit += 1
            newsteps.append(KroneckerStep(*newkron))
            steps = newsteps
        return ProductStep(*steps)

    def _draw_assemble(self, i=0):
        endlist = []
        for step in self._substeps:
            endlist += step._draw_assemble(i)
        return endlist

    def appending(self, *steps):
        return ProductStep(*self._substeps, *steps)

    def __deepcopy__(self, memo):
        return ProductStep(self._substeps.__deepcopy__(memo))

    def __repr__(self):
        return "ProductStep({})".format(repr(self._substeps)[1:-1])

