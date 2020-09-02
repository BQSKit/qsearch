import numpy as np
from . import utils, graphics, unitaries
from hashlib import md5

try:
    from qsrs import native_from_object
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

    def mat_jac(self, v):
        if self.num_inputs == 0:
            return (self.matrix(v), []) # a circuit component with no inputs has no jacobian to return
        raise NotImplementedError("Subclasses of QuantumStep are required to implement the mat_jac(v) method in order to be used with gradient optimizers.")

    def __eq__(self, other):
        return repr(self) == repr(other)

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
        self._I = np.array(np.eye(n), dtype='complex128')
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
        self._out = np.array(np.eye(2), dtype='complex128')
        self._buffer = np.array(np.eye(2), dtype = 'complex128')
        # need two buffers due to a bug in some implementations of numpy
       
    def matrix(self, v):
        utils.re_rot_z(v[0]*np.pi*2, self._rot_z)
        self._out = np.dot(self._x90, self._rot_z, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._out, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[2]*np.pi*2-np.pi, self._rot_z)
        return np.dot(self._rot_z, self._out)

    def mat_jac(self, v):
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
        
        utils.re_rot_z(v[2]*np.pi*2-np.pi, self._rot_z)
        U = np.dot(self._rot_z, self._out)
        return (U, [J1, J2, J3])

    def assemble(self, v, i=0):
        # later use IBM's parameterization and convert to ZXZXZ in post processing
        out = []
        v = np.array(v)%1 # confine the range of what we print to come up with nicer numbers at no loss of generality
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
        self._out = np.array(np.eye(2), dtype='complex128')
        self._buffer = np.array(np.eye(2), dtype = 'complex128')
        # need two buffers due to a bug in some implementations of numpy
        
    def matrix(self, v):
        utils.re_rot_z(v[0]*np.pi*2 + np.pi, self._rot_z)
        self._buffer = np.dot(self._rot_z, self._x90, out=self._buffer)
        self._out = np.dot(self._x90, self._buffer, out=self._out)
        utils.re_rot_z(v[1]*np.pi*2-np.pi, self._rot_z)
        return np.dot(self._rot_z, self._out)

    def mat_jac(self, v):
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

        utils.re_rot_z(v[1]*np.pi*2-np.pi, self._rot_z)
        U = np.dot(self._rot_z, self._out)
        return (U, [J1, J2])

    def assemble(self, v, i=0):
        # later use IBM's parameterization and convert to ZXZXZ in post processing
        out = []
        v = np.array(v)%1 # confine the range of what we print to come up with nicer numbers at no loss of generality
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
        return np.array([[ct, -st * (cl + 1j * sl)], [st * (cp + 1j * sp), ct * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128')

    def __eq__(self, other):
        return type(self) == type(other)

    def mat_jac(self, v):
        ct = np.cos(v[0] * np.pi)
        st = np.sin(v[0] * np.pi)
        cp = np.cos(v[1] * np.pi * 2)
        sp = np.sin(v[1] * np.pi * 2)
        cl = np.cos(v[2] * np.pi * 2)
        sl = np.sin(v[2] * np.pi * 2)

        U = np.array([[ct, -st * (cl + 1j * sl)], [st * (cp + 1j * sp), ct * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128')
        J1 = np.array([[-np.pi*st, -np.pi*ct * (cl + 1j * sl)], [np.pi*ct * (cp + 1j * sp), -np.pi*st * (cl * cp - sl * sp + 1j * cl * sp + 1j * sl * cp)]], dtype='complex128')
        J2 = np.array([[0, 0], [st * 2*np.pi*(-sp + 1j * cp), ct * 2*np.pi*(cl * -sp - sl * cp + 1j * cl * cp + 1j * sl * -sp)]], dtype='complex128')
        J3 = np.array([[0, -st * 2*np.pi*(-sl + 1j * cl)], [0, ct * 2*np.pi*(-sl * cp - cl * sp + 1j * -sl * sp + 1j * cl * cp)]], dtype='complex128')
        return (U, [J1, J2, J3])

    def assemble(self, v, i=0):
        v = np.array(v)%1 # confine the range of what we print to come up with nicer numbers at no loss of generality
        return [("gate", "U3", (v[0]*np.pi*2, v[1]*np.pi*2, v[2]*np.pi*2), (i,))]

    def _draw_assemble(self, i=0):
        return [("U", "q{}".format(i))]

    def __repr__(self):
        return "QiskitU3QubitStep()"

class ParameterFixedStep(QuantumStep):
    # this class was designed for debuggin purposes, but it may be a bit buggy itself
    def __init__(self, substep, fixedindices):
        self.substep = substep
        self.fixedindices = fixedindices
        self.num_inputs = substep.num_inputs - len(fixedindices)
        self.dits = self.substep.dits
        self.v = [0 for _ in range(substep.num_inputs)]

    def matrix(self, v):
        j = 0
        for i in range(self.substep.num_inputs):
            if i not in self.fixedindices:
                self.v[i] = v[j]
                j += 1
        return self.substep.matrix(self.v)

    def mat_jac(self, v):
        j = 0
        for i in range(self.substep.num_inputs):
            if i not in self.fixedindices:
                self.v[i] = v[j]
                j += 1
        U, Js = self.substep.mat_jac(self.v)
        Js = [Js[i] for i in range(self.substep.num_inputs) if i not in self.fixedindices]
        return (U, Js)

class SingleQutritStep(QuantumStep):
    def __init__(self):
        self.num_inputs = 8
        self.dits = 1

    def matrix(self, v):
        # for reference see the original implementation, qt_arb_rot in utils.py, which is now deprecated
        # this was re-written to be computationally more efficient, and in my opinion, more readable
        s1 = np.sin(v[0] * np.pi * 2)
        c1 = np.cos(v[0] * np.pi * 2)
        s2 = np.sin(v[1] * np.pi * 2)
        c2 = np.cos(v[1] * np.pi * 2)
        s3 = np.sin(v[2] * np.pi * 2)
        c3 = np.cos(v[2] * np.pi * 2)
        
        p1 = np.exp(1j * v[3] * np.pi * 2)
        m1 = np.exp(-1j * v[3] * np.pi * 2)
        p2 = np.exp(1j * v[4] * np.pi * 2)
        m2 = np.exp(-1j * v[4] * np.pi * 2)
        p3 = np.exp(1j * v[5] * np.pi * 2)
        m3 = np.exp(-1j * v[5] * np.pi * 2)
        p4 = np.exp(1j * v[6] * np.pi * 2)
        m4 = np.exp(-1j * v[6] * np.pi * 2)
        p5 = np.exp(1j * v[7] * np.pi * 2)
        m5 = np.exp(-1j * v[7] * np.pi * 2)

        return np.array([
            [c1*c2*p1, s1*p3, c1*s2*p4],
            [s2*s3*m4*m5 - s1*c2*c3*p1*p2*m3, c1*c3*p2, -c2*s3*m1*m5 - s1*s2*c3*p2*m3*p4],
            [-s1*c2*s3*p1*m3*p5 - s2*c3*m2*m4, c1*s3*p5, c2*c3*m1*m2 - s1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128')

    def mat_jac(self, v):
        s1 = np.sin(v[0] * np.pi * 2)
        c1 = np.cos(v[0] * np.pi * 2)
        s2 = np.sin(v[1] * np.pi * 2)
        c2 = np.cos(v[1] * np.pi * 2)
        s3 = np.sin(v[2] * np.pi * 2)
        c3 = np.cos(v[2] * np.pi * 2)
        
        p1 = np.exp(1j * v[3] * np.pi * 2)
        m1 = np.exp(-1j * v[3] * np.pi * 2)
        p2 = np.exp(1j * v[4] * np.pi * 2)
        m2 = np.exp(-1j * v[4] * np.pi * 2)
        p3 = np.exp(1j * v[5] * np.pi * 2)
        m3 = np.exp(-1j * v[5] * np.pi * 2)
        p4 = np.exp(1j * v[6] * np.pi * 2)
        m4 = np.exp(-1j * v[6] * np.pi * 2)
        p5 = np.exp(1j * v[7] * np.pi * 2)
        m5 = np.exp(-1j * v[7] * np.pi * 2)

        U = np.array([
            [c1*c2*p1, s1*p3, c1*s2*p4],
            [s2*s3*m4*m5 - s1*c2*c3*p1*p2*m3, c1*c3*p2, -c2*s3*m1*m5 - s1*s2*c3*p2*m3*p4],
            [-s1*c2*s3*p1*m3*p5 - s2*c3*m2*m4, c1*s3*p5, c2*c3*m1*m2 - s1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128')

        Jt1 = np.array([
            [-s1*c2*p1, c1*p3, -s1*s2*p4],
            [-c1*c2*c3*p1*p2*m3, -s1*c3*p2, -c1*s2*c3*p2*m3*p4],
            [-c1*c2*s3*p1*m3*p5, -s1*s3*p5, -c1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128') * 2 *np.pi

        Jt2 = np.array([
            [-c1*s2*p1, 0, c1*c2*p4],
            [c2*s3*m4*m5 + s1*s2*c3*p1*p2*m3, 0, s2*s3*m1*m5 - s1*c2*c3*p2*m3*p4],
            [s1*s2*s3*p1*m3*p5 -c2*c3*m2*m4, 0, -s2*c3*m1*m2 - s1*c2*s3*m3*p4*p5]
            ], dtype = 'complex128') * 2 * np.pi

        Jt3 = np.array([
            [0, 0, 0],
            [s2*c3*m4*m5 + s1*c2*s3*p1*p2*m3, -c1*s3*p2, -c2*c3*m1*m5 + s1*s2*s3*p2*m3*p4],
            [-s1*c2*c3*p1*m3*p5 + s2*s3*m2*m4, c1*c3*p5, -c2*s3*m1*m2 - s1*s2*c3*m3*p4*p5]
            ], dtype = 'complex128') * 2 * np.pi

        Je1 = np.array([
            [1j*c1*c2*p1, 0, 0],
            [-1j*s1*c2*c3*p1*p2*m3, 0, 1j*c2*s3*m1*m5],
            [-1j*s1*c2*s3*p1*m3*p5, 0, -1j*c2*c3*m1*m2]
            ], dtype = 'complex128') * 2 * np.pi

        Je2 = np.array([
            [0, 0, 0],
            [-1j*s1*c2*c3*p1*p2*m3, 1j*c1*c3*p2, -1j*s1*s2*c3*p2*m3*p4],
            [1j*s2*c3*m2*m4, 0, -1j*c2*c3*m1*m2]
            ], dtype = 'complex128') * 2 * np.pi

        Je3 = np.array([
            [0, 1j*s1*p3, 0],
            [1j*s1*c2*c3*p1*p2*m3, 0, 1j*s1*s2*c3*p2*m3*p4],
            [1j*s1*c2*s3*p1*m3*p5, 0, 1j*s1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128') * 2 * np.pi

        Je4 = np.array([
            [0, 0, 1j*c1*s2*p4],
            [-1j*s2*s3*m4*m5, 0, -1j*s1*s2*c3*p2*m3*p4],
            [1j*s2*c3*m2*m4, 0, -1j*s1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128') * 2 *np.pi

        Je5 = np.array([
            [0, 0, 0],
            [-1j*s2*s3*m4*m5, 0, 1j*c2*s3*m1*m5],
            [-1j*s1*c2*s3*p1*m3*p5, 1j*c1*s3*p5, -1j*s1*s2*s3*m3*p4*p5]
            ], dtype = 'complex128') * 2 * np.pi

        return (U, [Jt1, Jt2, Jt3, Je1, Je2, Je3, Je4, Je5])

    def assemble(self, v, i=0):
        return [("gate", "QUTRIT", (vv*2*np.pi for vv in v), (i,))]
    
    def __repr__(self):
        return "SingleQutritStep()"

class CSUMStep(QuantumStep):
    _csum =  np.array([[1,0,0, 0,0,0, 0,0,0],
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
    _cpi = np.array([[1,0,0, 0,0,0, 0,0,0],
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
        self._cpi = np.array([[1,0,0, 0,0,0, 0,0,0],
                               [0,1,0, 0,0,0, 0,0,0],
                               [0,0,1, 0,0,0, 0,0,0],
                               [0,0,0, 0,-1,0,0,0,0],
                               [0,0,0, 1,0,0, 0,0,0],
                               [0,0,0, 0,0,1, 0,0,0],
                               [0,0,0, 0,0,0, 1,0,0],
                               [0,0,0, 0,0,0, 0,1,0],
                               [0,0,0, 0,0,0, 0,0,1]
                              ], dtype='complex128')
        diag_mod = np.array(np.diag([1]*4 + [np.exp(2j * np.random.random()*np.pi) for _ in range(0,5)]))
        self._cpi = np.matmul(self._cpi, diag_mod)
        self.dits = 2

    def matrix(self, v):
        return self._cpi

    def assemble(self, v, i=0):
        return [("gate", "CPI-", (), (i, i+1))]

    def __repr__(self):
        return "CPIPhaseStep()"

class CNOTStep(QuantumStep):
    _cnot = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype='complex128')
    def __init__(self):
        self.num_inputs = 0
        self.dits = 2

    def __eq__(self, other):
        return type(self) == type(other)

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

class UpgradedConstantStep(QuantumStep):
    def __init__(self, other, df=3):
        if other.num_inputs > 0:
            raise AttributeError("UpgradedConstantStep is designed for only constant gates")
        OU = other.matrix([])
        di = int(OU.shape[0]**(1/other.dits))
        if df <= di:
            raise AttributeError("Gate cannot be upgraded because it is already of an equal or higher dit level")
        self.df = df
        self.dits = other.dits
        self.U = utils.upgrade_dits(OU, di, df)
        self.num_inputs = 0
        self.substep = other

    def matrix(self, v):
        return self.U

    def assemble(self, v, i=0):
        return self.substep.assemble(v, i)

    def _draw_assemble(self, v, i):
        return self.substep._draw_assemble(v, i)

    def __repr__(self):
        return "UpgradedConstantStep({}, df={})".format(repr(self.substep), self.df)


class CUStep(QuantumStep):
    def __init__(self, U, name=None, flipped=False):
        self.name = name
        self.flipped = flipped
        self.num_inputs = 0
        self._U = U
        n = np.shape(U)[0]
        I = np.array(np.eye(n))
        top = np.pad(self._U if flipped else I, [(0,n),(0,n)], 'constant')
        bot = np.pad(I if flipped else self._U, [(n,0),(n,0)], 'constant')
        self._CU = np.array(top + bot)
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
    _I = np.array(np.eye(2), dtype='complex128')
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
    _cnr = np.array([[1,0,0,0],
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
        if len(self._substeps) < 2:
            return self._substeps[0].matrix(v)
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

    def mat_jac(self, v):
        if len(self._substeps) < 2:
            return self._substeps[0].mat_jac(v)
        matjacs = []
        index = 0
        for step in self._substeps:
            MJ = step.mat_jac(v[index:index+step.num_inputs])
            matjacs.append(MJ)
            index += step.num_inputs

        U = None # deal with this how you like @ethan
        jacs = []
        for M, Js in matjacs:
            jacs = [np.kron(J, M) for J in jacs]
            for J in Js:
                jacs.append(J if U is None else np.kron(U,J))
            U = M if U is None else np.kron(U, M)

        return (U, jacs)

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
        if len(self._substeps) < 2:
            return self._substeps[0].matrix(v)
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

    def mat_jac(self, v):
        if len(self._substeps) < 2:
            return self._substeps[0].mat_jac(v)
        submats = []
        subjacs = []
        index = 0
        for step in self._substeps:
            U, Js = step.mat_jac(v[index:index+step.num_inputs])
            submats.append(U)
            subjacs.append(Js)
            index += step.num_inputs
        
        B = np.eye(submats[0].shape[0], dtype='complex128')
        A = submats[0]
        jacs = []
        ba1 = A.copy()
        ba2 = A
        bb1 = B.copy()
        bb2 = B
        bj = B.copy()
        for matrix in submats[1:]:
            A = np.matmul(matrix, A, out=ba1)
            buffertmp = ba2
            ba2 = ba1
            ba1 = ba2

        for i, Js in enumerate(subjacs):
            A = np.matmul(A, submats[i].T.conjugate(), out=ba1) # remove the current matrix from the "after" array
            for J in Js:
                tmp = np.matmul(J, B, out=bj)
                jacs.append(np.matmul(A, tmp, out=J))
            B = np.matmul(submats[i], B, out=bb1) # add the current matrix to the "before" array before progressing
            buffertmp = ba1
            ba1 = ba2
            ba2 = buffertmp
            buffertmp = bb1
            bb1 = bb2
            bb2 = buffertmp
            
        return (B, jacs)

    def assemble(self, v, i=0):
        out = []
        index = 0
        for step in self._substeps:
            out += step.assemble(v[index:index+step.num_inputs], i)
            index += step.num_inputs
        return out

    def _draw_assemble(self, i=0):
        endlist = []
        for step in self._substeps:
            endlist += step._draw_assemble(i)
        return endlist

    def appending(self, *steps):
        return ProductStep(*self._substeps, *steps)

    def inserting(self, *steps, depth=-1):
        return ProductStep(*self._substeps[:depth], *steps, *self._substeps[depth:])

    def __deepcopy__(self, memo):
        return ProductStep(self._substeps.__deepcopy__(memo))

    def __repr__(self):
        return "ProductStep({})".format(repr(self._substeps)[1:-1])

