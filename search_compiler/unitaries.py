import numpy as np

cnot = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]],
                  dtype='complex128')

sqrt_cnot = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0.5+0.5j,0.5-0.5j],
                       [0,0,0.5-0.5j,0.5+0.5j]],
                       dtype='complex128')

swap = np.array([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]],
                  dtype='complex128')

toffoli = np.array([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0]], 
                     dtype='complex128')


fredkin = np.array([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,1,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1]],
                     dtype='complex128')

peres = np.array([[1,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0],
                   [0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,1,0,0],
                   [0,0,0,0,1,0,0,0]], 
                   dtype='complex128')

logical_or = np.array([[1,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1,0]], 
                        dtype='complex128')

full_adder = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]],
                        dtype='complex128')
# Full adder that converts ABC0 to ABCS.  Behavior is not well defined when the last input bit is a 1.

def rot_z(theta):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype='complex128')

def rot_x(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_y(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def qft(n):
    root = np.e ** (2j * np.pi / n)
    Q = np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)
    return Q

def identity(n): # not super necessary but saves a little code length
    return np.array(np.eye(n), dtype='complex128')


# generates an arbitrary cnot gate by classical logic and brute force
# it may be a good idea to write a better version of this at some point, but this should be good enough for use with the search compiler on 2-4 qubits.
def arbitrary_cnot(dits, control, target):
    # this test returns if the given matrix index correspond to a "true" value of the bit at selected qubit index
    test = lambda x, value: (x // 2**(dits-value-1)) % 2 != 0
    def f(i,j):
        # unless the control is true in both columns, just return part of the identity
        if not (test(i, control) and test(j, control)):
            return i==j
        elif test(i, target) != test(j, target):
            # if the control is true in both columns and the target is mismatched, verify everything else is matched
            for k in range(0, dits):
                if k == target or k == control:
                    continue
                elif test(i, k) != test(j, k):
                    return 0
            return 1
        else:
            # if the control is true and matched and the target is matched, return 0
            return 0
    return np.array(np.fromfunction(np.vectorize(f), (2**dits,2**dits)),dtype='complex128')

