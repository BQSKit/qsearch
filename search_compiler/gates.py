import numpy as np

cnot = np.matrix([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]],
                  dtype='complex128')

swap = np.matrix([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]],
                  dtype='complex128')

toffoli = np.matrix([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0]], 
                     dtype='complex128')


fredkin = np.matrix([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,1,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1]],
                     dtype='complex128')

peres = np.matrix([[1,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0],
                   [0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,1,0,0],
                   [0,0,0,0,1,0,0,0]], 
                   dtype='complex128')

maybe_or = np.matrix([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,1,0,0,0]], 
                      dtype='complex128')

def rot_z(theta):
    return np.matrix([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype='complex128')

def rot_x(theta):
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_y(theta):
    return np.matrix([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def qft(n):
    root = np.e ** (2j * np.pi / n)
    Q = np.matrix(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)
    return Q

def identity(n): # not super necessary but saves a little code length
    return np.matrix(np.eye(n), dtype='complex128')


# generates an arbitrary cnot gate by classical logic and brute force
# it may be a good idea to write a better version of this at some point, but this should be good enough for use with the search compiler on 2-4 qubits.
def arbitrary_cnot(n, control, target):
    # this test returns if the given matrix index correspond to a "true" value of the bit at selected qubit index
    test = lambda x, value: (x // 2**(n-value-1)) % 2 != 0
    def f(i,j):
        # unless the control is true in both columns, just return part of the identity
        if not (test(i, control) and test(j, control)):
            return i==j
        elif test(i, target) != test(j, target):
            # if the control is true in both columns and the target is mismatched, verify everything else is matched
            for k in range(0, n):
                if k == target or k == control:
                    continue
                elif test(i, k) != test(j, k):
                    return 0
            return 1
        else:
            # if the control is true and matched and the target is matched, return 0
            return 0
    return np.matrix(np.fromfunction(np.vectorize(f), (2**n,2**n)),dtype='int')

