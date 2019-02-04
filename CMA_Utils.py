import numpy as np

cnot = np.matrix([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]
                 ], dtype='complex128')

def matrix_product(*LU):
    # performs matrix multiplication of a list of matrices
    result = np.eye(LU[0].shape[0])
    for U in LU:
        result = np.dot(result, U)
    return result

def matrix_kron(*LU):
    # performs the kronecker product on a list of matrices
    result = LU[0]
    for U in LU[1:]:
        result = np.kron(result,U)
    return result

def op_norm(A):
    # an implementation of the l1-l1 operator norm
    return max([np.linalg.norm(x,ord=1) for x in A])

def matrix_distance_squared(A,B):
    # this distance function is designed to be phase agnostic
    # optimized implementation
    return 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    #original implementation
    #return 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]

def matrix_distance(A,B):
    return np.sqrt(matrix_distance_squared(A,B))

def rot_z(theta):
    return np.matrix([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype='complex128')

def rot_x(theta):
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_y(theta):
    return np.matrix([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')


# copied from qnl_analysis.SimTools
def Q1_unitary(x):
    return matrix_product(rot_z(x[0]), rot_x(np.pi/2), rot_z(np.pi + x[1]), rot_x(np.pi/2), rot_z(x[2] - np.pi))

