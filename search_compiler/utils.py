import numpy as np
import scipy as sp
import scipy.linalg

cnot = np.matrix([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]
                 ], dtype='complex128')

def matrix_product(*LU):
    # performs matrix multiplication of a list of matrices
    result = np.eye(LU[0].shape[0])
    for U in LU:
        result = np.dot(result, U, out=result)
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
    return np.sqrt(np.abs(matrix_distance_squared(A,B)))

def rot_z(theta):
    return np.matrix([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype='complex128')

def rot_x(theta):
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_y(theta):
    return np.matrix([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')


def re_rot_z(theta, old_z):
    old_z[0,0] = np.exp(-1j*theta/2)
    old_z[1,1] = np.exp(1j*theta/2)


# copied from qnl_analysis.SimTools
def q1_unitary(x):
    return matrix_product(rot_z(x[0]), rot_x(np.pi/2), rot_z(np.pi + x[1]), rot_x(np.pi/2), rot_z(x[2] - np.pi))

def qt_arb_rot(Theta_1, Theta_2, Theta_3, Phi_1, Phi_2, Phi_3, Phi_4, Phi_5):
    """Using the parameterization found in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.38.1994,
        this method constructs an arbitrary single_qutrit unitary operation.

        Arguments:
        qutrit_params: a list of eight parameters, in the following order
            Theta_1, Theta_2, Theta_3, Phi_1, Phi_2, Phi_3, Phi_4, Phi_5
        The formula for the matrix is:
            u11 = cos[Theta_1]*cos[Theta_2]*exp[i*Phi_1]
            u12 = sin[Theta_1]*exp[i*Phi_3]
            u13 = cos[Theta_1]*sin[Theta_2]*exp[i*Phi_4]
            u21 = sin[Theta_2]*sin[Theta_3]*exp[-i*Phi_4 - i*Phi_5] -
                    sin[Theta_1]*cos[Theta_2]*cos[Theta_3]*exp[i*Phi_1+i*Phi_2-i*Phi_3]
            u22 = cos[Theta_1]*cos[Theta_3]*exp[i*Phi_2]
            u23 = -cos[Theta_2]*sin[Theta_3]*exp[-i*Phi_1 - i*Phi_5] -
                    sin[Theta_1]*sin[Theta_2]*cos[Theta_3]*exp[i*Phi_2 - i*Phi_3 + i*Phi_4]
            u31 = -sin[Theta_1]*cos[Theta_2]*sin[Theta_3]*exp[i*Phi_1 - i*Phi_3 + i*Phi_5]
                    - sin[Theta_2]*cos[Theta_3]*exp[-i*Phi_2-i*Phi_4]
            u32 = cos[Theta_1]*sin[Theta_3]*exp[i*Phi_5]
            u33 = cos[Theta_2]*cos[Theta_3]*exp[-i*Phi_1 - i*Phi_2] -
                    sin[Theta_1]*sin[Theta_2]*sin[Theta_3]*exp[-i*Phi_3 + i*Phi_4 + i*Phi_5]


    """

    # construct unitary, element by element
    u11 = np.cos(Theta_1)*np.cos(Theta_2)*np.exp(1j*Phi_1)
    u12 = np.sin(Theta_1)*np.exp(1j*Phi_3)
    u13 = np.cos(Theta_1)*np.sin(Theta_2)*np.exp(1j*Phi_4)
    u21 = np.sin(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_4 - 1j*Phi_5) - np.sin(Theta_1)*np.cos(Theta_2)*np.cos(Theta_3)*np.exp(1j*Phi_1+1j*Phi_2-1j*Phi_3)
    u22 = np.cos(Theta_1)*np.cos(Theta_3)*np.exp(1j*Phi_2)
    u23 = -1*np.cos(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_1 - 1j*Phi_5) - np.sin(Theta_1)*np.sin(Theta_2)*np.cos(Theta_3)*np.exp(1j*Phi_2 - 1j*Phi_3 + 1j*Phi_4)
    u31 = -1*np.sin(Theta_1)*np.cos(Theta_2)*np.sin(Theta_3)*np.exp(1j*Phi_1 - 1j*Phi_3 + 1j*Phi_5) - np.sin(Theta_2)*np.cos(Theta_3)*np.exp(-1j*Phi_2-1j*Phi_4)
    u32 = np.cos(Theta_1)*np.sin(Theta_3)*np.exp(1j*Phi_5)
    u33 = np.cos(Theta_2)*np.cos(Theta_3)*np.exp(-1j*Phi_1 - 1j*Phi_2) - np.sin(Theta_1)*np.sin(Theta_2)*np.sin(Theta_3)*np.exp(-1j*Phi_3 + 1j*Phi_4 + 1j*Phi_5)

    evaluated_unitary = np.matrix([[u11, u12, u13], [u21, u22, u23], [u31, u32, u33]])

    return evaluated_unitary

# based on old solovay kitaev code
def random_near_identity(n, alpha):
    # generate a random hermitian matrix
    H = np.matrix((np.random.rand(n,n) - 0.5) +1j * (np.random.rand(n,n) - 0.5))
    H = H + H.H
    # generate a unitary matrix from the hermitian matrix that is not far from the identity
    return np.matrix(sp.linalg.expm(1j * H * alpha))
    
def random_vector_evaluation(A, B, count=1000):
    total = 0.0
    mins = 10.0
    maxs = -10.0
    n = np.shape(A)[0]

    for _ in range(0, count):
        v = np.array([np.random.uniform() * np.e**(1j*np.random.uniform(0,2*np.pi)) for _ in range(0, n)])
        v = v / np.sqrt(np.sum(np.multiply(v, np.conj(v))))

        fv1 = np.dot(A, v)
        fv2 = np.dot(B, v)

        p1 = np.real(np.multiply(fv1, np.conj(fv1)))
        p2 = np.real(np.multiply(fv2, np.conj(fv2)))

        diff = 1-np.sum(np.abs(p1-p2))/2
        total += diff
        if diff > maxs:
            maxs = diff
        if diff < mins:
            mins = diff
    return (maxs, total/count, mins)

