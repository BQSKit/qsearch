import numpy as np
import scipy as sp
import scipy.linalg
from . import unitaries

def matrix_product(*LU):
    # performs matrix multiplication of a list of matrices
    result = np.array(np.eye(LU[0].shape[0]), dtype='complex128')
    for U in LU:
        result = np.matmul(U, result, out=result)
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
    #return 1 - np.abs(np.trace(np.dot(A,B.T.conjugate()))) / A.shape[0]

def matrix_distance_squared_jac(U, M, J):
    S = np.sum(np.multiply(U, np.conj(M)))
    dsq = 1 - np.abs(S)/U.shape[0]
    if S == 0:
        return np.array([np.inf]*len(J))
    JU = np.array([np.multiply(U,np.conj(K)) for K in J])
    JUS = np.sum(JU, axis=(1,2))
    jacs = -(np.real(S)*np.real(JUS) + np.imag(S)*np.imag(JUS))*U.shape[0] / np.abs(S)
    return (dsq, jacs)

def matrix_residuals(A, B, I):
    M = np.dot(B,np.conj(A.T))
    #M *= np.abs(M[0][0])/M[0][0]
    Re, Im = np.real(M), np.imag(M)
    Re -= I
    Re = np.reshape(Re, (1,-1))
    Im = np.reshape(Im, (1,-1))
    return np.append(Re, Im)

def matrix_residuals_jac(U, M, J):
    Ut = np.conj(U.T)
    #M2 = np.dot(M, Ut)
    #Ut *= np.abs(M2[0][0])/M2[0][0]
    JU = [np.matmul(K, Ut) for K in J]
    JU = np.array([np.append(np.reshape(np.real(K), (1,-1)), np.reshape(np.imag(K), (1,-1))) for K in JU])
    return JU.T


def matrix_distance(A,B):
    return np.sqrt(np.abs(matrix_distance_squared(A,B)))

def re_rot_z(theta, old_z):
    old_z[0,0] = np.exp(-1j*theta/2)
    old_z[1,1] = np.exp(1j*theta/2)

def re_rot_z_jac(theta, old_z, multiplier=1):
    old_z[0,0] = multiplier*0.5*(-np.sin(theta/2)-1j*np.cos(theta/2))
    old_z[1,1] = multiplier*0.5*(-np.sin(theta/2)+1j*np.cos(theta/2))


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

    evaluated_unitary = np.array([[u11, u12, u13], [u21, u22, u23], [u31, u32, u33]])

    return evaluated_unitary

# based on old solovay kitaev code
def random_near_identity(n, alpha):
    # generate a random hermitian matrix
    H = np.array((np.random.rand(n,n) - 0.5) +1j * (np.random.rand(n,n) - 0.5))
    H = H + H.T.conjugate()
    # generate a unitary matrix from the hermitian matrix that is not far from the identity
    return np.array(sp.linalg.expm(1j * H * alpha))
    
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

        kl = 0
        for i in range(0, len(p1)):
            kl += p1[i] * np.log(p1[i]/p2[i]) / np.log(10)

#        diff = 1-np.sum(np.abs(p1-p2))/2
        total += kl
        if kl > maxs:
            maxs = kl
        if kl < mins:
            mins = kl
    return (maxs, total/count, mins)

def remap(U, order, d=2):
    U = np.array(U, dtype='complex128')
    if d != 2:
        raise NotImplementedError("This function is not yet implemented for dits other than qubits because I have not implemented the swap for those qudits yet.")

    dits = int(np.round(np.log(np.shape(U)[0]) / np.log(d)))
    beforemat = np.array(np.eye(np.shape(U)[0]), dtype='complex128')
    aftermat  = np.array(np.eye(np.shape(U)[0]), dtype='complex128')
    I = np.array(np.eye(d), dtype = 'complex128')
    if dits == 1:
        return U
    current_order = [i for i in range(0, dits)]
    for i in range(0, dits):
        if not order[i] == current_order[i]:
            target_loc = i
            current_loc = current_order.index(order[i])
            while not target_loc == current_loc:
                if current_loc > target_loc:
                    # perform the swap current_loc and current_loc - 1
                    swapmat = matrix_kron(*[I]*(current_loc-1), unitaries.swap, *[I]*(dits - current_loc - 1))
#                    print("I"*(current_loc-1) + "SS" + "I" *(dits - current_loc - 1))
                    current_order[current_loc], current_order[current_loc - 1] = current_order[current_loc - 1], current_order[current_loc]
                    beforemat = np.dot(beforemat, swapmat)
                    aftermat  = np.dot(swapmat, aftermat)
                    current_loc = current_loc - 1
                else:
                    # perform the swap current_loc and current_loc + 1
                    swapmat = matrix_kron(*[I]*(current_loc), unitaries.swap, *[I]*(dits - current_loc - 2))
 #                   print("I"*(current_loc) + "SS" + "I" *(dits - current_loc - 2))
                    current_order[current_loc], current_order[current_loc + 1] = current_order[current_loc + 1], current_order[current_loc]
                    beforemat = np.dot(beforemat, swapmat)
                    aftermat  = np.dot(swapmat, aftermat)
                    current_loc = current_loc + 1

    return matrix_product(beforemat, U, aftermat)


def endian_reverse(U, d=2):
    n = int(np.log(U.shape[0])/np.log(d))
    return remap(U, list(reversed(range(0, n))))

