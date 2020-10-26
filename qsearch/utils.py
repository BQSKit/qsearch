"""
This module contains miscellaneous helper functions and tools.

The functions you may want to be aware of:

Attributes:
    endian_reverse : Reverses the endianness of the specified unitary.  Necessary for working with unitaries from Qiskit.
    matrix_distance_squared : The default error_func.  Returns the Hilbert-Schmidt norm between two matrices.
    matrix_distance_squared_jac : Returns the value that matrix_distance_squared would return, as well as the jacobian.
    matrix_residuals : The default error_residuals.  Returns residuals based on difference between the poduct of the implemented matrix and the hermitian conjugate of the target and the identitiy.
    matrix_residuals_jac : Returns the jacobian of matrix_residuals.  Does not return the value of matrix_residuals as well.
    remap : Remaps a unitary for acting on qudits in a different order.
    upgrade_qudits : Upgrades a unitary from a lower qudit size to a larger qudit size.
"""
import numpy as np
import scipy as sp
import scipy.linalg

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from . import unitaries

def matrix_product(*LU):
    """Performs matrix multiplication of a list of matrices."""
    result = np.array(np.eye(LU[0].shape[0]), dtype='complex128')
    for U in LU:
        result = np.matmul(U, result, out=result)
    return result

def matrix_kron(*LU):
    """Performs the kronecker product on a list of matrices."""
    result = LU[0]
    for U in LU[1:]:
        result = np.kron(result,U)
    return result

def op_norm(A):
    """An implementation of the l1-l1 operator norm."""
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

def matrix_residuals_v2(A, B, I):
    # TODO examine how this function behaves compared to the original implementation.  Faster?  Behaves better?  Roughly the same?  Exactly the same?
    M = B - A
    #M *= np.abs(M[0][0])/M[0][0]
    Re, Im = np.real(M), np.imag(M)
    Re = np.reshape(Re, (1,-1))
    Im = np.reshape(Im, (1,-1))
    return np.append(Re, Im)

def matrix_residuals_v2_jac(U, M, J):
    JU = np.array([np.append(np.reshape(np.real(K), (1,-1)), np.reshape(np.imag(K), (1,-1))) for K in J])
    return JU.T

def matrix_residuals_slice(slices, A, B, I):
    M = (B - A)[slices]
    #M *= np.abs(M[0][0])/M[0][0]
    Re, Im = np.real(M), np.imag(M)
    Re = np.reshape(Re, (1,-1))
    Im = np.reshape(Im, (1,-1))
    return np.append(Re, Im)

def matrix_residuals_slice_jac(slices, A, B, J):
    JU = np.array([np.append(np.reshape(np.real(K[slices]), (1,-1)), np.reshape(np.imag(K[slices]), (1,-1))) for K in J])
    return JU.T

def matrix_residuals_blacklist(badrows, badcols, A, B, I):
    M = (B - A)
    M = np.delete(M, badrows, 0)
    M = np.delete(M, badcols, 1)
    #M *= np.abs(M[0][0])/M[0][0]
    Re, Im = np.real(M), np.imag(M)
    Re = np.reshape(Re, (1,-1))
    Im = np.reshape(Im, (1,-1))
    return np.append(Re, Im)

def matrix_residuals_blacklist_jac(slices, A, B, J):
    JU = np.array([np.append(np.reshape(np.real(np.delete(np.delete(K, badrows, 0), badcols, 1)), (1,-1)), np.reshape(np.imag(np.delete(np.delete(K, badrows, 0), badcols, 1)), (1,-1))) for K in J])
    return JU.T

def nearest_unitary(A):
    """
    Calculate the closest unitary to a given matrix.

    Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A.

    D.M.Reich. "Characterisation and Identification of Unitary Dynamics
    Maps in Terms of Their Action on Density Matrices"

    Args:
        A (np.ndarray): The matrix input.

    Returns:
        (np.ndarray): The unitary matrix closest to A.
        Return U as a numpy matrix.

    Thank you to Ed Younis, this is based on code from qfast
    """
    try:
        if len(A.shape) == 2 and A.shape[0] != A.shape[1]:
            raise TypeError("A must be a square matrix.")

        V, __, Wh = sp.linalg.svd(A)
        U = np.array(V.dot(Wh))
        return U
    except Exception:
        return A

def index_test(i, di, df):
    if i < df:
        return False
    elif i % di >= df:
        return True
    else:
        return index_test(i//di)

def downgrade_qudits_residuals(di, df, A, B, I):
    M = (B - A)
    qudits = int(np.log(U.shape[0])/np.log(di))
    M = np.delete(M, [i for i in range(di**qudits) if index_test(i, di, df)], axis=0)
    M = np.delete(M, [i for i in range(di**qudits) if index_test(i, di, df)], axis=1)
    #M *= np.abs(M[0][0])/M[0][0]
    Re, Im = np.real(M), np.imag(M)
    Re = np.reshape(Re, (1,-1))
    Im = np.reshape(Im, (1,-1))
    return np.append(Re, Im)

def downgrade_qudits_residuals_jac(di, df, A, B, J):
    JU = [np.delete(K, [i for i in range(di**qudits) if index_test(i, di, df)], axis=0) for K in J]
    JU = [np.delete(K, [i for i in range(di**qudits) if index_test(i, di, df)], axis=1) for K in JU]
    JU = np.array([np.append(np.reshape(np.real(K), (1,-1)), np.reshape(np.imag(K), (1,-1))) for K in JU])
    return JU.T

def eval_func_from_residuals(f, A, B):
    return np.sum(np.square(f(A,B,I=np.array(np.eye(A.shape[0]), dtype='float64'))))

def generate_stateprep_target_matrix(state):
    # WARNING: the matrix generated by this function (currently) is not unitary, and therefore should only be used with functions like matrix_residuals_slice
    state = np.array(state, dtype='complex128')
    M = np.diag(state)
    M[0] = state
    return M

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
    # this code is now deprecated, because SingleQutritStep now contains a better implementation (more compact and more performant)
    # make sure to move this red comment over to SingleQutritStep before deleting this code for documentation purposes
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
    
def remap(U, order, d=2):
    U = np.array(U, dtype='complex128')

    swap = unitaries.general_swap(d)
    qudits = int(np.round(np.log(np.shape(U)[0]) / np.log(d)))
    beforemat = np.array(np.eye(np.shape(U)[0]), dtype='complex128')
    aftermat  = np.array(np.eye(np.shape(U)[0]), dtype='complex128')
    I = np.array(np.eye(d), dtype = 'complex128')
    if qudits == 1:
        return U
    current_order = [i for i in range(0, qudits)]
    for i in range(0, qudits):
        if not order[i] == current_order[i]:
            target_loc = i
            current_loc = current_order.index(order[i])
            while not target_loc == current_loc:
                if current_loc > target_loc:
                    # perform the swap current_loc and current_loc - 1
                    swapmat = matrix_kron(*[I]*(current_loc-1), swap, *[I]*(qudits - current_loc - 1))
#                    print("I"*(current_loc-1) + "SS" + "I" *(qudits - current_loc - 1))
                    current_order[current_loc], current_order[current_loc - 1] = current_order[current_loc - 1], current_order[current_loc]
                    beforemat = np.dot(beforemat, swapmat)
                    aftermat  = np.dot(swapmat, aftermat)
                    current_loc = current_loc - 1
                else:
                    # perform the swap current_loc and current_loc + 1
                    swapmat = matrix_kron(*[I]*(current_loc), swap, *[I]*(qudits - current_loc - 2))
 #                   print("I"*(current_loc) + "SS" + "I" *(qudits - current_loc - 2))
                    current_order[current_loc], current_order[current_loc + 1] = current_order[current_loc + 1], current_order[current_loc]
                    beforemat = np.dot(beforemat, swapmat)
                    aftermat  = np.dot(swapmat, aftermat)
                    current_loc = current_loc + 1

    return matrix_product(beforemat, U, aftermat)

def upgrade_qudits(U, di=2, df=3):
    qudits = int(np.log(U.shape[0])/np.log(di))
    new_unitary = np.array(np.eye(df**qudits), dtype='complex128')
    for i in range(df**qudits):
        skip = False
        testi = i
        oi = 0
        for dit in range(qudits):
            if testi % df >= di:
                skip = True
                break
            else:
                oi += (testi % df) *di**dit
            testi //= df

        if not skip:
            for j in range(df**qudits):
                skip = False
                testj = j
                oj = 0
                for dit in range(qudits):
                    if testj % df >= di:
                        skip = True
                        break
                    else:
                        oj += (testj % df) * di**dit
                    testj //= df

                if not skip:
                    new_unitary[i][j] = U[oi][oj]
    return new_unitary


def endian_reverse(U, d=2):
    n = int(np.log(U.shape[0])/np.log(d))
    return remap(U, list(reversed(range(0, n))), d)

def mpi_rank():
    if MPI is None:
        return 0 # allow fallback to multiprocessing
    comm = MPI.COMM_WORLD
    return comm.rank

def mpi_do_work(comm):
    """Do the work of a single compilation.

    Arguments:
        comm: An MPI communication object
    """
    done = False
    eval = None
    eval = comm.bcast(eval, root=0)
    while not done:
        done = comm.bcast(done, root=0)
        if done:
            break
        step = comm.recv(source=0, tag=comm.rank)
        if step is None:
            res = step # return None if we aren't doing work
        else:
            res = eval(step)
        data = comm.send(res, dest=0, tag=comm.rank)
        assert data is None

def mpi_worker():
    """Create a worker that will keep running compilation requests until told to stop"""
    # NOTE WELL: this should be kept in sync with the MPIParallelizer code in parallelizer.py
    if MPI is None:
        raise RuntimeError("MPI not installed")
    comm = MPI.COMM_WORLD
    done = False
    while not done:
        done = comm.bcast(done, root=0)
        if done:
            break
        mpi_do_work(comm)
