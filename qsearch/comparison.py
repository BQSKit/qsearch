"""
This module contains functions for comparing matrices, vectors, and other numerical objects.  These functions do not all follow a standardized form, but many of these have a standardized version found in evaluation.py.
"""
import numpy as np

from . import utils

def matrix_distance_squared(A,B):
    """
    This is a distance function used to compare two matrices. It is phase agnostic and fast to calculate.

    Args:
        A : A unitary matrix in the form of a numpy ndarray.
        B : Another unitary matrix of the same size as A, as a numpy ndarray.

    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return np.abs(1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0])
    #original implementation
    #return 1 - np.abs(np.trace(np.dot(A,B.T.conjugate()))) / A.shape[0]
    
def matrix_distance(A,B):
    """
    The square root of matrix_distance_squared is more analgous to "distance", although for most purposes, working with a distance squared is fine, since inequalities hold.

    Args:
        A : A unitary matrix in the form of a numpy ndarray.
        B : Another unitary matrix of the same size as A, as a numpy ndarray.

    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    return np.abs(np.sqrt(np.abs(matrix_distance_squared(A,B))))

def matrix_distance_squared_jac(U, M, J):
    """
    The jacobian version of matrix_distance_squared.

    Args:
        U : A constant unitary matrix in the form of a numpy ndarray.
        M : A variable unitary matrix of the same size as A, as a numpy ndarray.
        J : A list of nump ndarrays representing the jacobians of M with respect to the parameters of interest.

    Returns:
        dsq : The matrix distance squared as a float (the same value that would be returned from matrix_distance_squared)
        jacs : A list of the derivative of dsq with resepect to each of the parameters.
    """
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

def matrix_residuals_blacklist_jac(badrows, badcols, A, B, J):
    JU = np.array([np.append(np.reshape(np.real(np.delete(np.delete(K, badrows, 0), badcols, 1)), (1,-1)), np.reshape(np.imag(np.delete(np.delete(K, badrows, 0), badcols, 1)), (1,-1))) for K in J])
    return JU.T


def distance_with_initial_state(stateA, stateB, A, B):
    return np.abs(1-np.abs(np.vdot(A.dot(stateA),B.dot(stateB))))

def distance_with_initial_state_jac(stateA,stateB,A,B,J):
    si = A.dot(stateA)
    s = np.vdot(si,B.dot(stateB))
    dist = 1-np.abs(s)
    vu = np.array([np.vdot(si, K.dot(stateB)) for K in J])
    jacs = -(np.real(s)*np.real(vu) + np.imag(s)*np.imag(vu))/np.abs(s)
    return (dist, jacs)

def residuals_with_initial_state(stateA, stateB,A,B,I):
    v = 1-np.conj(A.dot(stateA)) * B.dot(stateB)
    Re, Im = np.real(v), np.imag(v)
    return np.append(Re,Im)

def residuals_with_initial_state_jac(stateA,stateB, U, M, J):
    vc = np.conj(U.dot(stateA))
    vu = [-vc * K.dot(stateB) for K in J]
    vu = np.array([np.append(np.real(v),np.imag(v)) for v in vu])
    return vu.T



def eval_func_from_residuals(f, A, B):
    return np.sum(np.square(f(A,B,I=np.array(np.eye(A.shape[0]), dtype='float64'))))
