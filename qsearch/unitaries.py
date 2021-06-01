"""
This module contains a list of predefined commonly used constant unitaries, and functions for generating commonly used unitaries.

Attributes:

    cnot : Constant `CNOT` unitary
    sqrt_cnot : Constant square root of `CNOT` unitary
    swap : Constant 2 qubit swap unitary
    toffoli : Constant toffoli unitary
    fredkin : Constant fredkin unitary
    peres : Constant peres unitary
    logical_or : Constant logical or unitary
    full_adder : Constant adder unitary

    rot_x : Function to generate an `X` rotation by `theta`
    rot_x_jac : Function that returns the jacobian of rot_x()
    rot_y : Function to generate an `Y` rotation by `theta`
    rot_y_jac : Function that returns the jacobian of rot_y()
    rot_z : Function to generate an `Z` rotation by `theta`
    rot_z_jac : Function that returns the jacobian of rot_z()
    qft : Returns a `n`x`n` qft matrix.
    identity : Returns a `n`x`n` identity matrix
    general_swap : Returns the swap matrix for qudits of the specified size.
    arbitrary_cnot : Returns a CNOT between any two qubits within the specified number of qubits
"""
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


pauli_x = np.array([[0,1],
                    [1,0]],
                    dtype='complex128')

pauli_y = np.array([[0,-1j],
                    [1j,0]],
                    dtype='complex128')

pauli_z = np.array([[1,0],
                    [0,-1]],
                    dtype='complex128')


# This definition is designed to match IBM Qiskit https://qiskit.org/documentation/stubs/qiskit.circuit.library.SXGate.html#qiskit.circuit.library.SXGate
sqrt_x = np.array([[0.5+0.5j,0.5-0.5j],
                   [0.5-0.5j,0.5+0.5j]],
                   dtype='complex128')

# Full adder that converts ABC0 to ABCS.  Behavior is not well defined when the last input bit is a 1.

def rot_z(theta: float):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype='complex128')

def rot_z_jac(theta: float):
    return np.array([[-1j/2*np.exp(-1j*theta/2), 0], [0, 1j/2*np.exp(1j*theta/2)]], dtype='complex128')

def rot_x(theta: float):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_x_jac(theta: float):
    return np.array([[-1/2*np.sin(theta/2), -1j/2*np.cos(theta/2)], [-1j/2*np.cos(theta/2), -1/2*np.sin(theta/2)]], dtype='complex128')

def rot_y(theta: float):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')

def rot_y_jac(theta: float):
    return np.array([[-1/2*np.sin(theta/2), -1/2*np.cos(theta/2)], [1/2*np.cos(theta/2), -1/2*np.sin(theta/2)]], dtype='complex128')

def qft(n: int):
    root = np.e ** (2j * np.pi / n)
    Q = np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)
    return Q

def identity(n: int): # not super necessary but saves a little code length
    return np.array(np.eye(n), dtype='complex128')

def general_swap(d: int = 2): # generates the swap matrix for qu-qudits
    f = lambda i, j: (i % d == j//d) and (i // d == j % d)
    return np.array(np.fromfunction(np.vectorize(f), (d**2, d**2)), dtype='complex128')

# generates an arbitrary cnot gate by classical logic and brute force
# it may be a good idea to write a better version of this at some point, but this should be good enough for use with the search compiler on 2-4 qubits.
def arbitrary_cnot(qudits: int, control: int, target: int):
    # this test returns if the given matrix index correspond to a "true" value of the bit at selected qubit index
    test = lambda x, value: (x // 2**(qudits-value-1)) % 2 != 0
    def f(i,j):
        # unless the control is true in both columns, just return part of the identity
        if not (test(i, control) and test(j, control)):
            return i==j
        elif test(i, target) != test(j, target):
            # if the control is true in both columns and the target is mismatched, verify everything else is matched
            for k in range(0, qudits):
                if k == target or k == control:
                    continue
                elif test(i, k) != test(j, k):
                    return 0
            return 1
        else:
            # if the control is true and matched and the target is matched, return 0
            return 0
    return np.array(np.fromfunction(np.vectorize(f), (2**qudits,2**qudits)),dtype='complex128')

