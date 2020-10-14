"""
A collection of constant gates and gate generators that are unusual or more complicated than those found in unitaries.py.
"""
from qsearch.circuits import *
import scipy as sp

def generate_miro():
    """Generates a gate that was described as the X gate on the space of multiple qubits."""
    theta = np.pi/3
    c = np.cos(theta/2)
    s = -1j*np.sin(theta/2)
    return np.array([
        [c,0,0,0,0,0,0,s],
        [0,c,0,0,0,0,s,0],
        [0,0,c,0,0,s,0,0],
        [0,0,0,c,s,0,0,0],
        [0,0,0,s,c,0,0,0],
        [0,0,s,0,0,c,0,0],
        [0,s,0,0,0,0,c,0],
        [s,0,0,0,0,0,0,c]
        ], dtype='complex128')
mirogate = generate_miro()

def generate_HHL():
    def hadamard(theta=0):
        return np.array([[np.cos(2*theta), np.sin(2*theta)],[np.sin(2*theta), -np.cos(2*theta)]])
    H = UStep(hadamard(), "H")
    RCH8 = CUStep(hadamard(np.pi/8), "H8", flipped=True)
    RCH16 = CUStep(hadamard(np.pi/16), "H16", flipped=True)
    RCY = CUStep(np.array([[0,-1j],[1j,0]]), "CY", flipped=True)
    RCNOT = UStep(np.array([[0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]), "RCNOT")
    SWAP = UStep(np.array([[1,0,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1]]))

    # input parameters
    t0 = 2*np.pi
    A = np.array([[1.5, 0.5],[0.5, 1.5]])
    AU = sp.linalg.expm(1j * t0 * A / 2)

    CAU = CUStep(AU, "CA")
    CSH = CUStep(np.array([[1,0],[0,-1j]]), "CSH")

    X = UStep(np.array([[0,1],[1,0]]), "X")
    I = IdentityStep(2)

    circuit = ProductStep()
    circuit = circuit.appending(KroneckerStep(I,I,H))
    circuit = circuit.appending(KroneckerStep(I,CSH))
    circuit = circuit.appending(KroneckerStep(I,H,I))
    circuit = circuit.appending(KroneckerStep(I,I,H))
    circuit = circuit.appending(KroneckerStep(RCY, I))
    circuit = circuit.appending(KroneckerStep(SWAP,I))
    circuit = circuit.appending(KroneckerStep(I,RCY))
    circuit = circuit.appending(KroneckerStep(SWAP,I))
    return circuit.matrix([])
HHL = generate_HHL()

