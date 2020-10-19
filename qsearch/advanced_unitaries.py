"""
A collection of constant gates and gate generators that are unusual or more complicated than those found in unitaries.py.
"""
from qsearch.gates import *
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
    def hadamard(theta: int = 0):
        return np.array([[np.cos(2*theta), np.sin(2*theta)],[np.sin(2*theta), -np.cos(2*theta)]])
    H = UGate(hadamard(), "H")
    RCH8 = CUGate(hadamard(np.pi/8), "H8", flipped=True)
    RCH16 = CUGate(hadamard(np.pi/16), "H16", flipped=True)
    RCY = CUGate(np.array([[0,-1j],[1j,0]]), "CY", flipped=True)
    RCNOT = UGate(np.array([[0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]), "RCNOT")
    SWAP = UGate(np.array([[1,0,0,0],
                            [0,0,1,0],
                            [0,1,0,0],
                            [0,0,0,1]]))

    # input parameters
    t0 = 2*np.pi
    A = np.array([[1.5, 0.5],[0.5, 1.5]])
    AU = sp.linalg.expm(1j * t0 * A / 2)

    CAU = CUGate(AU, "CA")
    CSH = CUGate(np.array([[1,0],[0,-1j]]), "CSH")

    X = UGate(np.array([[0,1],[1,0]]), "X")
    I = IdentityGate()

    circuit = ProductGate()
    circuit = circuit.appending(KroneckerGate(I,I,H))
    circuit = circuit.appending(KroneckerGate(I,CSH))
    circuit = circuit.appending(KroneckerGate(I,H,I))
    circuit = circuit.appending(KroneckerGate(I,I,H))
    circuit = circuit.appending(KroneckerGate(RCY, I))
    circuit = circuit.appending(KroneckerGate(SWAP,I))
    circuit = circuit.appending(KroneckerGate(I,RCY))
    circuit = circuit.appending(KroneckerGate(SWAP,I))
    return circuit.matrix([])
HHL = generate_HHL()

