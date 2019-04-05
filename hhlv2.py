from SampleImports import *
from SampleGates import *
from CMA_Solver import *
from CMA_Search_Compiler import *
import scipy as sp
import scipy.linalg

H = UStep(hadamard(), "H")
RCH8 = CUStep(hadamard(np.pi/8), "H8", flipped=True)
RCH16 = CUStep(hadamard(np.pi/16), "H16", flipped=True)
RCY = CUStep(np.matrix([[0,-1j],[1j,0]]), "CY", flipped=True)
RCNOT = UStep(np.matrix([[0,1,0,0],
                        [1,0,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]), "RCNOT")
SWAP = UStep(np.matrix([[1,0,0,0],
                        [0,0,1,0],
                        [0,1,0,0],
                        [0,0,0,1]]))

# input parameters
t0 = 2*np.pi
A = np.matrix([[1.5, 0.5],[0.5, 1.5]])
AU = sp.linalg.expm(1j * t0 * A / 2)

CAU = CUStep(AU, "CA")
CSH = CUStep(np.matrix([[1,0],[0,-1j]]), "CSH")

X = UStep(np.matrix([[0,1],[1,0]]), "X")
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




U = circuit.matrix([])

print("Generated the matrix")

threshold = 0.0000001
comp = CMA_Search_Compiler(threshold=threshold)
result = comp.compile(U, 10)

value = util.matrix_distance_squared(U, result[0])
if value < threshold:
    print("found a success!")
    f = open("hhlv2.qc", "w")
    f.write(repr(result[1]))
    f.close()



