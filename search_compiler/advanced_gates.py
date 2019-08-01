from . import gates
from search_compiler.circuits import *
import scipy as sp


def generate_miro():
    theta = np.pi/3
    c = np.cos(theta/2)
    s = -1j*np.sin(theta/2)
    return np.matrix([
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
        return np.matrix([[np.cos(2*theta), np.sin(2*theta)],[np.sin(2*theta), -np.cos(2*theta)]])
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
    return circuit.matrix([])
HHL = generate_HHL()

ethelyne = np.matrix([[-3.72110828e-26+0.00000000e+00j,
          0.00000000e+00+3.64959242e-16j,
          0.00000000e+00-1.10343461e-15j,
         -9.56077279e-04+0.00000000e+00j,
          0.00000000e+00+4.30211323e-13j,
         -9.80931916e-17+0.00000000e+00j,
         -1.67107927e-03+0.00000000e+00j,
          0.00000000e+00-1.09001829e-15j,
          0.00000000e+00+4.31264073e-13j,
          1.66947769e-03+0.00000000e+00j,
         -8.32578326e-17+0.00000000e+00j,
          0.00000000e+00-8.75266866e-16j,
          9.99996753e-01+0.00000000e+00j,
          0.00000000e+00+4.31225894e-13j,
          0.00000000e+00+4.30227414e-13j,
          1.96261557e-17+0.00000000e+00j],
        [ 0.00000000e+00-5.88784672e-17j,
         -1.57107358e-16+0.00000000e+00j,
          1.73819961e-13+0.00000000e+00j,
          0.00000000e+00-9.47402328e-16j,
         -5.86370771e-17+0.00000000e+00j,
          0.00000000e+00+4.30377949e-13j,
          0.00000000e+00-6.75109091e-16j,
         -1.66948078e-03+0.00000000e+00j,
          8.63276464e-14+0.00000000e+00j,
          0.00000000e+00+4.31633895e-13j,
          0.00000000e+00-1.37383090e-16j,
         -1.66613730e-16+0.00000000e+00j,
          0.00000000e+00+4.30597857e-13j,
          9.99998606e-01+0.00000000e+00j,
          5.86008774e-17+0.00000000e+00j,
          0.00000000e+00+4.30173583e-13j],
        [ 0.00000000e+00+1.01852324e-16j,
          1.73530626e-13+0.00000000e+00j,
          1.57009246e-16+0.00000000e+00j,
          0.00000000e+00-8.18028561e-16j,
         -8.68872231e-14+0.00000000e+00j,
          0.00000000e+00+1.96261557e-17j,
          0.00000000e+00+4.31209900e-13j,
         -1.96189772e-17+0.00000000e+00j,
         -5.55111513e-17+0.00000000e+00j,
          0.00000000e+00+8.17333450e-16j,
          0.00000000e+00+4.30713007e-13j,
         -1.66948078e-03+0.00000000e+00j,
          0.00000000e+00+4.30793537e-13j,
         -1.96223654e-17+0.00000000e+00j,
          9.99998606e-01+0.00000000e+00j,
          0.00000000e+00+4.30676361e-13j],
        [ 1.01643879e-20+0.00000000e+00j,
          0.00000000e+00+6.46104266e-17j,
          0.00000000e+00+7.85046229e-17j,
          9.83085585e-20+0.00000000e+00j,
          0.00000000e+00-3.93602322e-17j,
         -8.65696403e-14+0.00000000e+00j,
         -5.88856485e-17+0.00000000e+00j,
          0.00000000e+00+4.30933067e-13j,
          0.00000000e+00+3.73391348e-26j,
         -2.76572453e-17+0.00000000e+00j,
         -8.66885773e-14+0.00000000e+00j,
          0.00000000e+00+4.30620383e-13j,
          5.89218214e-17+0.00000000e+00j,
          0.00000000e+00+4.30988582e-13j,
          0.00000000e+00+4.31382903e-13j,
          1.00000000e+00+0.00000000e+00j],
        [ 0.00000000e+00+4.31932261e-13j,
         -1.48002692e-19+0.00000000e+00j,
          1.66948078e-03+0.00000000e+00j,
          0.00000000e+00+5.94151199e-16j,
         -7.85044853e-17+0.00000000e+00j,
          0.00000000e+00-1.96261558e-17j,
          0.00000000e+00+8.09589009e-16j,
         -1.73530683e-13+0.00000000e+00j,
          9.99998606e-01+0.00000000e+00j,
          0.00000000e+00+4.31461408e-13j,
          0.00000000e+00+4.31887053e-13j,
          3.91158428e-17+0.00000000e+00j,
          0.00000000e+00+4.30675786e-13j,
         -8.69120314e-14+0.00000000e+00j,
          8.33232518e-17+0.00000000e+00j,
          0.00000000e+00+5.88784672e-17j],
        [ 5.88807132e-17+0.00000000e+00j,
          0.00000000e+00+4.31820750e-13j,
          0.00000000e+00+7.13441427e-16j,
          1.66947769e-03+0.00000000e+00j,
          0.00000000e+00-6.62038901e-16j,
         -3.71074698e-26+0.00000000e+00j,
          2.78716481e-06+0.00000000e+00j,
          0.00000000e+00+7.16675119e-16j,
          0.00000000e+00+4.30597857e-13j,
          9.99997216e-01+0.00000000e+00j,
          9.81525368e-17+0.00000000e+00j,
          0.00000000e+00+4.31748076e-13j,
         -1.66787765e-03+0.00000000e+00j,
          0.00000000e+00+4.30713007e-13j,
          0.00000000e+00+7.40422305e-16j,
          2.77526108e-17+0.00000000e+00j],
        [ 8.65973959e-14+0.00000000e+00j,
          0.00000000e+00+5.87240610e-17j,
          0.00000000e+00+4.31099607e-13j,
          1.96172556e-17+0.00000000e+00j,
          0.00000000e+00-9.25365672e-17j,
         -5.75982479e-20+0.00000000e+00j,
          3.27506399e-20+0.00000000e+00j,
          0.00000000e+00+5.88784672e-17j,
          0.00000000e+00+4.30933060e-13j,
          1.37494780e-16+0.00000000e+00j,
          1.00000000e+00+0.00000000e+00j,
          0.00000000e+00+4.30440848e-13j,
          5.52816833e-17+0.00000000e+00j,
          0.00000000e+00-1.96261558e-17j,
          0.00000000e+00+4.31689352e-13j,
          8.66493834e-14+0.00000000e+00j],
        [ 0.00000000e+00-1.96261557e-17j,
          8.69704897e-14+0.00000000e+00j,
          3.92684473e-17+0.00000000e+00j,
          0.00000000e+00+4.30932579e-13j,
          1.73530502e-13+0.00000000e+00j,
          0.00000000e+00-4.63496428e-17j,
          0.00000000e+00+6.21303748e-16j,
         -3.72812273e-26+0.00000000e+00j,
         -9.64964764e-26+0.00000000e+00j,
          0.00000000e+00+4.31071608e-13j,
          0.00000000e+00+4.31225894e-13j,
          9.99998606e-01+0.00000000e+00j,
          0.00000000e+00-5.03778755e-16j,
          1.38777878e-16+0.00000000e+00j,
          1.66948078e-03+0.00000000e+00j,
          0.00000000e+00+4.31430588e-13j],
        [ 0.00000000e+00+4.31485940e-13j,
          1.66948078e-03+0.00000000e+00j,
         -3.70542688e-26+0.00000000e+00j,
          0.00000000e+00-8.37747288e-16j,
          9.99998606e-01+0.00000000e+00j,
          0.00000000e+00+4.31304398e-13j,
          0.00000000e+00+4.31265333e-13j,
         -3.92515069e-17+0.00000000e+00j,
         -3.72495716e-26+0.00000000e+00j,
          0.00000000e+00+8.37005856e-16j,
          0.00000000e+00+1.29609593e-16j,
         -1.73819987e-13+0.00000000e+00j,
          0.00000000e+00+4.30821224e-13j,
          3.92515067e-17+0.00000000e+00j,
          8.64738608e-14+0.00000000e+00j,
          0.00000000e+00-3.92523115e-17j],
        [ 8.65905654e-14+0.00000000e+00j,
          0.00000000e+00+4.31522656e-13j,
          0.00000000e+00+5.88784672e-17j,
         -5.56421332e-17+0.00000000e+00j,
          0.00000000e+00+4.30283838e-13j,
          1.00000000e+00+0.00000000e+00j,
          7.84102293e-17+0.00000000e+00j,
          0.00000000e+00+4.30933067e-13j,
          0.00000000e+00+1.96261558e-17j,
         -2.35382481e-16+0.00000000e+00j,
          5.08219693e-20+0.00000000e+00j,
          0.00000000e+00+6.45844374e-17j,
          7.88960899e-17+0.00000000e+00j,
          0.00000000e+00+4.31099600e-13j,
          0.00000000e+00+5.89863719e-17j,
          8.65973959e-14+0.00000000e+00j],
        [ 1.94421605e-16+0.00000000e+00j,
          0.00000000e+00+6.81403660e-16j,
          0.00000000e+00+4.30768512e-13j,
         -1.67107927e-03+0.00000000e+00j,
          0.00000000e+00+4.31664161e-13j,
          5.88654546e-17+0.00000000e+00j,
          9.99997210e-01+0.00000000e+00j,
          0.00000000e+00+4.30754866e-13j,
          0.00000000e+00+7.53771536e-16j,
          2.78716481e-06+0.00000000e+00j,
         -3.71802638e-26+0.00000000e+00j,
          0.00000000e+00-7.21380634e-16j,
          1.66947769e-03+0.00000000e+00j,
          0.00000000e+00+6.93815271e-16j,
          0.00000000e+00+4.32098192e-13j,
          1.96140271e-17+0.00000000e+00j],
        [ 0.00000000e+00-3.92523114e-17j,
          6.55182862e-20+0.00000000e+00j,
         -8.63080224e-14+0.00000000e+00j,
          0.00000000e+00+4.30342719e-13j,
          3.92446757e-17+0.00000000e+00j,
          0.00000000e+00+4.31692879e-13j,
          0.00000000e+00+4.31461408e-13j,
          9.99998606e-01+0.00000000e+00j,
          1.73819852e-13+0.00000000e+00j,
          0.00000000e+00+7.17569149e-16j,
          0.00000000e+00+3.92523114e-17j,
          1.12757176e-23+0.00000000e+00j,
          0.00000000e+00+8.88696864e-16j,
          1.66948078e-03+0.00000000e+00j,
         -6.77635086e-21+0.00000000e+00j,
          0.00000000e+00+4.31821239e-13j],
        [ 1.00000000e+00+0.00000000e+00j,
          0.00000000e+00+4.31539912e-13j,
          0.00000000e+00+4.31044083e-13j,
          3.92412876e-17+0.00000000e+00j,
          0.00000000e+00+4.30398230e-13j,
         -8.66102979e-14+0.00000000e+00j,
          6.55126299e-20+0.00000000e+00j,
          0.00000000e+00+3.92523115e-17j,
          0.00000000e+00+4.30933070e-13j,
         -7.85062924e-17+0.00000000e+00j,
         -8.65418848e-14+0.00000000e+00j,
          0.00000000e+00-7.88135747e-17j,
          1.31064892e-19+0.00000000e+00j,
          0.00000000e+00-3.92523115e-17j,
          0.00000000e+00+1.84992813e-16j,
         -1.01644029e-20+0.00000000e+00j],
        [ 0.00000000e+00+4.30676361e-13j,
          9.99998606e-01+0.00000000e+00j,
          5.88501550e-17+0.00000000e+00j,
          0.00000000e+00+4.30960308e-13j,
         -1.66948078e-03+0.00000000e+00j,
          0.00000000e+00+4.30601843e-13j,
          0.00000000e+00+6.60602407e-16j,
          8.32667268e-17+0.00000000e+00j,
          1.96325296e-17+0.00000000e+00j,
          0.00000000e+00+4.31210473e-13j,
          0.00000000e+00+1.96261557e-17j,
         -8.63350828e-14+0.00000000e+00j,
          0.00000000e+00-4.64758222e-16j,
         -1.57009246e-16+0.00000000e+00j,
         -1.73819919e-13+0.00000000e+00j,
          0.00000000e+00-2.40630202e-16j],
        [ 0.00000000e+00+4.30007158e-13j,
          2.60035917e-20+0.00000000e+00j,
          9.99998606e-01+0.00000000e+00j,
          0.00000000e+00+4.30362343e-13j,
          2.77526061e-17+0.00000000e+00j,
          0.00000000e+00+1.17756934e-16j,
          0.00000000e+00+4.31522656e-13j,
          8.68532676e-14+0.00000000e+00j,
         -1.66948078e-03+0.00000000e+00j,
          0.00000000e+00-7.33374241e-16j,
          0.00000000e+00+4.30155911e-13j,
         -2.62124015e-19+0.00000000e+00j,
          0.00000000e+00-6.87578677e-16j,
         -1.73530497e-13+0.00000000e+00j,
          1.57009027e-16+0.00000000e+00j,
          0.00000000e+00+1.17756934e-16j],
        [ 3.92523114e-17+0.00000000e+00j,
          0.00000000e+00+4.30228313e-13j,
          0.00000000e+00+4.31304398e-13j,
          9.99996753e-01+0.00000000e+00j,
          0.00000000e+00-4.63118522e-16j,
          5.55052216e-17+0.00000000e+00j,
          1.66947769e-03+0.00000000e+00j,
          0.00000000e+00+4.31263919e-13j,
          0.00000000e+00-3.94056408e-16j,
         -1.66787765e-03+0.00000000e+00j,
          3.92752237e-17+0.00000000e+00j,
          0.00000000e+00+4.30210365e-13j,
          9.61651611e-04+0.00000000e+00j,
          0.00000000e+00-4.76854253e-16j,
          0.00000000e+00+9.73369318e-16j,
         -1.57009246e-16+0.00000000e+00j]], dtype='complex128')
