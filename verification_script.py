import numpy as np
import SC_Utils as util
from numpy import matrix
from sys import argv

f = open(argv[1], "r")
original = eval(f.read())
f.close()
f = open(argv[2], "r")
final = eval(f.read())
f.close()

print("Matrix Distance: {}".format(util.matrix_distance(original, final)))

print("Running with 100 random vectors...")
n = np.shape(original)[0]
verbose = False
total = 0
if len(argv) > 3:
    verbose = argv[3] == "-v"
for _ in range(0, 100):
    v = np.array([np.random.uniform() * np.e**(1j*np.random.uniform(0,2*np.pi)) for _ in range(0, n)])
    v = v / np.linalg.norm(v) 

    # apply the matrices
    fv1 = np.ravel(np.dot(original, v))
    fv2 = np.ravel(np.dot(final, v))

    # calculate probability vectors
    p1 = np.real(np.multiply(fv1, np.conj(fv1)))
    p2 = np.real(np.multiply(fv2, np.conj(fv2)))

    #if verbose:
        #print("dot: {}\tprobsum: {}".format(np.vdot(v, v), sum(np.multiply(v, np.conj(v)))))
    #    print("Sanity Check: {}, {} (both should be close to 1)".format(sum(p1), sum(p2)))
    diff = 1-sum(np.abs(p1 - p2))
    print("Vector Match: {}%".format(diff * 100))
    total += diff

print("\n\nMatrix Match: {}%".format(total))

