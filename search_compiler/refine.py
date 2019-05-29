import numpy as np
from . import circuits as circ
from . import utils

# most denominators that will be expected are fractions of 360 (pi/2,3,4,6,12 etc.) and powers of 2.  3600 in "large denominators" is supposed to correspond to the nearest tenth of a degree.  Custom denominator classes can be chosen for specific problems.
# Only use this if you need it because it generally decreases the quality of the solution
small_denominators = [12,16]
usual_denominators = [360,512]
large_denominators = [3600, 4096]

def nearest_pi(x, allowed_denominators=usual_denominators):
    frac = x / np.pi
    best_frac = (int(frac),1)
    best_score = np.abs(int(frac) - frac)
    for denom in allowed_denominators:
        numerator = int(frac * int(denom))
        score = np.abs(frac - float(numerator) / float(denom))
        if score < best_score:
            best_score = score
            best_frac = (numerator, denom)
    return np.pi * (float(best_frac[0]) / float(best_frac[1]))

def refine_circuit(target, circ, v, allowed_denominators=large_denominators):
    v2 = [nearest_pi(x, allowed_denominators) for x in v]
    new_score = utils.matrix_distance_squared(target, circ.matrix(v2))
    old_score = utils.matrix_distance_squared(target, circ.matrix(v))

    if new_score <= old_score:
        print("The refined vectors gave a better score than the old vectors!")
    else:
        print("WARNING: the original vector gave a better score than the refined vectors.  {} > {}".format(new_score, old_score))

    return v2



