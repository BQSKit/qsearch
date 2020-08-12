from .circuits import *
from . import options as opt

class PostProcessor():
    def __init__(self, options = opt.Options()):
        self.options=options

    def post_process_circuit(self, circuit, vector, options=None):
        return circuit


class BasicSingleQubitReduction_PostProcessor(PostProcessor):
    def post_process_circuit(self, circuit, vector, options=None):
        options = self.options.updated(options)
        single_qubit_names = ["QiskitU3QubitStep()", "ZXZXZQubitStep()", "XZXZPartialQubitStep()"]
        identitystr = "IdentityStep(2)"
        circstr = repr(circuit)
        initial_count = sum([circstr.count(sqn) for sqn in single_qubit_names])
        options.logger.logprint("Initial count: {}".format(initial_count), verbosity=2)
        finalcirc = circuit
        finalx = vector
        for gate in single_qubit_names:
            components = circstr.split(gate)
            while len(components) > 1:
                newstr = components[0] + identitystr + "".join([component + gate for component in components[1:-1]]) + components[-1]
                newcirc = eval(newstr)
                mat, xopt = options.solver.solve_for_unitary(newcirc, options)
                if options.eval_func(options.target, mat) < options.threshold:
                    components = [components[0] + identitystr + components[1]] + components[2:]
                    finalx = xopt
                    finalcirc = newcirc
                else:
                    components = [components[0] + gate + components[1]] + components[2:]

            circstr = components[0]
        finalstr = repr(finalcirc)
        final_count = sum([finalstr.count(sqn) for sqn in single_qubit_names])
        options.logger.logprint("Final count: {}".format(final_count), verbosity=2)
        options.logger.logprint("Post-processing removed {}, or {}% of the single qubit gates".format(initial_count-final_count, 100*(initial_count-final_count)/initial_count))
        return {"structure":finalcirc, "vector":finalx}
