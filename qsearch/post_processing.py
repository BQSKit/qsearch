from .circuits import *
from . import options as opt

class PostProcessor():
    def __init__(self, options = opt.Options()):
        self.options=options

    def post_process_circuit(self, result, options=None):
        return result


class BasicSingleQubitReduction_PostProcessor(PostProcessor):
    def post_process_circuit(self, result, options=None):
        circuit = result["structure"]
        finalx = result["vector"]
        options = self.options.updated(options)
        single_qubit_names = ["QiskitU3QubitStep()", "ZXZXZQubitStep()", "XZXZPartialQubitStep()"]
        identitystr = "IdentityStep(2)"
        circstr = repr(circuit)
        initial_count = sum([circstr.count(sqn) for sqn in single_qubit_names])
        options.logger.logprint("Initial count: {}".format(initial_count), verbosity=2)
        finalcirc = circuit
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

class ParameterTuning_PostProcessor(PostProcessor):
    def post_process_circuit(self, result, options=None):
        circuit = result["structure"]
        initialx = result["vector"]
        options = self.options.updated(options)
        options.max_quality_optimization = True
        initial_value = options.eval_func(options.target, circuit.matrix(initialx))
        options.logger.logprint("Initial Distance: {}".format(initial_value))

        U, x = options.solver.solve_for_unitary(circuit, options)

        final_value = options.eval_func(options.target, U)
        if np.abs(final_value) < np.abs(initial_value):
            options.logger.logprint("Improved Distance: {}".format(final_value))
            return {"vector":x}
        else:
            options.logger.logprint("Rejected Distance: {}".format(final_value))
            return {}
        
