from multiprocessing import cpu_count
import qsearch
from qsearch import unitaries, advanced_unitaries, post_processing, multistart_solvers, leap_compiler

import numpy as np

# This simple script demonstrates how to use post-processing to reduce the number of single-qubit gates in the final solution.

# With default settings, qsearch will add far more single qubit gates than necessary in order to prioritize runtime and CNOT reduction.  If the final single qubit gate count is too high, post processing can be used to reduce it.  It does so by trying different variants of the circuit with the same CNOT structure, but with certain single qubit gates changed to the identity.

if __name__ == "__main__":
    # create the project
    with qsearch.Project("benchmarks") as project:
        # add a unitaries to compile
        project.add_compilation("qft2", unitaries.qft(4))
        project.add_compilation("qft3", unitaries.qft(8))
        project.add_compilation("fredkin", unitaries.fredkin)
        project.add_compilation("toffoli", unitaries.toffoli)
        project.add_compilation("peres", unitaries.peres)
        project.add_compilation("or", unitaries.logical_or)

        project.add_compilation("miro", advanced_unitaries.mirogate)
        project.add_compilation("hhl", advanced_unitaries.HHL)

        # 4 qubit benchmarks are generally too hard to run with the normal compiler
        # the Leap compiler runs much faster but generally produces longer circuits
        LeapClass = leap_compiler.LeapCompiler
    #    project.add_compilation("qft4", unitaries.qft(16), compiler_class=LeapClass)
    #    project.add_compilation("full adder", unitaries.full_adder, compiler_class=LeapClass)
        # project.add_compilation("ethylene", advanced_unitaries.ethylene, compiler_class=LeapClass) ethylene is hard even for the Leap compiler

        project.run()

        # after running the project, run post-processing to reduce single qubit gate count
        # I've used the multistart solver for better post-processing results.  Increase the number of threads that you give the post-processor to increase the likelihood that you get the optimal final circuit.
        project.post_process(post_processing.BasicSingleQubitReduction_PostProcessor(), solver=multistart_solvers.MultiStart_Solver(cpu_count()))
        for name in project.compilations:
            target = project.get_target(name)
            result = project.get_result(name)
            while np.abs(project.options.eval_func(target, result["structure"].matrix(result["vector"]))) > 1e-15:
                project.post_process(post_processing.ParameterTuning_PostProcessor(), name, solver=multistart_solvers.MultiStart_Solver(16), inner_solver=qsearch.solver.LeastSquares_Jac_Solver())
                result = project.get_result(name)

