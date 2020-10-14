"""
This example synthesizes a 5 qubit Quantum Fourier Transform for the IBM Yorktown
quantum computer.
"""
from qsearch import (
    gates, unitaries, Project,
    leap_compiler, post_processing,
    parallelizers, multistart_solvers,
    gatesets, assemblers,
)

if __name__ == '__main__':
    with Project('qft5-yorktown') as project:
            # Add a 5 qubit qft
            project.add_compilation("qft5", unitaries.qft(32))
            # configure qsearch to use the LEAP compiler, which scales to more qubits
            project["compiler_class"] = leap_compiler.LeapCompiler
            # set a miniumum search depth (to reduce frequent chopping that gets nowhere)
            project["min_depth"] = 3
            # use the IBM yorktown "bowtie" topology
            project["gateset"] = gatesets.QubitCNOTAdjacencyList([(0,1),(0,2),(1,2),(2,3),(2,4),(3,4)])
            # give verbose output to track the synthesis
            project["verbosity"] = 2
            # run synthesis
            project.run()
            # LEAP generates sub-optimal results, so we must re-synthesize to get the best results
            project.post_process(post_processing.LEAPReoptimizing_PostProcessor(), solver=multistart_solvers.MultiStart_Solver(24), parallelizer=parallelizers.ProcessPoolParallelizer, weight_limit=5)
            project.assemble("qft5", assembler=assemblers.ASSEMBLER_IBMOPENQASM)
