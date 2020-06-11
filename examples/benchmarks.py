import qsearch
from qsearch import unitaries, advanced_unitaries
import time

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

    # 4 qubit benchmarks (WARNING: These may take days to run.)
    #project.add_compilation("qft4", unitaries.qft(16))
    #project.add_compilation("full adder", unitaries.full_adder)
    #project.add_compilation("ethylene", advanced_unitaries.ethylene)


    # compiler configuration example
    #project["gateset"] = qsearch.gatesets.QubitCNOTRing() # use this to synthesize for the ring topology instead of the default line topology
    #project["solver"]  = qsearch.solver.BFGS_Jac_SolverNative() # use this to force the compiler to use the Rust version of the BFGS solver instead of using the default setting
    #project["verbosity"] = 2 # use this to have more information reported to stdout and the log files, or set it to 0 to disable logging altogether

    # once everything is set up, let the project run!
    project.run()


    # once its done you can use the following functions to get output
    # compilation_names = project.compilations # returns a list of the names that were specified when adding unitaries
    # circuit, vector = project.get_result(compilation_names[0]) # get a circuit/vector combination, which is the search_compiler format for describing finished circuits
    # project.assemble(compilation_names[0], write_location="my_circuit.qasm") # export the circuit as openqasm
    # project.assemble(compilation_names[0], language=assembly.ASSEMBLY_QISKIT, write_location="my_circuit.py") # export the circuit as a qiskit script

    # Read the wiki for more information and more features: https://github.com/WolfLink/search_compiler/wiki
