import search_compiler as sc
from search_compiler_rs import native_from_object
p = sc.Project("test")
p.clear()
p["solver"] = sc.solver.BFGS_Jac_SolverNative()
p.add_compilation("qft3", sc.unitaries.qft(8))
p.run()
U = sc.unitaries.qft(8)




circ, v = p.get_result("qft3")

print("comparison: {}".format(sc.utils.matrix_distance_squared(U, circ.matrix(v))))


