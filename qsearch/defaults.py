from . import utils, gatesets, solver, backend, parallelizer, heuristics



def default_eval_func(options):
    if options.error_func == utils.matrix_residuals:
        return utils.matrix_distance_squared
    else:
        return options.error_func

def default_heuristic(options):
    if options.search_type == "astar":
        return heuristics.astar
    if options.search_type == "breadth":
        return heuristics.breadth
    elif options.search_type == "greedy":
        return heuristics.greedy
    raise KeyError("Unknown search_type {}, and no alternative heuristic provided.".format(options.search_type))

def default_error_jac(options):
    if options.error_func == utils.matrix_distance_squared:
        return utils.matrix_distance_squared_jac
    else:
        return None

def default_error_resi_jac(options):
    if options.error_residuals == utils.matrix_residuals:
        return utils.matrix_residuals_jac
    else:
        return None

defaults = {
        "threshold":1e-10,
        "gateset":gatesets.Default(),
        "beams":-1,
        "depth":None,
        "search_type":"astar",
        "statefile":None,
        "error_func":utils.matrix_distance_squared,
        "error_residuals":utils.matrix_residuals,
        "backend":backend.SmartDefaultBackend(),
        "parallelizer":parallelizer.MultiprocessingParallelizer
        }
smart_defaults = {
        "eval_func":default_eval_func,
        "error_jac":default_error_jac,
        "error_resi_jac":default_error_resi_jac,
        "solver":solver.default_solver,
        "heuristic":default_heuristic
        }
