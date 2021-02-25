"""
The functions in this module are used as heuristics to guide the search in SearchCompiler.


The required format for a heuristic is to take in a circuit, a vector of parameters for that circuit, a weight for that circuit, and an Options object, and to return a single real valued number that will be used to order the search tree.
"""
# The heuristic is a function used to guide the search.  It is called on each node after it is evaluated, and the resulting value is used to position that node's children in the search queue.

def greedy(circ, v, weight, options):
    """Defines a heuristic that results in greedy search, which focuses soley on minimizing the eval_func, and behaves somewhat similarly to depth first sarch."""
    return options.objective.gen_eval_func(circ, options)(v)

def astar(circ, v, weight, options):
    """Defines a heuristic that combines the weight of the circuit with the value from eval_func.  It generally gives similar quality results to djikstra, but with a drastic reduction in the number of node evaluations."""
    return 10.0*options.objective.gen_eval_func(circ, options)(v) + weight

def djikstra(circ, v, weight, options):
    """Defines a heuristic that relies only on the weight, which gurantees a minimal-weight final solution, at the expense of a long runtime.  It behaves somewhat similarly to breadth first search."""
    return weight

