# The heuristic is a function used to guide the search.  It is called on each node after it is evaluated, and the resulting value is used to position that node's children in the search queue.

def greedy(circ, v, weight, options):
    return options.eval_func(options.target, circ.matrix(v))

def astar(circ, v, weight, options):
    return 10.0*options.eval_func(options.target, circ.matrix(v)) + weight

def djikstra(circ, v, weight, options):
    return weight

