from .post_processing import LEAPReoptimizing_PostProcessor
import warnings

def ReoptimizingCompiler(*args, **kwargs):
    warnings.warn("This name is deprecated, please use qsearch.post_processing.LEAPReoptimizing_PostProcessor.")
    return LEAPReoptimizing_PostProcessor(*args, **kwargs)
