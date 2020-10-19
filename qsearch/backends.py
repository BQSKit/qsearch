"""
This module describes Backend, a class which is called before the Solver is run in order to replace a Python Qsearch circuit with a Qsearch circuit based on another implementation, such as Rust or GPU.

There are three provided Backend implementations:

Attributes:
    PythonBackend : This simply returns the Python circuit, such that Python and Numpy are used for computation.
    NativeBackend : This returns the converted circuit from native_from_object, which uses the Rust implementation of Qsearch circuits provided in the qsrs module.
    SmartDefaultBackend : This backend tries to use the native Rust backend, but if it fails to convert the circuit (such as if there are unsupported gates), it will fallback to Python rather than throwing an error.
"""

try:
    from qsrs import native_from_object
    RUST_ENABLED = True
except ImportError:
    RUST_ENABLED = False
    def native_from_object(o):
        raise Exception("Native code not installed.")
from .options import Options

class Backend():
    """This class prepares a circuit for solving, replacing a Python circuit with another implementation."""
    
    def __init__(self, options=Options()):
        self.options = options

    def prepare_circuit(self, circ, options=None):
        """This function accepts a Python Qsearch circuit and returns a Qsearch circuit with a different implementation.
        Args:
            circ : The Python Qsearch circuit to be converted.
        """
        raise NotImplementedError("Subclasses of Backend must implpement prepare_circuit")

class SmartDefaultBackend(Backend):
    """This Backend tries to use the native Rust code, but will gracefully fallback to Python if there is an issue."""

    def prepare_circuit(self, circuit, options=None):
        try:
            return native_from_object(circuit)
        except:
            return circuit

class PythonBackend(Backend):
    """This Backend will simply return the Python Qsearch circuit passed in, therefore using Python and Numpy for matrix computation."""

    def prepare_circuit(self, circuit, options=None):
        return circuit

class NativeBackend(Backend):
    """This Backend will use the native Rust implementation of Qsearch circuits for faster matrix computation."""
    def prepare_circuit(self, circuit, options=None):
        return native_from_object(circuit)
