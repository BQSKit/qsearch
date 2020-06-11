try:
    from qsearch_rs import native_from_object
    RUST_ENABLED = False
except ImportError:
    RUST_ENABLED = False
    def native_from_object(o):
        raise Exception("Native code not installed.")

class Backend():
    def prepare_circuit(self, circ):
        raise NotImplementedError("Subclasses of Backend must implpement prepare_circuit")

class SmartDefaultBackend(Backend):
    def prepare_circuit(self, circuit):
        try:
            return native_from_object(circuit)
        except:
            return circuit

class PythonBackend(Backend):
    def prepare_circuit(self, circuit):
        return circuit

class NativeBackend(Backend):
    def prepare_circuit(self, circuit):
        return native_from_object(circuit)
