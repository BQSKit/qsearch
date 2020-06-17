try:
    from qsearch_rs import native_from_object
    RUST_ENABLED = True
except ImportError:
    try:
        from search_compiler_rs import native_from_object # compatability with the old version should be kept at least until its time to release the renamed version
        RUST_ENABLED = True
    except ImportError:
        RUST_ENABLED = False
        def native_from_object(o):
            raise Exception("Native code not installed.")

class Backend():
    def prepare_circuit(self, circ, options):
        raise NotImplementedError("Subclasses of Backend must implpement prepare_circuit")

class SmartDefaultBackend(Backend):
    def prepare_circuit(self, circuit, options):
        try:
            return native_from_object(circuit)
        except:
            return circuit

class PythonBackend(Backend):
    def prepare_circuit(self, circuit, options):
        return circuit

class NativeBackend(Backend):
    def prepare_circuit(self, circuit, options):
        return native_from_object(circuit)
