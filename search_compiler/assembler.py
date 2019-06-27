from .circuits import *

def flatten_intermediate(intermediate):
    flattened = []
    for step in intermediate:
        if step[0] == "gate":
            flattened.append(step)
        elif step[0] == "block":
            flattened += flatten_intermediate(step[1])
        else:
            print("Found unexpected tuple type {} in intermediate language.".format(step[0]))
    return flattened

def assemble(circuit, v, assembly, write_location=None):
    il = flatten_intermediate(circuit.assemble(v))
    if write_location == None:
        out = assembly.initialize(circuit.dits)
    else:
        out = open(write_location, "w")
        out.write(assembly.initialize(circuit.dits))

    for step in il:
        parsed = assembly.parse(step)
        if parsed == None:
            print("Unable to compile intermediate language tuple {} to the target language.".format(step))
            continue

        if write_location == None:
            out += parsed
        else:
            out.write(parsed)

    if write_location == None:
        return out
    else:
        out.close()
        return None


class QuantumAssembly():
    # both of these functions should return a string corresponding to a line or lines of code in the respective language
    def initialize(self, dits):
        raise NotImplementedError()
    def parse(self, segment):
        # currently segments are layed out as follows:
        # segment[0] (unused)
        # segment[1] (gate type identifier as a string)
        # segment[2] (gate parameters)
        # segment[3] (qubit indices used, starting with the control)
        raise NotImplementedError()

class DictionaryAssembly(QuantumAssembly):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def initialize(self, dits):
        return self.dictionary["initial"].format(dits)

    def parse(self, segment):
        return self.dictionary[segment[1]].format(*segment[2], *segment[3])

qiskitdict = {
        "initial" : "qc = QuantumCircuit({})\n",
        "X" : "qc.rx({}, {})\n",
        "Y" : "qc.ry({}, {})\n",
        "Z" : "qc.rz({}, {})\n",
        "qiskit-u3" : "qc.u3({}, {}, {}, {})\n",
        "CNOT" : "qc.cx({}, {})\n",
}

# this dict is based off of how IBM assembles to qiskit from https://github.com/Qiskit/openqasm/blob/master/examples/generic/qelib1.inc
openqasmdict = {
        "initial" : 'OPENQASM 2.0;\nqreg q[{}];\n',
        "X" : "U({}, -pi/2, pi/2) q[{}];\n",
        "Y" : "U({}, 0, 0) q[{}];\n",
        "Z" : "U(0, 0, {}) q[{}];\n",
        "qiskit-u3" : "U({}, {}, {}) q[{}];\n",
        "CNOT" : "CX q[{}] q[{}];\n"
}


ASSEMBLY_QISKIT = DictionaryAssembly(qiskitdict)
ASSEMBLY_OPENQASM = DictionaryAssembly(openqasmdict)



