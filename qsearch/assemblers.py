"""
This module defines the Assembler class, which is used to convert a Qsearch-style circuit into other formats, such as Qiskit or Qasm.

The DictionaryAssembler subclass is provided as the default implementation of an Assembler.  Use it as-is or as an example for writing your own Assembler.
Some constants are also defined as DictionaryAssembler instances preloaded with the most common assembly dictionaries.

Attributes:
    ASSEMBLER_QISKIT : Outputs Python code that generates a Qiskit circuit object.
    ASSEMBLER_OPENQASM : Outputs generic Openqasm code.  This may not be compatible with IBM Qiskit.
    ASSEMBLER_IBMOPENQASM : Outputs Openqasm code with the IBM imports and gate names.  This flavor of Openqasm is compatible with IBM Qiskit.
    ASSEMBLER_QUTRIT : Outputs pseudocode for circuits built with single-qutrit gates and CNOTs.
"""

from .options import Options

class Assembler():
    """This class is used to translate Qsearch-style circuits to other formats."""

    def __init__(self, options=Options()):
        self.options = options

    def assemble(self, resultdict, options=None):
        """
        The assemble function is used to convert the circuit described in resultdict.  See DictionaryAssembler for an example implementation.

        Args:
            resultdict : The dictionary representing the desired circuit.  It is expected to contain the entries "stucture" and "parameters".  It may contain other entries.

        Returns:
            str: A string representing the converted circuit code.
        """
        raise NotImplementedError()

class DictionaryAssembler(Assembler):
    """This subclass of Assembler uses a dictionary that specifies mappings from gate names to output code, as well as an output code initial line.

        Options:
        assemblydict (required) : A dictionary that specifies mappings from gate names to output code.
    """

    def assemble(self, resultdict, options=None):
        options = self.options.updated(options)
        options.make_required("assemblydict")

        circuit, v = resultdict["structure"], resultdict["parameters"]
        il = flatten_intermediate(circuit.assemble(v))
        assemblydict = options.assemblydict

        out = assemblydict["initial"].format(circuit.qudits)

        for segment in il:
            if not segment[0] == "gate":
                continue
            try:
                parsed = assemblydict[segment[1]].format(*segment[2], *segment[3])
            except:
                raise KeyError("Unable to compile gate {} to the language defined by{}".format(segment, assemblydict))
            out += parsed

        return out


def flatten_intermediate(intermediate):
    """This is a helper function for working with the intermediate tuple language that is output by the assemble method of QuantumStep objects."""
    flattened = []
    for step in intermediate:
        if step[0] == "gate":
            flattened.append(step)
        elif step[0] == "block":
            flattened += flatten_intermediate(step[1])
        else:
            raise KeyError("Found unexpected tuple type {} in assembly tuple {}".format(step[0], step))
    return flattened





# These are example dictionaries for use with DictionaryAssembler.  The can be modified or used as examples for writing your own such dictionaries.

assemblydict_qiskit = {
        # A dictionary that allows DictionaryAssembler to export Python code to generate a Qiskit circuit
        "initial" : "num_qubits = {}\nqr = QuantumRegister(num_qubits)\ncr = ClassicalRegister(num_qubits)\nqc = QuantumCircuit(qr,cr)\n",
        "X" : "qc.rx({}, {})\n",
        "Y" : "qc.ry({}, {})\n",
        "Z" : "qc.rz({}, {})\n",
        "U3" : "qc.u({}, {}, {}, {})\n",
        "SX" : "qc.sx({})\n",
        "CNOT" : "qc.cx({}, {})\n",
        "CZ": "qc.cz({}, {})\n",
        "XX": "qc.rxx({}, {})\n",
        "ISWAP": "qc.iswap({}, {})\n",
}

# Currently ibm's quantum experience cannot handle normal openqasm, even though supposedly u3 and cx below should evaluate to U3 and CX above.
# This pure openqasm does not seem to be supported by IBM.  I have double checked that the syntax is theoretically correct.  The IBM Quantum Experience circuit builder will only recognize the ibm versions.
assemblydict_openqasm = {
        # A dictionary that allows DictionaryAssembler to export code to generate an Openqasm script.
        "initial" : 'OPENQASM 2.0;\nqreg q[{}];\n',
        "X" : "U({}, -pi/2, pi/2) q[{}];\n",
        "Y" : "U({}, 0, 0) q[{}];\n",
        "Z" : "U(0, 0, {}) q[{}];\n",
        "SX" : "U(pi/2, -pi/2, pi/2) q[{}];\n",
        "U3" : "U({}, {}, {}) q[{}];\n",
        "CNOT" : "CX q[{}], q[{}];\n",
        # these are based on the official definition of CZ/XX/ISWAP
        "CZ" : "U(pi/2, 0, pi) {1};\nCX {0},{1};\nU(pi/2, 0, pi) {0};\n",
        "XX" : "U(pi/2, 0, pi) {0};\nU(pi/2, 0, pi) {1};\nCX {0},{1};\nU(0, 0, pi/2) {1};\nCX {0},{1};\nU(pi/2, 0, pi) {1};\nU(pi/2, 0, pi) {0};\n",
        "ISWAP": "U(0, 0, pi/2) {0};\nU(0, 0, pi/2) {1};\nU(pi/2, 0, pi) {0};\nCX {0},{1};\nCX {1},{0};\nU(pi/2, 0, pi) {1};\n",
}

# currently ibm's quantum experience cannot handle normal openqasm, even though supposedly u3 and cx below should evaluate to U3 and CX above.
assemblydict_ibmopenqasm = {
        # A dictionary that allows DictionaryAssembler to export code to generate an Openqasm script that is compatible with IBM Qiskit.
        "initial" : 'OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\nqreg q[{}];\n',
        "X" : "rx({}) q[{}];\n",
        "Y" : "ry({}) q[{}];\n",
        "Z" : "rz({}) q[{}];\n",
        "SX" : "sx q[{}];\n",
        "U3" : "u3({}, {}, {}) q[{}];\n",
        "CNOT" : "cx q[{}], q[{}];\n",
        "CZ" : "cz q[{}], q[{}];\n",
        "XX" : "rxx(pi/2) q[{}], q[{}];\n",
        # this is based on the official definition of ISWAP
        "ISWAP": "s q[{0}];\ns q[{1}];\nh q[{0}];\ncx q[{0}],q[{1}];\ncx q[{1}],q[{0}];\nh q[{1}];",
}

assemblydict_qutrit = {
        # A dictionary that allows DictionaryAssembler to export a human-readable pseudocode representation of a circuit using single qutrit gates and CNOTs.
        "initial" : "",
        "QUTRIT" : "qutrit({}, {}, {}, {}, {}, {}, {}, {}) index: {}\n",
        "CNOT" : "cnot control: {} target: {}\n"
}


# Some constants for the most common use cases.  These are instances of Assembler subclasses, and can be passed unmodified to Project or used on a Qsearch-style circuit.
ASSEMBLER_QISKIT = DictionaryAssembler(Options(assemblydict=assemblydict_qiskit))
ASSEMBLER_OPENQASM = DictionaryAssembler(Options(assemblydict=assemblydict_openqasm))
ASSEMBLER_IBMOPENQASM = DictionaryAssembler(Options(assemblydict=assemblydict_ibmopenqasm))
ASSEMBLER_QUTRIT = DictionaryAssembler(Options(assemblydict=assemblydict_qutrit))
