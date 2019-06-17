from .circuits import *

def assemble(circuit, v, language_dict, write_location=None):
    il = circuit.assemble(v)
    if write_location == None:
        out = "qc = QuantumCircuit({})\n\n".format(circuit._dits)
    else:
        out = open(write_location, "w")
        out.write("qc = QuantumCircuit({})\n\n".format(circuit._dits))

    for step in il:
        if not step[0] in language_dict:
            print("No specification for gate '{}' was provided in the language dict.".format(step[0]))
            continue

        stepstr = language_dict[step[0]].format(*step[2], *step[1])

        if write_location == None:
            out += stepstr
        else:
            out.write(stepstr)

    if write_location == None:
        return out
    else:
        out.close()
        return None


qiskit = {
        "Z" : "qc.rz({}, {})\n",
        "X" : "qc.rx({}, {})\n",
        "Y" : "qc.ry({}, {})\n",
        "CNOT" : "qc.cx({}, {})\n"
        }

