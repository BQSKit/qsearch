import CMA_Solver as solver
import QASM_Gates as gates
import parse


# parses a qasm input file and returns a graph of CMA_Solver QuantumStep elements
def parse_qasm(infile):
    with open(infile, "r") as inqasm:

        # circuit specs (should be decided at beginning of file with current implementation)
        mindex = None
        maxdex = None

        steps = []
        vector = []

        for line in inqasm:
            elements = iter(line.split())
            element = next(elements)
            if element == "Allocate":
                if next(elements) != "|":
                    print("weird allocate: {}".format(line))
                    break
                index = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if index == None:
                    print("weird allocate: {}".format(line))
                
                if mindex == None or maxdex == None:
                    mindex = index
                    maxdex = index
                else:
                    if index > maxdex:
                        maxdex = index
                    elif index < mindex:
                        mindex = index
            if element == "X":
                if next(elements) != "|":
                    print("weird gate: {}".format(line))
                    break
                index = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if index == None:
                    print("weird gate: {}".format(line))
                step = gates.step_X
                I = gates.step_I
                steps.append(solver.KroneckerStep(*([I] * (index - mindex) + [step] + [I] * (maxdex - index))))

            if element == "H":
                if next(elements) != "|":
                    print("weird gate: {}".format(line))
                    break
                index = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if index == None:
                    print("weird gate: {}".format(line))
                step = gates.step_H
                I = gates.step_I
                steps.append(solver.KroneckerStep(*([I] * (index - mindex) + [step] + [I] * (maxdex - index))))

            if element == "CX":
                if next(elements) != "|" or next(elements) != "(":
                    print("weird gate: {}".format(line))
                    break
                ifrom = int(parse.parse("Qureg[{:d}],", next(elements))[0])
                ito = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if ifrom == None or ito == None:
                    print("weird gate: {}".format(line))
                if ito - ifrom != 1:
                    print("have to deal with this CNOT case: {}".format(line))
                    break
                step = gates.step_CX
                I = gates.step_I
                steps.append(solver.KroneckerStep(*([I] * (ifrom - mindex) + [step] + [I] * (maxdex - ito))))

            if "Rx" in element:
                angle = float(parse.parse("Rx({:g})", element)[0])
                if next(elements) != "|":
                    print("weird gate: {}".format(line))
                    break
                index = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if index == None:
                    print("weird gate: {}".format(line))
                step = gates.step_RX
                I = gates.step_I
                steps.append(solver.KroneckerStep(*([I] * (index - mindex) + [step] + [I] * (maxdex - index))))
                vector.append(angle)


            if "Rz" in element:
                angle = float(parse.parse("Rz({:g})", element)[0])
                if next(elements) != "|":
                    print("weird gate: {}".format(line))
                    break
                index = int(parse.parse("Qureg[{:d}]", next(elements))[0])
                if index == None:
                    print("weird gate: {}".format(line))
                step = gates.step_RZ
                I = gates.step_I
                steps.append(solver.KroneckerStep(*([I] * (index - mindex) + [step] + [I] * (maxdex - index))))
                vector.append(angle)

        circuit = solver.ProductStep(*steps)
        result = circuit.matrix(vector)
        return (result, circuit)



