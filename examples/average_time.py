import qsearch
from qsearch import unitaries, advanced_unitaries
import numpy as np

if __name__ == "__main__":
    with qsearch.Project("benchmarks") as project:
        project.clear()
        project["gateset"] = qsearch.gatesets.QubitCNOTRing()
        project.add_compilation("qft2", unitaries.qft(4))
        project.add_compilation("qft3", unitaries.qft(8))
        project.add_compilation("fredkin", unitaries.fredkin)
        project.add_compilation("toffoli", unitaries.toffoli)
        project.add_compilation("peres", unitaries.peres)
        project.add_compilation("or", unitaries.logical_or)

        project.add_compilation("miro", advanced_unitaries.mirogate)
        project.add_compilation("hhl", advanced_unitaries.HHL)


        TFIMS = {}
        for spec in [(3,3),(6,3),(42,3),(60,3)]:
            U = np.loadtxt(f"TFIMs/circuit_eph_1.0_time_{spec[0]}_{spec[1]}.unitary", dtype="complex128")
            TFIMS[f"TFIM_{spec[0]}_{spec[1]}"] = U

        for k in TFIMS:
            project.add_compilation(k, TFIMS[k])

        # run the benchmarks script with default settings 10x and average the timing results
        # reported times are between 0.1s and 5s on my 2018 Macbook Pro

        times = {}
        best_results = {}
        avg_result = {}
        for compilation in project.compilations:
            times[compilation] = 0
            best_results[compilation] = None
            avg_result[compilation] = 0

        for _ in range(10):
            project.reset()
            project.run()
            for compilation in project.compilations:
                times[compilation] += project.get_time(compilation)
                cnotcount = repr(project.get_result(compilation)["structure"]).count("CNOT")
                if best_results[compilation] is None:
                    best_results[compilation] = cnotcount
                elif best_results[compilation] > cnotcount:
                    best_results[compilation] = cnotcount

                avg_result[compilation] += cnotcount
        with open("final_output.tsv", "w") as out:
            for compilation in project.compilations:
                out.write(f"{compilation}\t{times[compilation]/10}s\t{best_results[compilation]}\t{avg_result[compilation]/10}\n")
