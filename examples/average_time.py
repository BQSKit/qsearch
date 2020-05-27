import search_compiler as sc
from search_compiler import unitaries, advanced_unitaries
import time
project = sc.Project("benchmarks")
project.add_compilation("qft2", unitaries.qft(4))
project.add_compilation("qft3", unitaries.qft(8))
project.add_compilation("fredkin", unitaries.fredkin)
project.add_compilation("toffoli", unitaries.toffoli)
project.add_compilation("peres", unitaries.peres)
project.add_compilation("or", unitaries.logical_or)

project.add_compilation("miro", advanced_unitaries.mirogate)
project.add_compilation("hhl", advanced_unitaries.HHL)

# run the benchmarks script with default settings 10x and average the timing results
# reported times are between 0.1s and 5s on my 2018 Macbook Pro

times = {}
for compilation in project.compilations:
    times[compilation] = 0

for _ in range(10):
    project.reset()
    project.run()
    for compilation in project.compilations:
        times[compilation] += project.get_time(compilation)

for compilation in project.compilations:
    print(f'Compilation {compilation} took {times[compilation]/10}s on average.')
