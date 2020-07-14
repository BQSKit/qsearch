from qsearch import Project, parallelizer, unitaries, utils


qft3 = unitaries.qft(8)

def test_sequential(project):
    project.add_compilation('qft2', unitaries.qft(4))
    project['parallelizer'] = parallelizer.SequentialParallelizer
    project.run()

def test_multiprocessing_parallelizer(project):
    project.add_compilation('qft3', qft3)
    project['parallelizer'] = parallelizer.MultiprocessingParallelizer
    project.run()

def test_processpool_parallelizer(project):
    project.add_compilation('qft3', qft3)
    project['parallelizer'] = parallelizer.ProcessPoolParallelizer
    project.run()
