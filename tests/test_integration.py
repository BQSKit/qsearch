from qsearch import Options, unitaries, backends

def test_per_compilation_options(project):
    project['backend'] = backends.SmartDefaultBackend()
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    qft3_opts = project._compilations['qft3']['options']
    assert 'backend' not in qft3_opts

def test_per_compilation_options2(project):
    opts = Options(backend=backends.PythonBackend())
    project.add_compilation('qft3', unitaries.qft(8), options=opts)
    qft2_opts = project._compilations['qft3']['options']
    assert isinstance(qft2_opts.backend, backends.PythonBackend)

def test_project_clear(project):
    project.add_compilation("qft3", unitaries.qft(8))
    project.run()
    project.clear()
    project.add_compilation("qft3", unitaries.qft(8))
    project.run()
