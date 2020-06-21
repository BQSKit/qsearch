from qsearch import Options, unitaries, backend

def test_per_compilation_options(project):
    opts = Options(backend=backend.PythonBackend())
    project['backend'] = backend.SmartDefaultBackend()
    project.add_compilation('qft2', unitaries.qft(4), options=opts)
    project.add_compilation('qft3', unitaries.qft(8))
    project.run()
    qft3_opts = project._compilations['qft3']['options']
    assert 'backend' not in qft3_opts
    qft2_opts = project._compilations['qft2']['options']
    assert isinstance(qft2_opts.backend, backend.PythonBackend)
