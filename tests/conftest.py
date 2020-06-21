"""A pytest fixture to create a new qsearch project in a temporary directory"""
import pytest
from qsearch import Project
from qsearch.utils import matrix_distance_squared

@pytest.fixture
def project(tmp_path):
    return Project(str(tmp_path))

@pytest.fixture
def check_project():
    def check(project):
        for comp in project.compilations:
            c, v = project.get_result(comp)
            assert matrix_distance_squared(c.matrix(v), project.get_target(comp)) < 1e-10
    return check