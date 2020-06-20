"""A pytest fixture to create a new qsearch project in a temporary directory"""
import pytest
from qsearch import Project

@pytest.fixture
def project(tmp_path):
    return Project(str(tmp_path))