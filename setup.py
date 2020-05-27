from setuptools import setup
from pathlib import Path

README = Path('README.md').read_text()

setup(
    long_description_content_type="text/markdown",
)
