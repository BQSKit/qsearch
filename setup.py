from setuptools import setup
from pathlib import Path

README = Path('README.md').read_text()

setup(
    name="search_compiler",
    version="1.1.0",
    description="Search-Based Quantum Synthesis/Compilation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="LBNL - AQT",
    author_email="marc.davis@lbl.gov",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Compilers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='quantum compilers synthesis computing',
    packages=['search_compiler'],
    python_requires='>=3.6, <4',
    install_requires=['numpy', 'scipy'],
    extras_require={
        'cma': ['cma'],
        'graphics': ['matplotlib'],
        'native' : ['scrs >= 0.6'],
    },
)
