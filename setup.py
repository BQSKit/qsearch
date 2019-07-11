from setuptools import setup
from pathlib import Path

README = Path('README.md').read_text()

setup(
    name="quantum_synthesis",
    version="0.1.0",
    description="Quantum Synthesis",
    long_description=README,
    long_description_content_type="text/markdown",
    author="LBNL - AQT",
    author_email="marc.davis@lbl.gov",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Compilers',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='quantum compilers synthesis computing',
    packages=['search_compiler'],
    python_requires='>=3.6, <4',
    install_requires=['numpy', 'threadpoolctl'],
    extras_require={
        'cma': ['cma'],
        'cobyla': ['scipy'],
        'bfgs': ['scipy'],
        'graphics': ['matplotlib'],
    },
)
