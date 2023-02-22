"""
 This is the setup script for the package.
"""
#from setuptools import setup

from distutils.core import setup, Extension
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    # Try building with Cython
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                "src.utils._cpython._metrics_cy", ["src/utils/_cpython/_metrics_cy.pyx"]
                
            )
        ]
    )
except ImportError:
    # Else just use the C file from the repo
    from distutils.command.build_ext import build_ext

    ext_modules = [Extension("src.utils._cpython._metrics_cy", ["src/utils/_cpython/_metrics_cy.c"])]

setup(
    name="hembedder",
    version="0.1",
    description="A package for embedding  hematological data",
    author="Huibert-Jan Joosse, Bram van Es, Chontira Chumsaeng, Jille van der Togt",
    author_email="bes3@umcutrecht.nl",
    packages=["hembedder.prepping", "hembedder.utils"],
    package_dir={"hembedder": "src"},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    install_requires=["numpy", "pandas", "scikit-learn", "scipy", "tqdm", "miceforest", "cython"],
)
