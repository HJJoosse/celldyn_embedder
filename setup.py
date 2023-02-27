"""
 This is the setup script for the package.

 install.packages("Rcpp")
 
"""
# from setuptools import setup
# https://github.com/himbeles/ctypes-example/blob/master/setup.py

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy

import os

#os.environ["LD_LIBRARY_PATH"] = "/usr/share/R/include:" + os.environ["LD_LIBRARY_PATH"]

#os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

with open("README.md", "r") as fh:
    long_description = fh.read()


class build_ext(build_ext):
    # https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


class CTypes(Extension):
    pass


try:
    # Try building with Cython
    # from Cython.Distutils import build_ext
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                name="hembedder.utils._cpython._metrics_cy",
                sources=["src/utils/_cpython/_metrics_cy.pyx"],
            )
        ]
    )
    ext_modules.append(
        CTypes(
            name="hembedder.utils._ctypes.coranking",
            sources=["src/utils/_ctypes/coranking.cpp"]
        )
    )
except ImportError:
    ext_modules = [
        Extension(
            "hembedder.utils._cpython._metrics_cy", ["src/utils/_cpython/_metrics_cy.c"]
        )
    ]
    ext_modules.append(
        CTypes(
            name="hembedder.utils._ctypes.coranking",
            sources=["src/utils/_ctypes/coranking.cpp"],
        )
    )
# dlls = [("hembedder.utils._ctypes", ["src/utils/_ctypes/coranking.so"])]

setup(
    name="hembedder",
    version="0.1",
    description="A package for embedding  hematological data",
    author="Huibert-Jan Joosse, Bram van Es, Chontira Chumsaeng, Jille van der Togt",
    author_email="bes3@umcutrecht.nl",
    packages=["hembedder.prepping", "hembedder.utils"],
    package_dir={"hembedder": "src"},
    py_modules=["hembedder"],
    # data_files=dlls,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "miceforest",
        "cython",
    ],
)
