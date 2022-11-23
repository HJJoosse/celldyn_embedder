"""
 This is the setup script for the package.
"""

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hembedder",
    version="0.1",
    description="A package for embedding  hematological data",
    author="Huibert-Jan Joosse, Bram van Es, Chontira Chumsaeng",
    author_email="bes3@umcutrecht.nl",
    packages=["hembedder.prepping"],
    package_dir={"hembedder": "src"},
    install_requires=["numpy", "pandas", "scikit-learn", "scipy", "tqdm", "miceforest"],
)
