from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'tgalign._tgalign_cpp', 
        ['tgalign/src/dna_sketch.cpp'],
        include_dirs=[
            # Call str() explicitly to trigger the __str__ method
            str(get_pybind_include()), 
            str(get_pybind_include()) + '/user' 
        ],
        language='c++'
    ),
]

setup(
    name='tgalign',
    version='0.1.0',
    author='Justin Boone',
    description='Task-Geometry Alignment: High-performance alignment-free sequence search',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'numpy',
        'faiss-cpu', 
        'pybind11>=2.5.0',
        'scikit-learn'
    ],
    setup_requires=['pybind11>=2.5.0'],
    zip_safe=False,
)
