from setuptools import setup
from setuptools import Extension
import pybind11

ext_modules = [
    Extension(
        'listinvert._listinvert',
        ['listinvert/invert.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='listinvert',
    version='0.1',
    description='List inverter using C++ and pybind11 (GPU-ready)',
    ext_modules=ext_modules,
    packages=['listinvert'],
)
