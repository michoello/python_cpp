from setuptools import setup, Extension
import pybind11

"""
ext_modules = [
    Extension(
        'listinvert._listinvert',
        ['listinvert/invert.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]
"""



ext_modules = [
    Extension(
        'listinvert._listinvert',
        sources=[
            "listinvert/bindings.cpp",
            "listinvert/invert.cpp",       # include core cpp
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        #extra_compile_args=["-O3", "-std=c++17"],
        extra_compile_args=["-std=c++17"],
    )
]



setup(
    name='listinvert',
    version='0.1',
    description='List inverter using C++ and pybind11 (GPU-ready)',
    ext_modules=ext_modules,
    packages=['listinvert'],
)

