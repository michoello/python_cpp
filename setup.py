from setuptools import setup, Extension
import pybind11


ext_modules = [
    Extension(
        'listinvert._listinvert',
        sources=[
            "listinvert/bindings.cpp",
            "cpp/invert.cpp",       # include core cpp
        ],
        include_dirs=[
            pybind11.get_include(),
            "cpp"
        ],
        extra_compile_args=["-std=c++17"],
        #
        # TODO: Try adding -O3 for maximal performance
        # extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(
    name='listinvert',
    version='0.1',
    description='List inverter using C++ and pybind11 (GPU-ready)',
    ext_modules=ext_modules,
    packages=['listinvert'],
)

