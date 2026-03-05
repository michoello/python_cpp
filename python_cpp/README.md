# listinvert (C++ version, GPU-ready)

## Clang formatting

```
sudo yum install -y clang-tools-extra

clang-format -i cpp/invert.h
clang-format -i cpp/test_matrix.cpp

```


## Preparation

Run anywhere

```bash
sudo yum groupinstall "Development Tools" -y

sudo yum install python3-devel -y

sudo yum install pip

pip install pybind11
```

## Build the cpp-python library (aka build_cppy.sh)
```
python3 -m pip install -e .
```

## Usage

```
# Run Example
python3 examples/invert_cli.py 1 2 3 4 5

# Tests (aka test_cppy.sh)
python3 -m unittest discover -s tests
```

## Benchmarks using multiplications of 100*100 matrices
```
# Python (50 multiplications in 8.87 seconds)
./bench_py.sh

# Cpp (5000 multiplications in 4.95 seconds)
./bench_cpp.sh

# Cpp + Python (5000 multiplications in 11 seconds)
./bench_cppy.sh
```
