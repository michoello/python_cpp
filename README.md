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

## Installation
```
python3 -m pip install -e .
```

## Usage

```
# Run Example
python3 examples/invert_cli.py 1 2 3 4 5

# Tests
python3 -m unittest discover -s tests
```
