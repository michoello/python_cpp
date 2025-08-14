# listinvert (C++ version, GPU-ready)

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
cd listinvert

python3 -m pip install -e .   # editable mode, good for dev
```

## Usage

In listinvert dir
```
# Example
python3 examples/invert_cli.py 1 2 3 4 5

# Tests
python3 -m unittest discover -s tests
```
