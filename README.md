# Py(W)STL: Python (Weighted) Signal Temporal Logic Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![codecov](https://codecov.io/gh/ruyakrgl/pywstl/graph/badge.svg?token=S68VJhKilG)](https://codecov.io/gh/ruyakrgl/pywstl)

A comprehensive Python library for constructing, evaluating, and optimizing **Weighted Signal Temporal Logic (WSTL)** formulas.

## Features
- **Weighted Robustness**: Quantitative semantics with weights. If weights are set to 1, reduces to standard STL robustness.
- **Dual Backends** with **Unified API**:
  - switch backends with a single call; all formula classes work identically across backends
  - Backends:
    - `numpy` (default): numpy robustness computation via `wstlpy`
    - `torch`: differentiable robustness via `wstltorch`, compatible with gradient-based optimization and cuda
## Installation

### From source
```bash
git clone https://github.com/ruyakrgl/pywstl.git
cd pywstl
pip install -e .
```

### Requirements
- Python ≥ 3.9
- NumPy ≥ 1.20
- PyTorch ≥ 2.0 *(optional — required only for the `torch` backend)*


## Examples

See the `examples/` directory for complete usage examples:
- `basic_usage.py`: Getting started with signals and formulas

## Quick Start

### Creating Signals and Predicates

```python
import numpy as np
from pywstl import Signal

# Create signals: value must be array or tensor
# with shape (# of signals, # of time steps)
x = Signal("x", np.array([[1.0, 2.0, 3.0, 4.0]]))
y = Signal("y", np.array([[4.0, 3.0, 2.0, 1.0]]))

# Create predicates using comparison operators
phi1 = x >= 2.0  # GreaterThan predicate
phi2 = y <= 3.0  # LessThan predicate
```

### Building WSTL Formulas

The recommended way to build formulas is through the top-level unified API.
These classes automatically use whichever backend is currently active.
Default is numpy, but you can switch to torch with `pywstl.set_backend('torch')`.

```python
from pywstl import Always, Eventually, And, Or

phi_always    = Always(phi1, interval=[0, 3])      # G[0,3] (x >= 2)
phi_eventually = Eventually(phi2, interval=[1, 2]) # F[1,2] (y <= 3)
phi_and = And(phi1, phi2)                          # (x >= 2) ∧ (y <= 3)
phi_or  = Or(phi_always, phi_eventually)           # G[0,3](x >= 2) ∨ F[1,2](y <= 3)
```

### Computing Robustness

```python
# Set uniform weights
# (default is the standard STL robustness, i.e. all weights = 1)
phi_and.set_weights(signals=(x, y))

# Compute robustness at time t=0
# (time= 0 is the default time index)
robustness = phi_and.robustness((x, y)) #expected -1.0
print(f"STL Robustness: {robustness}")
```

### Setting Custom Weights

```python
import numpy as np

# Inspect available weight keys
print(list(phi_and.weights.keys()))
# e.g. ['(x>=2_0)and(y<=3_0)'], dots are replaced with underscores in keys

custom_weights = {
    "(x>=2_0)and(y<=3_0)": np.array([1.5, 1.0])
}

phi_and.set_weights_from_dict(custom_weights)
robustness = phi_and.robustness((x, y)) # expected -1.5
print(f"Robustness with custom weights: {robustness}")
```

### Weight Management

```python
# Uniform weights (default)
phi_and.set_weights((x,y), w_range=[1.0, 1.0])

# Random sampling with reproducible seed
phi_and.set_weights((x,y), w_range=[0.5, 2.0], no_samples=100, random=True, seed=42)

# Export all weights as a single 2-D array / tensor
weights_array = phi_and.get_weight_array()
```

### Robustness Computation

```python
# Scalar robustness at a specific time step
# expected shape (1,100) for 1 signal and 100 samples
rho = phi_and.robustness((x, y), t=0)

# Per-subformula robustness values (numpy backend only)
# dictionary with keys corresponding to subformulas.
# for weighted operators, i.e. and, the values are of shape (1,100, 1)
# for 1 signal, 100 samples, and 1 time step,
# for non-weighted operators, the values are of shape (1,1,1) for 1 signal and 1 samples,
# and 1 time step.
all_rho = phi_and.subrobustness((x, y), t=0)
```

---

## Backend Selection

### Checking and switching backends

```python
import pywstl

print(pywstl.get_backend())        # 'numpy'  (default)
print(pywstl.is_torch_available()) # True / False

pywstl.set_backend('torch')        # switch to pytorch
pywstl.reset_backend()             # reset to numpy (default)
```

### numpy backend (default)

```python
import numpy as np
from pywstl import Signal, Always, And

x = Signal("x", np.array([[1.0, 2.0, 3.0, 4.0]]))

formula = Always(x >= 2.0, interval=[0, 2])
formula.set_weights(x)
rho = formula.robustness(x, t=0)
print(f" Robustness value: {rho.item()}, data type: {type(rho)}")
```

### pytorch backend

```python
import pywstl
from pywstl import Signal
from pywstl import Always, GreaterThan

pywstl.set_backend('torch')

x = Signal("x", torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

formula = Always(x>=2.0, interval=[0, 2])
formula.set_weights(x)
rho = formula.robustness(x, t=0)
print(f" Robustness value: {rho.item()}, data type: {type(rho)}")
print(rho.grad_fn)  # supports autograd

pywstl.reset_backend()
```
---

## Module Overview

| Module | Description |
|---|---|
| `pywstl.backend` | Backend selection and unified factory API |
| `pywstl.intervals` | `Interval` class for finite and infinite temporal bounds |
| `pywstl.operations` | Core quantitative semantics operations for both backends |
| `pywstl.signals` | `Signal` class and predicate classes|
| `pywstl.utils` | Helper functions |
| `pywstl.validation` | Input validation utilities |
| `pywstl.wstlbase` | Base classes and mixins for WSTL formulas |
| `pywstl.wstlpy` | numpy-based WSTL formula classes |
| `pywstl.wstltorch` | pytorch-based WSTL formula classes |
---

## Citation

If you use Py(W)STL in your research, please cite:

```bibtex
@software{pywstl2025,
  title = {Py(W)STL: Python (Weighted) Signal Temporal Logic Library},
  author = {Karagulle, Ruya},
  year = {2025},
  url = {https://github.com/ruyakrgl/pywstl}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Developed at the University of Michigan. This library is part of ongoing research.

## Contact

- **Author**: Ruya Karagulle
- **Email**: ruyakrgl@umich.edu
- **GitHub**: [@ruyakrgl](https://github.com/ruyakrgl)
