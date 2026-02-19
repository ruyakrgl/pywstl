# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Py(W)STL: Python (Weighted) Signal Temporal Logic Library

A comprehensive library for constructing and evaluating
Weighted Signal Temporal Logic (WSTL) formulas.

Main modules:
    pywstl.backend:  Backend selection and unified factory API
    pywstl.intervals: Interval class for finite and infinite temporal bounds
    pywstl.operations: Core quantitative semantics operations for both backends
    pywstl.signals: `Signal` class and predicate classes
    pywstl.utils: Helper functions
    pywstl.validation: Input validation utilities
    pywstl.wstlbase: Base classes and mixins for WSTL formulas
    pywstl.wstlpy: numpy-based WSTL formula classes
    pywstl.wstltorch: pytorch-based WSTL formula classes
"""

# Version information
__version__ = "1.0.0"
__author__ = "Ruya Karagulle"
__license__ = "MIT"

# Import submodules - order matters to avoid circular imports
from . import intervals, operations, signals, validation, utils, wstlpy  # NOQA

# Import backend selection module
from . import backend  # NOQA

# Import backend control functions
from .backend import (  # NOQA
    set_backend,
    get_backend,
    reset_backend,
    is_torch_available,
    is_cuda_available,
)

# Import unified API (factory classes)
from .backend import (  # NOQA
    Always,
    Eventually,
    And,
    Or,
    Negation,
    LessThan,
    GreaterThan,
    Equal,
)

# Also keep direct Signal import from signals module
from .signals import Signal, Predicate  # NOQA

# Keep access to backend-specific implementations if needed
from .wstlpy import WSTLFormula as WSTLFormulaPy  # NOQA

# Try to import torch version if available
try:
    from . import wstltorch  # NOQA
    from .wstltorch import WSTLFormulaTorch  # NOQA

    _torch_available = True
except ImportError:
    wstltorch = None
    WSTLFormulaTorch = None
    _torch_available = False

# Define what's available when using "from pywstl import *"
__all__ = [
    # Core classes
    "Signal",
    "Predicate",
    # Backend control
    "set_backend",
    "get_backend",
    "reset_backend",
    "is_torch_available",
    "is_cuda_available",
    # Unified API (works with both backends)
    "Always",
    "Eventually",
    "And",
    "Or",
    "Negation",
    "LessThan",
    "GreaterThan",
    "Equal",
    # Backend-specific (for advanced use)
    "WSTLFormulaPy",
    "WSTLFormulaTorch",
    # Submodules
    "backend",
    "signals",
    "wstlpy",
    "wstltorch",
    "intervals",
    "operations",
    "validation",
    "utils",
    # Version info
    "__version__",
]
