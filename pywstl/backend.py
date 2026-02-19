# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
pywstl Backend Selection Module

This module provides backend selection functionality allowing users to choose
between numpy and pytorch implementations while maintaining a consistent API.

Usage:
    >>> import pywstl
    >>> pywstl.set_backend('torch')  # Use pytorch backend
    >>> # OR
    >>> pywstl.set_backend('numpy')  # Use numpy backend (default)
    >>>
    >>> # Then use the unified API
    >>> from pywstl import Always, And, Signal
    >>> formula = Always(signal >= 2.0, interval=[0, 10])

Author: Ruya Karagulle
Date: February 2026
"""

import logging

logger = logging.getLogger(__name__)

# Global backend state
_BACKEND = "numpy"  # Default backend
_TORCH_AVAILABLE = False

# Check if pytorch is available
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_backend(backend: str) -> None:
    """
    Set the computational backend for WSTL operations.

    Args:
        backend: Either 'numpy' or 'torch'

    Raises:
        ValueError: If backend is not 'numpy' or 'torch'
        RuntimeError: If 'torch' is selected but pytorch is not installed

    Example:
        >>> import pywstl
        >>> pywstl.set_backend('torch')
        >>> # All WSTL operations will now use pytorch
    """
    global _BACKEND

    backend = backend.lower()  # backend is case insensitive

    if backend not in ["numpy", "torch"]:
        raise ValueError(f"Backend must be 'numpy' or 'torch', got '{backend}'")

    if backend == "torch" and not _TORCH_AVAILABLE:
        raise RuntimeError(
            "pytorch backend selected but pytorch is not installed. "
            "Install pytorch with: pip install torch"
        )

    _BACKEND = backend
    logger.info(f"WSTL backend set to: {backend}")


def get_backend() -> str:
    """
    Get the currently active backend.

    Returns:
        Current backend name ('numpy' or 'torch')

    Example:
        >>> import pywstl
        >>> print(pywstl.get_backend())
        'numpy'
    """
    return _BACKEND


def is_torch_available() -> bool:
    """
    Check if pytorch backend is available.

    Returns:
        True if pytorch is installed, False otherwise
    """
    return _TORCH_AVAILABLE


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for pytorch backend.

    Returns:
        True if CUDA is available, False otherwise
    """
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def reset_backend() -> None:
    """
    Reset backend to default (numpy).

    Example:
        >>> import pywstl
        >>> pywstl.set_backend('torch')
        >>> pywstl.reset_backend()
        >>> print(pywstl.get_backend())
        'numpy'
    """
    global _BACKEND
    _BACKEND = "numpy"
    logger.info("WSTL backend reset to: numpy")


# ================= BACKEND-AWARE FACTORY FUNCTIONS =================
def _get_implementation_module():
    """
    Get the appropriate implementation module based on current backend.

    Returns:
        Either wstlpy or wstltorch module

    Raises:
        RuntimeError: If backend is invalid
    """
    if _BACKEND == "numpy":
        from . import wstlpy

        return wstlpy
    elif _BACKEND == "torch":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("pytorch backend not available. Install pytorch first.")
        from . import wstltorch

        return wstltorch
    else:
        raise RuntimeError(f"Invalid backend: {_BACKEND}")


# ================= FACTORY CLASSES =================
class Always:
    """
    Factory for Always operator (G).

    Automatically uses the currently selected backend.

    Args:
        subformula: Subformula for the Always operator
        interval: Time interval [a, b] or None for [0, ∞)

    Example:
        >>> import pywstl
        >>> spec = pywstl.Always(predicate, interval=[0, 10])
    """

    def __new__(cls, subformula, interval=None):
        module = _get_implementation_module()
        if _BACKEND == "numpy":
            return module.Always(subformula, interval)
        else:
            return module.AlwaysTorch(subformula, interval)


class Eventually:
    """
    Factory for Eventually operator (F).

    Automatically uses the currently selected backend.

    Args:
        subformula: Subformula for the Eventually operator
        interval: Time interval [a, b] or None for [0, ∞)

    Example:
        >>> import pywstl
        >>> spec = pywstl.Eventually(predicate, interval=[0, 5])
    """

    def __new__(cls, subformula, interval=None):
        module = _get_implementation_module()
        if _BACKEND == "numpy":
            return module.Eventually(subformula, interval)
        else:
            return module.EventuallyTorch(subformula, interval)


class And:
    """
    Factory for And operator (∧).

    Automatically uses the currently selected backend.

    Args:
        subformula1: First subformula
        subformula2: Second subformula

    Example:
        >>> import pywstl
        >>> spec = pywstl.And(pred1, pred2)
    """

    def __new__(cls, subformula1, subformula2):
        module = _get_implementation_module()
        if _BACKEND == "numpy":
            return module.And(subformula1, subformula2)
        else:
            return module.AndTorch(subformula1, subformula2)


class Or:
    """
    Factory for Or operator (∨).

    Automatically uses the currently selected backend.

    Args:
        subformula1: First subformula
        subformula2: Second subformula

    Example:
        >>> import pywstl
        >>> spec = pywstl.Or(pred1, pred2)
    """

    def __new__(cls, subformula1, subformula2):
        module = _get_implementation_module()
        if _BACKEND == "numpy":
            return module.Or(subformula1, subformula2)
        else:
            return module.OrTorch(subformula1, subformula2)


class Negation:
    """
    Factory for Negation operator (¬).

    Automatically uses the currently selected backend (numpy or pytorch).

    Args:
        subformula: Subformula to negate

    Example:
        >>> import pywstl
        >>> spec = pywstl.Negation(predicate)
    """

    def __new__(cls, subformula):
        module = _get_implementation_module()
        if _BACKEND == "numpy":
            return module.Negation(subformula)
        else:
            return module.NegationTorch(subformula)


# ================= PREDICATE FACTORIES =================
class LessThan:
    """
    Factory for LessThan predicate (<=).

    Automatically uses the currently selected backend.

    Args:
        signal: Signal object
        val: Threshold value

    Example:
        >>> from pywstl import Signal, LessThan
        >>> signal = Signal("speed", values)
        >>> pred = LessThan(signal, 30.0)
    """

    def __new__(cls, signal, val):
        from . import signals

        if _BACKEND == "numpy":
            return signals.LessThan(signal, val)
        else:
            module = _get_implementation_module()
            return module.LessThanTorch(signal, val)


class GreaterThan:
    """
    Factory for GreaterThan predicate (>=).

    Automatically uses the currently selected backend.

    Args:
        signal: Signal object
        val: Threshold value

    Example:
        >>> from pywstl import Signal, GreaterThan
        >>> signal = Signal("distance", values)
        >>> pred = GreaterThan(signal, 2.0)
    """

    def __new__(cls, signal, val):
        from . import signals

        if _BACKEND == "numpy":
            return signals.GreaterThan(signal, val)
        else:
            module = _get_implementation_module()
            return module.GreaterThanTorch(signal, val)


class Equal:
    """
    Factory for Equal predicate (==).

    Automatically uses the currently selected backend.

    Args:
        signal: Signal object
        val: Target value

    Example:
        >>> from pywstl import Signal, Equal
        >>> signal = Signal("position", values)
        >>> pred = Equal(signal, 0.0)
    """

    def __new__(cls, signal, val):
        from . import signals

        if _BACKEND == "numpy":
            return signals.Equal(signal, val)
        else:
            module = _get_implementation_module()
            return module.EqualTorch(signal, val)
