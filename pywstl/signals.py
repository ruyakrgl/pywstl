# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Signal and Predicate Classes for WSTL

This module provides the Signal class for representing time-series data, and
the Predicate class for creating atomic predicates in WSTL formulas.

Classes:
    Signal: Represents multidimensional time-series signals with support for
            arithmetic operations and numpy/pytorch integration
    Predicate: Base class for atomic predicates in WSTL formulas
    LessThan: Predicate for <= comparisons
    GreaterThan: Predicate for >= comparisons
    Equal: Predicate for == comparisons

The Signal class supports standard arithmetic operations (+, -, *, /) and can be
used with numpy universal functions. When torch backend is selected, it also accepts
torch tensors and converts them to numpy for storage.

Author: Ruya Karagulle
Last Update: February 2026
"""
import numpy as np

from pywstl import backend as backend_module
from pywstl import wstlpy, wstlbase, wstltorch
from pywstl.validation import Validator

# Try to import torch for backend-aware Signal
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ================= SIGNAL CLASS =====================
class Signal:
    """
    Wrapper class for arithmetic operations on signals.

    Attributes:
        name (str): The name associated with the Signal.
        value (np.ndarray): The numeric value of the Signal.

    Methods:
        set_name(new_name): Updates the name attribute of the Signal.
        set_value(new_value): Updates the value attribute of the Signal.
    """

    def __init__(self, name: str, value):
        """
        Initialize a Signal instance.

        Args:
            name (str): The name associated with the Signal.
            value (np.ndarray or torch.Tensor): The numeric value of the Signal.
                  Type depends on current backend setting.

        Raises:
            TypeError: If name is not a string or value type doesn't match backend.
        """
        from pywstl import backend as backend_module

        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}.")

        # Check backend and validate value type
        current_backend = backend_module.get_backend()

        if current_backend == "torch":
            # Torch backend: accept torch tensors or convert numpy to torch
            if _TORCH_AVAILABLE:
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                elif not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Value must be torch.Tensor or numpy array, got {type(value)}."
                    )
            else:
                raise RuntimeError("Torch backend selected but pytorch not available")
        else:
            # numpy backend: accept numpy arrays or convert torch to numpy
            if isinstance(value, np.ndarray):
                pass  # Already numpy
            elif _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                raise TypeError(f"Value must be numpy array or torch tensor, got {type(value)}.")

        self.name = name
        self.value = value
        if value.ndim == 2:
            self._value = value.reshape(value.shape[0], value.shape[1], 1, 1)
        else:
            self._value = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy universal functions to route them to Signal operations.

        This allows numpy arrays to work seamlessly with Signal objects:
        - np.add(array, signal) -> signal.__radd__(array)
        - np.subtract(array, signal) -> signal.__rsub__(array)
        - np.multiply(array, signal) -> signal.__rmul__(array)
        """
        # Only handle __call__ method for ufuncs
        if method != "__call__":
            return NotImplemented

        # Map numpy ufuncs to Signal methods
        ufunc_map = {
            np.add: "__add__",
            np.subtract: "__sub__",
            np.multiply: "__mul__",
            np.divide: "__truediv__",
            np.true_divide: "__truediv__",
        }

        if ufunc not in ufunc_map:
            return NotImplemented

        # Determine which input is the Signal object
        signal_idx = None
        other_input = None

        for i, inp in enumerate(inputs):
            if isinstance(inp, Signal):
                signal_idx = i
            else:
                other_input = inp

        if signal_idx is None:
            return NotImplemented

        signal_obj = inputs[signal_idx]
        method_name = ufunc_map[ufunc]

        # Handle order of operations
        if signal_idx == 0:
            # signal op other: use normal method
            method = getattr(signal_obj, method_name)
            return method(other_input)
        else:
            # other op signal: use reverse method
            reverse_method_name = "__r" + method_name[2:]  # __add__ -> __radd__
            if hasattr(signal_obj, reverse_method_name):
                method = getattr(signal_obj, reverse_method_name)
                return method(other_input)
            else:
                return NotImplemented

    def set_name(self, new_name: str) -> None:
        """
        Update the name attribute of the Signal.

        Args:
            new_name (str): The new name to be assigned to the Signal.

        Raises:
            TypeError: If new_name is not a string.
        """
        if not isinstance(new_name, str):
            raise TypeError(f"New name must be a string, got {type(new_name)}.")

        self.name = new_name

    def set_value(self, new_value) -> None:
        """
        Update the value attribute of the Signal.

        Args:
            new_value (np.ndarray or torch.Tensor): The new numeric value to be assigned.
                      Type depends on current backend.

        Raises:
            TypeError: If new_value type doesn't match backend.
        """
        from pywstl import backend as backend_module

        current_backend = backend_module.get_backend()

        if current_backend == "torch":
            # Torch backend: accept torch tensors or convert numpy to torch
            if _TORCH_AVAILABLE:
                if isinstance(new_value, np.ndarray):
                    new_value = torch.from_numpy(new_value).float()
                elif not isinstance(new_value, torch.Tensor):
                    raise TypeError(
                        f"New value must be torch.Tensor or numpy array, got {type(new_value)}."
                    )
            else:
                raise RuntimeError("Torch backend selected but pytorch not available")
        else:
            # numpy backend: accept numpy arrays or convert torch to numpy
            if isinstance(new_value, np.ndarray):
                pass  # Already numpy
            elif _TORCH_AVAILABLE and isinstance(new_value, torch.Tensor):
                new_value = new_value.detach().cpu().numpy()
            else:
                raise TypeError(
                    f"New value must be numpy array or torch tensor, got {type(new_value)}."
                )

        self.value = new_value
        if new_value.ndim == 2:
            self._value = new_value.reshape(new_value.shape[0], new_value.shape[1], 1, 1)
        else:
            self._value = new_value

    # ============ SIGNAL OPERATIONS ============
    def __neg__(self):
        """Return negated Signal."""
        return Signal(name=f"-{self.name}", value=-self.value)

    def __add__(self, other):
        """Add Signal with another Signal, array, or tensor."""
        if isinstance(other, Signal):
            return Signal(name=self.name + "+" + other.name, value=self.value + other.value)
        elif isinstance(other, (np.ndarray, torch.Tensor if _TORCH_AVAILABLE else type(None))):
            if self.value.shape != other.shape:
                raise ValueError("Incompatible shapes for addition")
            else:
                return Signal(name=self.name + "+", value=self.value + other)
        else:
            raise TypeError("Invalid input for addition")

    def __radd__(self, other):
        """Right addition operator."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract another Signal, array, or tensor from this Signal."""
        if isinstance(other, Signal):
            return Signal(name=self.name + "-" + other.name, value=self.value - other.value)
        elif isinstance(other, (np.ndarray, torch.Tensor if _TORCH_AVAILABLE else type(None))):
            if self.value.shape != other.shape:
                raise ValueError("Incompatible shapes for subtraction")
            else:
                return Signal(name=self.name + "-", value=self.value - other)
        else:
            raise TypeError("Invalid input for subtraction")

    def __rsub__(self, other):
        """Right subtraction operator: other - self."""
        if isinstance(other, Signal):
            return Signal(name=self.name + "-" + other.name, value=other.value - self.value)
        elif isinstance(other, (np.ndarray, torch.Tensor if _TORCH_AVAILABLE else type(None))):
            if self.value.shape != other.shape:
                raise ValueError("Incompatible shapes for subtraction")
            else:
                return Signal(name=self.name + "-", value=other - self.value)
        else:
            raise TypeError("Invalid input for subtraction")

    def __mul__(self, other):
        """Multiply Signal with another Signal or value."""
        if isinstance(other, Signal):
            return Signal(name=self.name + "*" + other.name, value=self.value * other.value)
        else:
            return Signal(name=self.name + "*", value=self.value * other)

    def __rmul__(self, other):
        """Right multiplication operator."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide Signal by another Signal or value."""
        numerator = self
        denominator = other
        num_name = "num"
        denom_name = str(other)
        if isinstance(numerator, Signal):
            num_name = numerator.name
            numerator = numerator.value
        if isinstance(denominator, Signal):
            denom_name = denominator.name
            denominator = denominator.value
        return Signal(name=num_name + "/" + denom_name, value=numerator / denominator)

    # ============ COMPARISON OPERATORS ============
    def __le__(self, rhs: float) -> "LessThan":
        """Create less than or equal predicate."""
        assert isinstance(
            rhs, (float, int)
        ), f"Right hand side must be float or int, got {type(rhs)}."
        if _TORCH_AVAILABLE and backend_module.get_backend() == "torch":
            return LessThanTorch(self, rhs)
        else:
            return LessThan(self, rhs)

    def __ge__(self, rhs: float) -> "GreaterThan":
        """Create greater than or equal predicate."""
        assert isinstance(
            rhs, (float, int)
        ), f"Right hand side must be float or int, got {type(rhs)}."
        if _TORCH_AVAILABLE and backend_module.get_backend() == "torch":
            return GreaterThanTorch(self, rhs)
        else:
            return GreaterThan(self, rhs)

    def __eq__(self, rhs: float) -> "Equal":
        """Create equality predicate."""
        assert isinstance(
            rhs, (float, int)
        ), f"Right hand side must be float or int, got {type(rhs)}."
        if _TORCH_AVAILABLE and backend_module.get_backend() == "torch":
            return EqualTorch(self, rhs)
        else:
            return Equal(self, rhs)

    def __str__(self) -> str:
        """Return string representation of Signal."""
        return str(self.name)

    def __repr__(self) -> str:
        """Return string representation of Signal."""
        return self.__str__()


class Predicate(wstlbase.PredicateMixin, wstlpy.WSTLFormula):
    """Define predicates."""

    def __init__(self, signals: Signal, value: float = 0):
        """
        Initialize a Predicate.

        Args:
            signals (Signal): Left-hand side Signal of the predicate.
            value (float): Right-hand side value of the predicate.
        """
        Validator._validate_predicate_threshold(value)
        Validator._validate_signal_input(signals)
        self.signals = signals
        self.value = value

        super().__init__()

    def _set_weights(self, signals, w_range, no_samples, random):
        """
        Assigns weight values to the predicate given a range.

        As there is no weight in predicates, this function returns nothing.
        """

    def _set_weights_from_dict(self, weights_dict):
        """
        Set weights from a dictionary for the predicate.

        Parameters:
            weights_dict (dict): Dictionary containing weight names and values.

        As there is no weight in predicates, this function returns nothing.
        """

    def _robustness(self, signals):
        """
        Computes weighted robustness of the predicate.

        Parameters:
            signals (Signal): Input signals.

        Returns:
            np.ndarray: Weighted robustness values.
        """
        return self._operator_robustness(signals._value)

    def _subrobustness(self, signals, t, all_robustness):
        outputs = self._robustness(signals)
        all_robustness[(self.key, t)] = outputs[:, t, :, :]
        return all_robustness


class LessThan(wstlbase.LessThanMixin, Predicate):
    """
    Defines the predicate with less than or equal to comparison:
    "lhs <= val" where lhs is the signal, and val is the constant.
    """

    def __init__(self, signals: Signal, value: float):
        super().__init__(signals=signals, value=value)


class GreaterThan(wstlbase.GreaterThanMixin, Predicate):
    """
    Defines the predicate with greater than or equal to comparison:
    "lhs >= val" where lhs is the signal, and val is the constant.
    """

    def __init__(self, signals: Signal, value: float):
        super().__init__(signals=signals, value=value)


class Equal(wstlbase.EqualMixin, Predicate):
    """
    Defines the predicate with equality comparison:
    "lhs == val" where lhs is the signal, and val is the constant.
    """

    def __init__(self, signals: Signal, value: float):
        super().__init__(signals=signals, value=value)


# ================= PREDICATE CLASSES (TORCH VERSIONS) =================
class PredicateTorch(wstlbase.PredicateMixin, wstltorch.WSTLFormulaTorch):
    """Base class for predicates using torch."""

    def __init__(self, signal: Signal, val: float = 0):
        """
        Initialize a predicate.

        Args:
            signal: Signal object
            val: Constant value for comparison
        """
        if not isinstance(signal, Signal):
            raise TypeError("signal must be a Signal object")

        # Must call super().__init__() BEFORE setting attributes
        super().__init__()
        self.signal = signal
        self.value = val  # stored as .value to match PredicateMixin convention
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    @property
    def val(self):
        """Alias for .value kept for API compatibility."""
        return self.value

    def __str__(self):
        return f"{self.signal.name}{self.operation_name}{self.value}"

    def _set_weights(self, signals, w_range, no_samples, random) -> None:
        """Predicates have no weights."""
        pass

    def _set_weights_from_dict(self, weights_dict):
        """Predicates have no weights; this is a no-op."""

    def _robustness(self, signals) -> torch.Tensor:
        """Compute predicate robustness."""
        if isinstance(signals, Signal):
            if isinstance(signals.value, torch.Tensor):
                trace = signals.value
            else:
                trace = torch.from_numpy(signals.value).float()
        elif isinstance(signals, torch.Tensor):
            trace = signals
        elif isinstance(signals, np.ndarray):
            trace = torch.from_numpy(signals).float()
        else:
            raise TypeError(f"Invalid signal type: {type(signals)}")

        # Ensure shape (batch, time, weight_samples, 1)
        if trace.dim() == 2:
            trace = trace.unsqueeze(-1).unsqueeze(-1)
        elif trace.dim() == 3:
            trace = trace.unsqueeze(-1)

        trace = trace.to(self.device)
        return self._operator_robustness(trace)


class LessThanTorch(wstlbase.LessThanMixin, PredicateTorch):
    """Predicate: signal <= val"""

    def __init__(self, signal: Signal, val: float):
        super().__init__(signal, val)


class GreaterThanTorch(wstlbase.GreaterThanMixin, PredicateTorch):
    """Predicate: signal >= val"""

    def __init__(self, signal: Signal, val: float):
        super().__init__(signal, val)


class EqualTorch(wstlbase.EqualMixin, PredicateTorch):
    """Predicate: signal == val"""

    def __init__(self, signal: Signal, val: float):
        super().__init__(signal, val)
