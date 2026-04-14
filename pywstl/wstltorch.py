# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL pytorch Programming Module

This module implements the quantitative semantics computation for WSTL formulas
with pytorch library. It provides classes that inherit from pywstl base classes
but use pytorch tensors for computation.

Classes:
    WSTLFormulaTorch: Torch version of WSTLFormula base class
    UnaryOperatorTorch: Torch version for unary temporal operators (Always, Eventually)
    BinaryOperatorTorch: Torch version for binary logical operators (And, Or)
    AlwaysTorch: Temporal "always" operator
    EventuallyTorch: Temporal "eventually" operator
    AndTorch: Logical conjunction operator
    OrTorch: Logical disjunction operator
    NegationTorch: Logical negation operator

Author: Ruya Karagulle
Date: February 2026
"""
import logging
from typing import Any, Optional, List

import numpy as np

# Import base classes from pywstl
from pywstl import intervals, operations, wstlbase
from pywstl.utils import get_signal_shape
from pywstl.validation import Validator

logger = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    # Create a dummy module for type hints
    class _TorchDummy:
        nn = Any
        nn.Module = Any
        Tensor = Any

    torch = _TorchDummy()  # type: ignore


# ================= BASE FORMULA CLASS (TORCH VERSION) =================
class WSTLFormulaTorch(wstlbase.WSTLKeyMixin, torch.nn.Module):
    """
    Base class for all WSTL formula objects using pytorch.

    This class wraps the pywstl.WSTLFormula interface but uses pytorch tensors
    for computation instead of NumPy arrays.

    Attributes:
        keys: List of names for the weights dictionary
        weights: Dictionary storing weight values (as torch tensors)
        key: String identifier for this formula
    """

    def __init__(self, *args, **kwargs):
        """Initialize a WSTLFormulaTorch instance."""
        super(WSTLFormulaTorch, self).__init__()
        self.weights = {}
        self.keys = []
        self.key_hashes = []
        self.key = None
        self.key_map = [self.keys, self.key_hashes]
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        """Move the formula to a device (e.g. 'cuda', 'cpu').

        Extends ``torch.nn.Module.to()``.
        Submodule subformulas are handled automatically by the base
        class.

        Args:
            *args / **kwargs: Same arguments as ``torch.nn.Module.to()``.

        Returns:
            self
        """
        result = super().to(*args, **kwargs)

        # Resolve target device once
        try:
            if args and isinstance(args[0], (str, torch.device)):
                new_device = torch.device(args[0])
            elif "device" in kwargs:
                new_device = torch.device(kwargs["device"])
        except Exception:
            new_device = None

        # We walk the full module tree to update both the
        # plain-dict weights and the device attribute on every WSTLFormulaTorch.
        for module in self.modules():
            if isinstance(module, WSTLFormulaTorch):
                module.weights = {k: v.to(*args, **kwargs) for k, v in module.weights.items()}
                if new_device is not None:
                    module.device = new_device

        return result

    def requires_grad(self, requires_grad: bool = True) -> "WSTLFormulaTorch":
        """Enable or disable gradient tracking on all weights in the formula tree.

        Args:
            requires_grad: If True, weights will track gradients; if False, they won't.

        Returns:
            self, to allow chaining (e.g. ``formula.to('cuda').requires_grad(True)``)
        """
        for module in self.modules():
            if isinstance(module, WSTLFormulaTorch):
                for key, w in module.weights.items():
                    module.weights[key] = w.requires_grad_(requires_grad)
        return self

    # ================= WEIGHT MANAGEMENT =================
    def set_weights(
        self,
        signals,
        w_range: list = [1.0, 1.0],
        no_samples: int = 1,
        random: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Set weights for the formula using pytorch tensors.

        Args:
            signals: Input signals (Signal or tuple of Signals)
            w_range: Range [min, max] for weight initialization
            no_samples: Number of weight samples
            random: If True, randomly initialize weights
            seed: Random seed for reproducibility
        """
        # Validate inputs (using pywstl validator)
        Validator._validate_signal_input(signals)
        Validator._validate_weight_range(w_range)
        Validator._validate_no_samples(no_samples)
        Validator._validate_random_flag(random)

        if seed is not None:
            torch.manual_seed(seed)
            if random is False:
                logger.info("Seed only makes sense when 'random' is True.")

        if random and w_range == [1, 1]:
            logger.warning("Random weights requested but range is [1, 1]. All weights are 1.")

        # Set weights using torch
        self._set_weights(signals, w_range, no_samples, random)

    def _set_weights(self, signals, w_range, no_samples, random) -> None:
        """Assign weight values. Override in subclasses."""
        pass

    def set_weights_from_dict(self, weights_dict: dict) -> None:
        """
        Set weights from a dictionary (torch tensors or numpy arrays).

        Args:
            weights_dict: Dictionary mapping keys to weight tensors/arrays
        """
        for key, value in weights_dict.items():
            if isinstance(value, np.ndarray):
                self.weights[key] = torch.from_numpy(value).float().to(self.device)
            elif isinstance(value, torch.Tensor):
                self.weights[key] = value.clone().detach().to(self.device)
            else:
                raise TypeError(
                    f"Weight values must be numpy arrays or torch tensors, got {type(value)}"
                )

    def get_weight_array(self) -> torch.Tensor:
        """
        Convert weights dictionary to a 2D torch tensor.

        Returns:
            Tensor of shape (num_weights, num_samples)
        """
        if not self.weights or len(self.weights) == 0:
            return torch.empty(0, 0)

        weight_list = []
        for key in self.weights.keys():
            w = self.weights[key].reshape(-1, 1)
            weight_list.append(w)

        return torch.cat(weight_list, dim=0)

    # ================= ROBUSTNESS COMPUTATION =================
    def robustness(self, signals, t: int = 0) -> torch.Tensor:
        """
        Return weighted robustness value at time t.

        Args:
            signals: Input signals
            t: Time instance

        Returns:
            Torch tensor with robustness values
        """
        # Validate
        _, time_length = get_signal_shape(signals)
        Validator._validate_signal_input(signals)
        Validator._validate_time_parameter(t, time_length)

        # Compute robustness
        robustness = self._robustness(signals)[:, t, :, :]
        return robustness.squeeze(-1)

    def _robustness(self, signals) -> torch.Tensor:
        """Compute robustness trace. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _robustness")

    def _convert_signals_to_torch(self, signals) -> torch.Tensor:
        """
        Convert pywstl Signal objects to torch tensors.

        Args:
            signals: Signal object or tuple of Signal objects

        Returns:
            Torch tensor representation
        """
        if isinstance(signals, tuple):
            return tuple(self._convert_signals_to_torch(s) for s in signals)
        elif isinstance(signals, torch.Tensor):
            return signals
        elif isinstance(signals, np.ndarray):
            return torch.from_numpy(signals).float()
        else:
            raise TypeError(f"Invalid signal type: {type(signals)}")

    def forward(self, signals, **kwargs):
        """
        Forward pass for torch.nn.Module compatibility.

        Args:
            signals: Input signals
            **kwargs: Additional arguments

        Returns:
            Robustness tensor
        """
        return self._robustness(signals)


# ================= UNARY OPERATOR (TORCH VERSION) =================
class UnaryOperatorTorch(wstlbase.UnaryMixin, WSTLFormulaTorch):
    """
    Base class for temporal operators using pytorch.

    Attributes:
        subformula: Subformula encapsulated by the operator
        interval: Time interval for the operator
        operation: Min or Max operation (torch version)
    """

    def __init__(self, subformula: WSTLFormulaTorch, interval: Optional[List[int]] = None):
        """
        Initialize a UnaryOperatorTorch.

        Args:
            subformula: Subformula (must be WSTLFormulaTorch)
            interval: Time interval [a, b] or None for [0, inf)
        """
        if not isinstance(subformula, WSTLFormulaTorch):
            raise TypeError(f"subformula must be WSTLFormulaTorch, got {type(subformula)}")

        # Must call super().__init__() BEFORE setting attributes (torch.nn.Module requirement)
        super().__init__()
        self.interval = intervals.Interval(interval)
        self.subformula = subformula
        # set_keys() is NOT called here; each concrete subclass calls it after
        # setting operation_name, so that _set_operator_key() has all it needs.

    def _set_weights(self, signals, w_range, no_samples, random) -> None:
        """Assign weights to subformula and operator."""
        # Set subformula weights first
        self.subformula._set_weights(signals, w_range, no_samples, random)

        # Get robustness trace to set interval
        trace = self.subformula._robustness(signals)
        self.interval.set_interval(trace.shape[1])

        # Copy subformula weights
        for key in self.subformula.weights.keys():
            self.weights[key] = self.subformula.weights[key]

        # Set operator-specific weights
        self._set_operator_weights(w_range, no_samples, random)

    def _set_operator_weights(self, w_range, no_samples, random):
        """Assign weight values for this operator using torch."""
        interval_length = self.interval.value[1] - self.interval.value[0] + 1

        if random:
            self.weights[self.key] = w_range[0] + (w_range[1] - w_range[0]) * torch.rand(
                interval_length, no_samples, dtype=torch.float32, device=self.device
            )
        else:
            self.weights[self.key] = torch.ones(
                interval_length, no_samples, dtype=torch.float32, device=self.device
            )

    def _robustness(self, signals) -> torch.Tensor:
        """Compute weighted robustness of temporal operator."""
        _, time_length = get_signal_shape(signals)
        self.interval.set_interval(time_length)

        # Compute subformula robustness
        trace = self.subformula._robustness(signals)

        # Compute operator robustness
        outputs = self._operator_robustness(trace)
        return outputs

    def _operator_robustness(self, trace: torch.Tensor) -> torch.Tensor:
        """Compute weighted robustness using torch operations."""
        assert self.operation is not None, "Operation is not defined."

        w = self.weights[self.key]
        _output = None

        for i in range(trace.shape[1] - self.interval.value[0]):
            trace_lb = i + self.interval.value[0]
            trace_ub = min(i + self.interval.value[1], trace.shape[1] - 1)

            w_current = w[: min(w.shape[0], trace_ub - trace_lb) + 1, :]
            trace_current = trace[:, trace_lb : trace_ub + 1, :, 0]

            operated = self.operation(w_current * trace_current, axis=1)[:, :, :, None]

            if _output is None:
                _output = operated
            else:
                _output = torch.cat((_output, operated), dim=1)

        return _output


# ================= BINARY OPERATOR (TORCH VERSION) =================
class BinaryOperatorTorch(wstlbase.BinaryMixin, WSTLFormulaTorch):
    """
    Base class for binary logical operators using pytorch.

    Attributes:
        subformula1: First subformula
        subformula2: Second subformula
        operation: Min or Max operation (torch version)
    """

    def __init__(self, subformula1: WSTLFormulaTorch, subformula2: WSTLFormulaTorch):
        """
        Initialize a BinaryOperatorTorch.

        Args:
            subformula1: First subformula (must be WSTLFormulaTorch)
            subformula2: Second subformula (must be WSTLFormulaTorch)
        """
        if not isinstance(subformula1, WSTLFormulaTorch):
            raise TypeError(f"subformula1 must be WSTLFormulaTorch, got {type(subformula1)}")
        if not isinstance(subformula2, WSTLFormulaTorch):
            raise TypeError(f"subformula2 must be WSTLFormulaTorch, got {type(subformula2)}")

        # Must call super().__init__() BEFORE setting attributes
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        # set_keys() is NOT called here; each concrete subclass calls it after
        # setting operation_name, so that _set_operator_key() has all it needs.

    def _set_weights(self, signals, w_range, no_samples, random) -> None:
        """Assign weights to subformulas and operator."""
        # Signals should be a tuple for binary operators
        if not isinstance(signals, tuple) or len(signals) != 2:
            raise ValueError("Binary operators require a tuple of 2 signal inputs")

        # Set subformula weights
        self.subformula1._set_weights(signals[0], w_range, no_samples, random)
        self.subformula2._set_weights(signals[1], w_range, no_samples, random)

        # Copy subformula weights
        for key in self.subformula1.weights.keys():
            self.weights[key] = self.subformula1.weights[key]
        for key in self.subformula2.weights.keys():
            self.weights[key] = self.subformula2.weights[key]

        # Set operator-specific weights
        self._set_operator_weights(w_range, no_samples, random)

    def _set_operator_weights(self, w_range, no_samples, random):
        """Assign weight values for this operator using torch."""
        if random:
            self.weights[self.key] = w_range[0] + (w_range[1] - w_range[0]) * torch.rand(
                2, no_samples, dtype=torch.float32, device=self.device
            )
        else:
            self.weights[self.key] = torch.ones(
                2, no_samples, dtype=torch.float32, device=self.device
            )

    def _robustness(self, signals) -> torch.Tensor:
        """Compute weighted robustness of binary operator."""
        if not isinstance(signals, tuple) or len(signals) != 2:
            raise ValueError("Binary operators require a tuple of 2 signal inputs")

        # Compute subformula robustness
        trace1 = self.subformula1._robustness(signals[0])
        trace2 = self.subformula2._robustness(signals[1])

        trace_length = min(trace1.shape[1], trace2.shape[1])

        # Handle different weight sample counts
        if trace1.shape[2] != trace2.shape[2]:
            if trace1.shape[2] < trace2.shape[2]:
                trace1 = trace1.repeat(1, 1, trace2.shape[2], 1)
            else:
                trace2 = trace2.repeat(1, 1, trace1.shape[2], 1)

        trace = torch.cat([trace1[:, :trace_length, :], trace2[:, :trace_length, :]], dim=-1)
        return self._operator_robustness(trace)

    def _operator_robustness(self, trace: torch.Tensor) -> torch.Tensor:
        """Compute weighted robustness using torch operations."""
        assert self.operation is not None, "Operation is not defined."

        w = self.weights[self.key]
        _output = None

        for i in range(trace.shape[1]):
            operated = self.operation(w.T * trace[:, i, :, :].unsqueeze(1), axis=-1)

            if _output is None:
                _output = operated
            else:
                _output = torch.cat((_output, operated), dim=1)

        return _output


# ================= SPECIFIC OPERATORS (TORCH VERSIONS) =================
class AlwaysTorch(UnaryOperatorTorch):
    """Temporal always operator: G[a,b] φ"""

    def __init__(self, subformula: WSTLFormulaTorch, interval: Optional[List[int]] = None):
        super().__init__(subformula, interval)
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    @property
    def operation_name(self):
        return "G"

    @property
    def operation(self):
        return operations.wstlMinTorch()


class EventuallyTorch(UnaryOperatorTorch):
    """Temporal eventually operator: F[a,b] φ"""

    def __init__(self, subformula: WSTLFormulaTorch, interval: Optional[List[int]] = None):
        super().__init__(subformula, interval)
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    @property
    def operation_name(self):
        return "F"

    @property
    def operation(self):
        return operations.wstlMaxTorch()


class AndTorch(BinaryOperatorTorch):
    """Logical conjunction: phi1 ∧ phi2"""

    def __init__(self, subformula1: WSTLFormulaTorch, subformula2: WSTLFormulaTorch):
        super().__init__(subformula1, subformula2)
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    @property
    def operation_name(self):
        return "and"

    @property
    def operation(self):
        return operations.wstlMinTorch()


class OrTorch(BinaryOperatorTorch):
    """Logical disjunction: φ1 ∨ φ2"""

    def __init__(self, subformula1: WSTLFormulaTorch, subformula2: WSTLFormulaTorch):
        super().__init__(subformula1, subformula2)
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    @property
    def operation_name(self):
        return "or"

    @property
    def operation(self):
        return operations.wstlMaxTorch()


class NegationTorch(WSTLFormulaTorch):
    """Logical negation: ¬phi"""

    def __init__(self, subformula: WSTLFormulaTorch):
        if not isinstance(subformula, WSTLFormulaTorch):
            raise TypeError(f"subformula must be WSTLFormulaTorch, got {type(subformula)}")

        # Must call super().__init__() BEFORE setting attributes
        super().__init__()
        self.subformula = subformula
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]

    def _set_key(self, key_list: list) -> list:
        """Set dictionary names for negation and subformula."""
        key_list = self.subformula._set_key(key_list)
        key = f"not({self.subformula.key})"

        while key in key_list:
            key += "_"

        self.key = key
        key_list.append(self.key)
        return key_list

    def _set_weights(self, signals, w_range, no_samples, random) -> None:
        """Negation has no weights, only subformula weights."""
        self.subformula._set_weights(signals, w_range, no_samples, random)

        # Copy subformula weights
        for key in self.subformula.weights.keys():
            self.weights[key] = self.subformula.weights[key]

    def _robustness(self, signals) -> torch.Tensor:
        """Compute negation robustness."""
        return -self.subformula._robustness(signals)

    def __str__(self):
        return f"¬({self.subformula})"
