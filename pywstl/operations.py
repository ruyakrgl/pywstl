# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL Operations Module

This module provides core operations for WSTL computations.

Author: Ruya Karagulle
Last Update: February 2026
"""
import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

LARGE_NUMBER = 10**6


class wstlMax:
    """Compute maximum values along specified axis for WSTL operations."""

    def __call__(self, x: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute maximum values along the specified axis.

        Args:
            x: Input numpy array
            axis: Axis along which to compute maximum

        Returns:
            Array with maximum values, keeping dimensions

        Raises:
            AssertionError: If input is not a numpy array
        """
        assert isinstance(x, np.ndarray), f"Input should be a numpy array but got {type(x)}."
        return np.max(x, axis=axis, keepdims=True)


class wstlMin:
    """Compute minimum values along specified axis for WSTL operations."""

    def __call__(self, x: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute minimum values along the specified axis.

        Args:
            x: Input numpy array
            axis: Axis along which to compute minimum

        Returns:
            Array with minimum values, keeping dimensions

        Raises:
            AssertionError: If input is not a numpy array
        """
        assert isinstance(x, np.ndarray), f"Input should be a numpy array but got {type(x)}."
        return np.min(x, axis=axis, keepdims=True)


# ================= TORCH OPERATIONS =================
class wstlMaxTorch:
    """Compute maximum values along specified axis using pytorch."""

    def __call__(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Compute maximum values along the specified axis.

        Args:
            x: Input torch tensor
            axis: Axis along which to compute maximum

        Returns:
            Tensor with maximum values, keeping dimensions
        """
        assert isinstance(x, torch.Tensor), f"Input should be a torch tensor but got {type(x)}."
        return torch.max(x, dim=axis, keepdim=True)[0]


class wstlMinTorch:
    """Compute minimum values along specified axis using pytorch."""

    def __call__(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Compute minimum values along the specified axis.

        Args:
            x: Input torch tensor
            axis: Axis along which to compute minimum

        Returns:
            Tensor with minimum values, keeping dimensions
        """
        assert isinstance(x, torch.Tensor), f"Input should be a torch tensor but got {type(x)}."
        return torch.min(x, dim=axis, keepdim=True)[0]
