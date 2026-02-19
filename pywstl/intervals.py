# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Interval Module for Temporal Logic Operations

This module provides the Interval class for defining time intervals
used in WSTL formulas.

Classes:
    Interval: Represents time intervals with support for finite and infinite bounds

Examples:
    >>> # Finite interval
    >>> interval = Interval([0, 10])
    >>> print(interval)  # [0,10]
    >>> print(len(interval))  # 11

    >>> # Infinite interval
    >>> interval = Interval([5, np.inf])
    >>> interval.set_interval(100)  # Signal length = 100
    >>> print(interval.value)  # (5, 99)
    >>> print(interval)  # [5,∞]
    >>> print(len(interval))  # 95

Author: Ruya Karagulle
Date: September 2025
"""
import numpy as np


class Interval:
    """
    Define intervals for temporal operators.

    Attributes:
        interval: The interval specified during initialization.
        value: The computed interval values based on the input length.

    Methods:
        set_interval: Sets the interval for robustness computation.
    """

    def __init__(self, interval: tuple[int, int] | list[int, int] | None = None):
        """
        Initialize an instance of the Interval class.

        Args:
            interval: A tuple or list specifying the interval boundaries.

        Raises:
            ValueError: If the interval is invalid or start > end.

        Examples:
            >>> Interval((0, 5)) # Valid finite interval
            >>> Interval([2, np.inf]) # Infinite interval, requires signal_length for set_interval
            >>> Interval() # No interval specified
        """
        if interval is not None:
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                if interval[0] > interval[1]:
                    raise ValueError("Interval start must be less than or equal to end.")
                self.interval = tuple(interval)  # Convert to tuple for consistency
            else:
                raise ValueError(
                    f"Invalid interval format. Expected None or tuple/list of length 2, "
                    f"got {type(interval)}."
                )
        else:
            self.interval = None

    def set_interval(self, input_length: int | None = None):
        """
        Set the interval for robustness computation.

        If interval goes to infinity, we need the input_length
        info to compute the interval.

        Args:
            input_length: The length of the input signal.

        Raises:
            ValueError: If the interval is invalid or input_length is required but not provided.
        """
        if self.interval is None:
            if input_length is None:
                raise ValueError("Input length is required.")
            val = (0, input_length - 1)

        elif self.interval[1] == np.inf:
            if input_length is None:
                raise ValueError("Input length is required.")
            val = (self.interval[0], input_length - 1)

        elif len(self.interval) == 2:
            val = self.interval

        else:
            raise ValueError("Invalid interval format.")

        self.value = val

    def __len__(self):
        """Return the length of the interval."""
        if hasattr(self, "value") and self.value is not None:
            return self.value[1] - self.value[0] + 1  # include both ends
        elif self.interval is not None:
            if self.interval[1] == np.inf:
                e = "Cannot compute length of infinite interval without setting input length."
                raise ValueError(e)
            return self.interval[1] - self.interval[0] + 1
        else:
            raise ValueError("Interval not properly initialized")

    def __str__(self):
        """Return a string representation of the Interval."""
        if self.interval is None:
            return ""
        elif self.interval[1] == np.inf:
            return f"[{str(self.interval[0])},∞]"
        elif len(self.interval) == 2:
            return f"[{self.interval[0]},{self.interval[1]}]"
        else:
            raise ValueError("Invalid interval format.")

    def __repr__(self):
        return self.__str__()
