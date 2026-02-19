# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL Validation Module

This module provides validation functions for WSTL-related operations.

Example:
    >>> Validator._validate_signal_attributes('x', np.array([[1, 2]]))
    # Passes validation

    >>> Validator._validate_signal_attributes('x', [1, 2])
    TypeError: Value must be a numpy array, got <class 'list'>

Author: Ruya Karagulle
Date: September 2025
"""
import numbers
from typing import Any

import numpy as np


class Validator:
    """Validator class for WSTL objects and parameters."""

    @staticmethod
    def _validate_signal_attributes(name, value) -> None:
        """
        Helper function to validate signal attributes.

        Args:
            name: Name of the signal attribute
            value: Value of the signal attribute

        Raises:
            TypeError: If name is not a string or value is not a numpy array
            ValueError: If value is not at least 2-dimensional
        """
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}.")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Value must be a numpy array, got {type(value)}.")
        if value.ndim < 2:
            raise ValueError(f"Value must be at least 2-dimensional, got {value.ndim}D array.")

    @staticmethod
    def _validate_signal_input(signals) -> None:
        """
        Helper function to validate signal input format.

        Args:
            signals: Input signals to validate

        Raises:
            TypeError: If signals format is invalid
        """

        def is_valid_signal(obj):
            """Check if object is a Signal using duck typing."""
            obj_type = type(obj)
            if obj_type.__name__ == "Signal" and "signals" in obj_type.__module__:
                if hasattr(obj, "name") and hasattr(obj, "value"):
                    return True
            return False

        def is_valid_signal_tuple(obj):
            """Check if object is a valid signal or tuple of signals."""
            if is_valid_signal(obj):
                return True
            if isinstance(obj, tuple):
                if len(obj) == 0:
                    return False
                return all(is_valid_signal_tuple(item) for item in obj)
            return False

        if not (is_valid_signal_tuple(signals) or is_valid_signal(signals)):
            raise TypeError(f"Signal must be a tuple or Signal instance, got {type(signals)}.")

    @staticmethod
    def _validate_predicate_threshold(value) -> None:
        """
        Helper function to validate predicate threshold values.

        Args:
            value: Value to validate

        Raises:
            TypeError: If value format is invalid
        """
        if not isinstance(value, numbers.Number):
            raise TypeError(f"value should be a numeric value, not {type(value)}.")

    @staticmethod
    def _validate_weights_dict(weights_dict: dict[str, Any], formula=None) -> None:
        """
        Helper function to validate weights dictionary format and values.

        Args:
            weights_dict: Dictionary containing weight names and values
            formula: Optional formula to check weight keys against

        Raises:
            TypeError: If weights_dict is not a dictionary
            ValueError: If weights dictionary is empty or contains non-positive weights
        """
        if not isinstance(weights_dict, dict):
            raise TypeError(f"'weights_dict' should be a dictionary, got {type(weights_dict)}.")
        if len(weights_dict) == 0:
            raise ValueError("'weights_dict' should not be empty.")
        for key, values in weights_dict.items():
            if (np.array(values) <= 0).any():
                raise ValueError(
                    f"All weights in `weights_dict` must be strictly positive. "
                    f"Found non-positive values for key '{key}'."
                )

        if formula is not None and hasattr(formula, "keys"):
            formula_weights = formula.keys
            for key in formula_weights:
                if key not in weights_dict.keys():
                    if key.startswith("G") or key.startswith("F") or key.startswith("("):
                        raise ValueError(
                            f"Weight name '{key}' not found in weights dictionary "
                            f"keys {list(weights_dict.keys())}."
                        )

    @staticmethod
    def _validate_time_parameter(time: int, signal_length: int = None) -> None:
        """
        Helper function to validate time parameter.

        Args:
            time: Time instance to validate
            signal_length: Maximum allowed time length

        Raises:
            TypeError: If time is not an integer
            ValueError: If time is negative or exceeds signal length
        """
        if not isinstance(time, int):
            raise TypeError(f"Time should be an integer, got {type(time)}.")
        if time < 0:
            raise ValueError(f"Time should be non-negative, got {time}.")
        if signal_length is not None and time >= signal_length:
            raise ValueError(
                f"Time {time} exceeds the maximum allowed time length {signal_length}."
            )

    @staticmethod
    def _validate_weight_range(w_range: list[float]) -> None:
        """
        Helper function to validate weight range format and values.

        Args:
            w_range: Weight range to validate.
                Must be a list of two strictly positive numbers in ascending order.

        Raises:
            TypeError: If w_range format is invalid
            ValueError: If weight range values are invalid or not strictly positive
                and in ascending order
        """
        if not isinstance(w_range, list) or len(w_range) != 2:
            raise TypeError(f"'w_range' should be a list of two elements, got {type(w_range)}.")
        if not all(isinstance(x, numbers.Number) for x in w_range):
            raise TypeError(f"'w_range' should contain numeric values only, got {w_range}.")
        if w_range[1] < w_range[0]:
            raise ValueError(f"'w_range' should be in the form [min, max], got {w_range}.")
        if w_range[0] <= 0:
            raise ValueError(f"'w_range' should be strictly positive, got {w_range}.")

    @staticmethod
    def _validate_no_samples(no_samples: int) -> None:
        """
        Helper function to validate number of samples.

        Args:
            no_samples: Number of samples to validate

        Raises:
            TypeError: If no_samples is not an integer
            ValueError: If no_samples is not a positive integer
        """
        if not isinstance(no_samples, int):
            raise TypeError(f"'no_samples' should be an integer, got {type(no_samples)}.")
        if no_samples <= 0:
            raise ValueError(f"'no_samples' should be a positive integer, got {no_samples}.")

    @staticmethod
    def _validate_random_flag(random: bool) -> None:
        """
        Helper function to validate random flag.

        Args:
            random: Random flag to validate

        Raises:
            TypeError: If random is not a boolean
        """
        if not isinstance(random, bool):
            raise TypeError(f"'random' should be a boolean, got {type(random)}.")

    @staticmethod
    def _validate_seed(seed: int | None) -> None:
        """
        Helper function to validate seed value.

        Args:
            seed: Seed value to validate. Can be None or an integer.

        Raises:
            TypeError: If seed is not None or an integer
        """
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"'seed' should be an integer or None, got {type(seed)}.")

    @staticmethod
    def _validate_formula(formula: Any) -> None:
        """
        Helper function to validate formula using duck typing.

        Args:
            formula: Formula to validate. Should be a WSTLFormula.

        Raises:
            TypeError: If formula is not a WSTLFormula.
        """
        # Check inheritance for WSTLFormula
        for cls in formula.__class__.__mro__:
            if "WSTLFormula" in cls.__name__:
                return

        raise TypeError(f"'formula' should be a WSTLFormula, got {type(formula)}.")

    @staticmethod
    def _validate_interval(interval: list[int]) -> None:
        """
        Helper function to validate interval format for temporal operators.

        Args:
            interval: Interval to validate. Must be a list of two integers.

        Raises:
            TypeError: If interval is not a list of two integers
            ValueError: If interval format is invalid
        """
        if interval is not None and (not isinstance(interval, (list, tuple)) or len(interval) != 2):
            raise TypeError(
                f"'interval' should be a list/tuple of two elements, got {type(interval)}."
            )
        if isinstance(interval, (list, tuple)):
            if not all(isinstance(x, int) for x in interval):
                raise TypeError(f"'interval' should contain integers only, got {interval}.")
            if interval[0] > interval[1]:
                raise ValueError(f"'interval' should be in the form [start, end], got {interval}.")
            if interval[0] < 0:
                raise ValueError(
                    f"'interval' should contain non-negative integers, got {interval}."
                )

    @staticmethod
    def _validate_signal_length(signal_length: int) -> None:
        """
        Helper function to validate signal length.

        Args:
            signal_length: Length of the signal to validate

        Raises:
            TypeError: If signal_length is not an integer
            ValueError: If signal_length is not a positive integer
        """
        if not isinstance(signal_length, int):
            raise TypeError(f"'signal_length' should be an integer, got {type(signal_length)}.")
        if signal_length <= 0:
            raise ValueError(f"'signal_length' should be a positive integer, got {signal_length}.")

    @staticmethod
    def _validate_time_length(time_length: int) -> None:
        """
        Helper function to validate time length.

        Args:
            time_length: Length of the time to validate

        Raises:
            TypeError: If time_length is not an integer
            ValueError: If time_length is not a positive integer
        """
        if not isinstance(time_length, int):
            raise TypeError(f"'time_length' should be an integer, got {type(time_length)}.")
        if time_length <= 0:
            raise ValueError(f"'time_length' should be a positive integer, got {time_length}.")

    @staticmethod
    def _validate_dict_names(dict_names: list[str]) -> None:
        """
        Helper function to validate dictionary names.

        Args:
            dict_names: List of dictionary names to validate

        Raises:
            TypeError: If dict_names is not a list of strings
            ValueError: If dict_names is empty
        """
        if not isinstance(dict_names, list):
            raise TypeError(f"'dict_names' should be a list, got {type(dict_names)}.")
        if len(dict_names) == 0:
            raise ValueError("'dict_names' should not be empty.")
        if not all(isinstance(name, str) for name in dict_names):
            raise TypeError(f"'dict_names' should contain strings only, got {dict_names}.")
