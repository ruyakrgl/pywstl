# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL Utility Functions

This module provides utility functions for working with WSTL signals and formulas.
"""
from pywstl.validation import Validator


def get_signal_shape(inputs):
    """
    Get the shape of signal inputs.

    Args:
        inputs: Signal or tuple of signals

    Returns:
        tuple: (signal_no, time_no) - number of signals and timesteps

    Raises:
        TypeError: If inputs are not valid Signal objects or tuples
    """
    Validator._validate_signal_input(inputs)

    if isinstance(inputs, tuple):
        inputs = inputs[0]
        signal_no, time_no = get_signal_shape(inputs)
    # Check if it's a Signal object (duck typing)
    elif hasattr(inputs, "value") and hasattr(inputs.value, "shape"):
        signal_no, time_no = inputs.value.shape[0], inputs.value.shape[1]
    else:
        raise TypeError(
            f"Signals are expected to be of type Signal or tuple of Signals, got {type(inputs)}."
        )

    return signal_no, time_no


def get_subformulas(formula: str) -> tuple[str, str]:
    """
    Parse a formula string to extract its two subformulas.

    Args:
        formula: String representation of a binary formula

    Returns:
        Tuple of (subformula1, subformula2)

    Raises:
        ValueError: If formula cannot be parsed
    """
    inner = formula.strip()

    # Find the main operator by counting parentheses
    paren_count = 0
    operator_pos = -1

    # Look for 'and' or 'or' at the top level (paren_count == 0)
    i = 0
    while i < len(inner):
        if inner[i] == "(":
            paren_count += 1
        elif inner[i] == ")":
            paren_count -= 1
        elif paren_count == 0:
            # Check for 'and'
            if inner[i : i + 3] == "and":
                operator_pos = i
                operator_length = 3
                break
            # Check for 'or'
            elif inner[i : i + 2] == "or":
                operator_pos = i
                operator_length = 2
                break
        i += 1

    if operator_pos != -1:
        sf1 = inner[:operator_pos]
        sf2 = inner[operator_pos + operator_length :]

        while sf1[-1] == "_":
            sf1 = sf1[:-1]
        while sf2[-1] == "_":
            sf2 = sf2[:-1]

        sf1 = sf1[1:-1].strip()
        sf2 = sf2[1:-1].strip()

        return sf1, sf2
    else:
        raise ValueError(f"Could not parse subformulas from: {formula}")
