# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL (Weighted Signal Temporal Logic) Numpy Programming Module

This module implements the quanitative semanics computation for Weighted Signal Temporal Logic
(WSTL) formulas and operators with Numpy library. It provides classes for creating and evaluating
WSTL formulas.

Classes:
    WSTLFormula: Base class for all WSTL formulas
    UnaryOperator: Base class for unary temporal operators (Always, Eventually)
    BinaryOperator: Base class for binary logical operators (And, Or)
    And: Logical conjunction operator
    Or: Logical disjunction operator
    Always: Temporal "always" operator (globally)
    Eventually: Temporal "eventually" operator (finally)

Key Features:
    - Support for weighted robustness computation

Usage:
    Create WSTL formulas by combining operators and predicates, then use them
    for robustness computation.

Dependencies:
    - numpy: Numerical computations
    - intervals: Interval handling utilities
    - operators: WSTL min/max operators
-----------------------------------------------------------------------------
"""

import logging

import numpy as np

from pywstl import intervals, operations, wstlbase
from pywstl.utils import get_signal_shape
from pywstl.validation import Validator

logger = logging.getLogger(__name__)


# ================= WSTL FORMULA CLASS =================
class WSTLFormula(wstlbase.WSTLKeyMixin):
    """
    Base class for all WSTL formula objects.

    Attributes:
        keys: List of names for the weights dictionary.
        weights: Dictionary storing weight values for the formula.

    Methods:
        robustness: Compute the weighted robustness of the formula for given signals.
        set_weights: Set weights for the formula.
        set_weights_from_dict: Set weights from a provided weight dictionary.
        get_weight_array: Convert weights dictionary to a 2D numpy array.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a WSTLFormula instance.

        Attributes:
            keys: A list of names for the weights dictionary,
                used to uniquely identify weights associated with different
                formulas.
            weights: A dictionary storing weight values for the
                formula, where keys are weight names and values are numerical
                arrays.
        """
        self.set_keys()
        self.set_key_hash()
        self.key_map = [self.keys, self.key_hashes]
        self.weights = {}  # Ensure weights dictionary is always initialized

    # ================= WEIGHT MANAGEMENT =================
    def set_weights(
        self,
        signals: tuple,
        w_range: list[float] = [1, 1],
        no_samples: int = 1,
        random: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Set weights for the formula.

        Args:
            signals: Input signals for the formula
            w_range: Range of weights to be set, in the form [min, max]
            no_samples: Number of samples for weight valuation
            random: If True, weights are randomly initialized within the range
            seed: Random seed for reproducibility (optional)

        Raises:
            TypeError: If signals format is invalid or parameter types are wrong
            ValueError: If parameter values are invalid
        """
        # Validate signals
        Validator._validate_signal_input(signals)
        Validator._validate_weight_range(w_range)
        Validator._validate_no_samples(no_samples)
        Validator._validate_random_flag(random)
        Validator._validate_seed(seed)

        if self.keys is None:
            raise ValueError(
                "Weight keys are not set. Please call set_keys() before setting weights."
            )

        # Set random seed if provided
        if seed is not None:
            # Warning if seed is set when random is False
            if random is False:
                logging.info("Seed only makes sense when 'random' is True.")
            np.random.seed(seed)

        if random and w_range == [1, 1]:
            logging.warning("Random weights requested but range is [1, 1]. All weights are 1.")

        # Set weights for the formula
        self._set_weights(signals, w_range, no_samples, random)

    def set_weights_from_dict(self, weights_dict: dict[str, np.ndarray]) -> None:
        """
        Set weights from a dictionary.

        Args:
            weights_dict: Dictionary containing weight names and values.
                Example:
                    weights_dict = {
                        "weight1": np.array([0.5, 0.8]),
                        "weight2": np.array([1.2, 0.9])
                    }

        Raises:
            TypeError: If weights_dict is not a dictionary
            ValueError: If weights dictionary is empty or contains non-positive weights
        """
        # Validate signals
        Validator._validate_weights_dict(weights_dict, self)

        if self.keys is None:
            raise ValueError(
                "Weight keys are not set. Please call set_keys() before setting weights."
            )

        # Set weights from the dictionary
        self._set_weights_from_dict(weights_dict)

    def get_weight_array(self) -> np.ndarray:
        """
        Convert weights dictionary to a 2D numpy array.

        Returns:
            A 2D numpy array combining all weights.

        Raises:
            ValueError: If the weights dictionary is empty.
        """
        if self.weights is None or len(self.weights) == 0:
            raise ValueError("Weights dictionary is empty.")

        weight_list = []
        for key in self.weights.keys():
            w = self.weights[key].reshape(-1, 1)
            weight_list.append(w)

        weights_array = np.vstack(weight_list)
        return weights_array

    # ================ ROBUSTNESS COMPUTATION =================
    def robustness(self, signals: tuple, t: int = 0) -> np.ndarray:
        """
        Return the weighted robustness value for given input signals at time t.

        Note that robustness is computed per each time instant.

        Args:
            signals: Input signals
            t: Time instance for which to compute the robustness

        Returns:
            WSTL weighted robustness values

        Raises:
            TypeError: If signals have invalid type
            ValueError: If time instance is invalid
        """
        # Get the time length of signals
        _, time_length = get_signal_shape(signals)

        # Validate signals
        Validator._validate_signal_input(signals)
        Validator._validate_time_parameter(t, time_length)

        # Convert signals to the appropriate format
        robustness = self._robustness(signals)[:, t, :, :]

        return robustness.squeeze(-1)

    def subrobustness(self, signals: tuple, t: int = 0) -> dict[str, np.ndarray]:
        if self.weights == {}:
            self.set_weights(signals)
        robustness_dict = {}
        robustness_dict = self._subrobustness(signals, t, robustness_dict)
        return robustness_dict

    def get_horizon(self, signals, time, horizon=[0, 0]):
        """
        Get the horizon for the formula based on the signals and time.
        Args:
            signals: Input signals for the formula
            time: Time instance for constraints
            horizon: Current horizon range [start, end]
        Returns:
            Horizon for the formula, which is a range of time steps
        """
        _, time_length = get_signal_shape(signals)

        if isinstance(self, UnaryOperator):
            self.interval.set_interval(time_length)
            horizon = self.subformula.get_horizon(signals, time, horizon)
            interval = self.interval.value
            return [
                time + min(horizon[0], interval[0]),
                min(time_length - 1, time + horizon[1] + interval[1]),
            ]

        elif isinstance(self, BinaryOperator):
            # For binary operators, we need to consider both subformulas
            horizon1 = self.subformula1.get_horizon(signals, time, horizon)
            horizon2 = self.subformula2.get_horizon(signals, time, horizon)

            # Combine horizons from both subformulas
            return [min(horizon1[0], horizon2[0]), max(horizon1[1], horizon2[1])]

        return horizon


class UnaryOperator(wstlbase.UnaryMixin, WSTLFormula):
    """
    Defines Temporal operators in the syntax: Always, Eventually.
    Until is defined separately.

    Attributes:
        subformula (WSTLFormula): Subformula encapsulated by the temporal operator.
        interval (Interval): Time interval associated with the temporal operator.
        operation (None): Placeholder for the specific temporal operation.
            Subclasses must define this attribute to specify the behavior of the operator.
            For example, subclasses can set it to a function like `wstlMin` or `wstlMax`
            to represent the "always" or "eventually" temporal operations, respectively.

    Methods:
        _set_key: Sets dictionary names for the unary operator and its subformula.
        _set_weights: Assigns weight values to temporal operators given a range.
        _set_weights_from_dict: Set weights from a dictionary for the temporal operator.
        _robustness: Computes weighted robustness of temporal operators.
    """

    def __init__(self, subformula: WSTLFormula, interval: list[int] | None = None):
        """
        Initialize a UnaryOperator instance.

        Args:
            subformula: Subformula encapsulated by the temporal operator
            interval: Time interval associated with the temporal operator (optional)

        Raises:
            TypeError: If subformula is not a WSTLFormula
            ValueError: If interval format is invalid
        """
        # Validate subformula type
        Validator._validate_formula(subformula)
        self.interval = intervals.Interval(interval)
        self.subformula = subformula
        super().__init__()

    def _set_weights(
        self, signals: tuple, w_range: list[float], no_samples: int, random: bool
    ) -> None:
        """
        Assign weight values to temporal operators given a range.

        Args:
            signals: Input signals
            w_range: Weight range for random initialization
            no_samples: Number of weight valuation samples to be set (useful for random case)
            random: Flag for random initialization
        """
        Validator._validate_signal_input(signals)
        Validator._validate_weight_range(w_range)
        Validator._validate_no_samples(no_samples)
        Validator._validate_random_flag(random)

        # Set weights for the subformula first
        self.subformula._set_weights(signals, w_range, no_samples, random)

        # Get robustness trace to determine interval
        trace = self.subformula._robustness(signals)
        self.interval.set_interval(trace.shape[1])

        # Copy subformula weights
        for key in self.subformula.weights.keys():
            self.weights[key] = self.subformula.weights[key]

        # Set operator-specific weights
        self._set_operator_weights(w_range, no_samples, random)

    def _set_operator_weights(self, w_range, no_samples, random):
        """
        Assigns weight values for Always.

        Args:
            w_range (list): The range within which weight values are generated.
            no_samples (int): The number of samples for weight values.
            random (bool): If True, generates random weight values; otherwise, uses unit weights.
        """

        interval_length = self.interval.value[1] - self.interval.value[0] + 1
        if random:
            self.weights[self.key] = w_range[0] + (w_range[1] - w_range[0]) * np.random.rand(
                interval_length, no_samples
            )
        else:
            self.weights[self.key] = np.ones((interval_length, no_samples))

    def _set_weights_from_dict(self, weights_dict: dict[str, np.ndarray]) -> None:
        """
        Set weights from a dictionary for the temporal operator.

        Args:
            weights_dict: Dictionary containing weight names and values
        """
        Validator._validate_weights_dict(weights_dict, self)

        # Set weights for the subformula first
        self.subformula._set_weights_from_dict(weights_dict)

        # Copy subformula weights to the operator's weights
        if hasattr(self.subformula, "weights"):
            for key in self.subformula.weights.keys():
                self.weights[key] = self.subformula.weights[key]

        # Set operator-specific weights from the dictionary
        self._set_operator_weights_from_dict(weights_dict)

    def _set_operator_weights_from_dict(self, weights_dict: dict):
        """
        Set weights from a dictionary for the Eventually operator.

        Args:
            weights_dict: Dictionary containing weight names and values.
        """
        Validator._validate_weights_dict(weights_dict, self)

        if self.key in weights_dict:
            if weights_dict[self.key].ndim == 1:
                self.weights[self.key] = np.array(weights_dict[self.key]).reshape(-1, 1)
            else:
                self.weights[self.key] = np.array(weights_dict[self.key])
        else:
            raise KeyError(f"Weight '{self.key}' not found in the provided dictionary.")

    def _robustness(self, signals):
        """
        Computes weighted robustness of temporal operators.

        Args:
            signals: Input signals.

        Returns:
            np.ndarray: Weighted robustness values
        """
        Validator._validate_signal_input(signals)

        _, time_length = get_signal_shape(signals)
        self.interval.set_interval(time_length)

        # Compute the robustness trace for the subformula
        trace = self.subformula._robustness(signals)

        # Compute the robustness of the unary operator
        outputs = self._operator_robustness(trace)

        return outputs

    def _subrobustness(self, signals, t, all_robustness):
        Validator._validate_signal_input(signals)

        _, time_length = get_signal_shape(signals)
        self.interval.set_interval(time_length)
        etl = min(
            self.interval.value[1] - self.interval.value[0] + 1,
            time_length - t - self.interval.value[0],
        )
        start = t + self.interval.value[0]
        tt_list = list(np.arange(start, start + etl))

        for tt in tt_list:
            all_robustness = self.subformula._subrobustness(signals, tt.item(), all_robustness)

        outputs = self._robustness(signals)

        all_robustness[(self.key, t)] = outputs[:, t, :, :]
        return all_robustness

    def _operator_robustness(self, trace):
        """
        Computes the weighted robustness of Eventually.

        Args:
            trace (np.ndarray): The input trace for computing robustness.

        Returns:
            np.ndarray: The computed weighted robustness.
        """

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
                _output = np.concatenate((_output, operated), axis=1)

        return _output


class BinaryOperator(wstlbase.BinaryMixin, WSTLFormula):
    """
    Defines Logic Operators in the syntax: And, Or.

    Attributes:
        subformula1 (WSTLFormula): Subformula1 encapsulated by the binary operator.
        subformula2 (WSTLFormula): Subformula2 encapsulated by the binary operator.
        operation (None): Placeholder for the specific operation.
            Subclasses must define this attribute to specify the behavior of the operator.
            For example, subclasses can set it to a function like `wstlMin` or `wstlMax`
            to represent the "and" or "or"  operations, respectively.

    Methods:
        _set_key: Sets dictionary names for the operator and its subformula.
        _set_weights: Assigns weight values to operators given a range.
        _set_weights_from_dict: Set weights from a dictionary for the binary operator.
        _robustness: Computes weighted robustness of binary operators.
    """

    def __init__(self, subformula1: WSTLFormula, subformula2: WSTLFormula):
        """
        Initialize a BinaryOperator instance.

        Args:
            subformula1: First subformula
            subformula2: Second subformula

        Raises:
            TypeError: If subformulas are not WSTLFormula instances
        """
        # Validate subformulas
        Validator._validate_formula(subformula1)
        Validator._validate_formula(subformula2)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

        super().__init__()

    def _set_weights(self, signals, w_range, no_samples, random):
        """
        Assigns weight values to logic operators given a range.

        Args:
            signals (tuple): Tuple containing input signals for subformula 1 and 2.
            w_range (list): Weight range for random initialization.
            no_samples (int): Number of weight valuation samples to be set
                            (useful for random case).
            random (bool): Flag for random initialization.
        """
        Validator._validate_signal_input(signals)
        Validator._validate_weight_range(w_range)
        Validator._validate_no_samples(no_samples)
        Validator._validate_random_flag(random)

        # Set weights for subformula 1
        self.subformula1._set_weights(signals[0], w_range, no_samples, random)

        # Copy subformula 1 weights to the binary operator's weights
        # and also to subformula 2
        for keys in self.subformula1.weights.keys():
            self.weights[keys] = self.subformula1.weights[keys]
            self.subformula2.weights[keys] = self.subformula1.weights[keys]

        # Set weights for subformula 2
        self.subformula2._set_weights(signals[1], w_range, no_samples, random)

        # Copy subformula 2 weights to the binary operator's weights
        # and also to subformula 1
        for keys in self.subformula2.weights.keys():
            self.weights[keys] = self.subformula2.weights[keys]
            self.subformula1.weights[keys] = self.subformula2.weights[keys]

        # Set operator-specific weights
        self._set_operator_weights(w_range, no_samples, random)

    def _set_weights_from_dict(self, weights_dict: dict[str, np.ndarray]) -> None:
        """
        Set weights from a dictionary for the binary operator.

        Args:
            weights_dict: Dictionary containing weight names and values
        """
        Validator._validate_weights_dict(weights_dict, self)

        # Set weights for both subformulas
        self.subformula1._set_weights_from_dict(weights_dict)
        self.subformula2._set_weights_from_dict(weights_dict)

        # Copy subformula weights to the binary operator's weights
        for key in self.subformula1.weights.keys():
            self.weights[key] = self.subformula1.weights[key]

        for key in self.subformula2.weights.keys():
            self.weights[key] = self.subformula2.weights[key]

        # Set operator-specific weights from the dictionary
        self._set_operator_weights_from_dict(weights_dict)

    def _robustness(self, signals):
        """
        Computes weighted robustness of logic operators.

        Args:
            signals (tuple): Tuple containing input signals for subformula 1 and 2.

        Returns:
            np.ndarray: Weighted robustness values.
        """
        assert (
            isinstance(signals, tuple) and len(signals) == 2
        ), f"signals should be a tuple of two Signals, got {type(signals)}."

        # Compute the robustness traces for both subformulas
        trace1 = self.subformula1._robustness(signals[0])
        trace2 = self.subformula2._robustness(signals[1])

        # Get the minimum trace length
        # We cannot compare more than the shortest trace
        trace_length = min(trace1.shape[1], trace2.shape[1])
        sample_length = max(trace1.shape[2], trace2.shape[2])

        trace1 = np.pad(
            trace1[:, :trace_length, :],
            ((0, 0), (0, 0), (0, sample_length - trace1.shape[2]), (0, 0)),
            "edge",
        )
        trace2 = np.pad(
            trace2[:, :trace_length, :],
            ((0, 0), (0, 0), (0, sample_length - trace2.shape[2]), (0, 0)),
            "edge",
        )

        trace = np.concatenate((trace1, trace2), axis=-1)

        # Compute the robustness of the binary operator
        outputs = self._operator_robustness(trace)

        return outputs

    def _subrobustness(self, signals, t, all_robustness):
        # Compute the robustness traces for both subformulas
        all_robustness = self.subformula1._subrobustness(signals[0], t, all_robustness)
        all_robustness = self.subformula2._subrobustness(signals[1], t, all_robustness)
        outputs = self._robustness(signals)

        all_robustness[(self.key, t)] = outputs[:, t, :, :]

        return all_robustness

    def _set_operator_weights(self, w_range: list, no_samples: int, random: bool):
        """
        Assigns weight values for And.

        Args:
            w_range (list): A list specifying the range for weight assignment.
            no_samples (int): The number of weight samples to generate.
            random (bool): If True, generates random weights; otherwise, uses default weights.
        """
        Validator._validate_weight_range(w_range)
        Validator._validate_no_samples(no_samples)
        Validator._validate_random_flag(random)

        if random:
            self.weights[self.key] = w_range[0] + (w_range[1] - w_range[0]) * np.random.rand(
                2, no_samples
            )
        else:
            self.weights[self.key] = np.ones((2, no_samples))

    def _set_operator_weights_from_dict(self, weights_dict: dict):
        """
        Set weights from a dictionary for the And operator.

        Args:
            weights_dict (dict): Dictionary containing weight names and values.
        """
        Validator._validate_weights_dict(weights_dict, self)

        if self.key in weights_dict:
            if weights_dict[self.key].ndim == 1:
                self.weights[self.key] = np.array(weights_dict[self.key]).reshape(-1, 1)
            else:
                self.weights[self.key] = np.array(weights_dict[self.key])
        else:
            raise KeyError(f"'{self.key}' not found in the provided weight dictionary.")

    def _operator_robustness(self, signals):
        """
        Computes the weighted robustness of And.

        Args:
            signals (np.ndarray): The input array for robustness computation.

        Returns:
            np.ndarray: The computed weighted robustness values
        """
        # Get weights for the operator
        w = self.weights[self.key]
        _output = self.operation(w.T * signals, axis=-1)

        return _output


class Negation(WSTLFormula):
    def __init__(self, subformula):
        """
        Initialize a Negation instance.

        Args:
            subformula: Subformula to be negated.

        Raises:
            NotImplementedError: Always raised as formulas should be in positive normal form.
        """
        raise NotImplementedError("The formula should be in positive normal form.")


class And(BinaryOperator):
    """
    Defines "And" operator. And operator needs two subformulas.

    Attributes:
        subformula1: First subformula.
        subformula2: Second subformula.

    Attributes:
        operation: wstlMin() - The operation used for conjunction.
    """

    def __init__(self, subformula1: WSTLFormula, subformula2: WSTLFormula):
        """Initialize an And instance."""
        super().__init__(subformula1=subformula1, subformula2=subformula2)

    @property
    def operation_name(self):
        return "and"

    @property
    def operation(self):
        return operations.wstlMin()


class Or(BinaryOperator):
    """
    Defines Or operator. Or operator needs two subformulas.

    Attributes:
        subformula1: First subformula.
        subformula2: Second subformula.
    """

    def __init__(self, subformula1: WSTLFormula, subformula2: WSTLFormula):
        """Initialize an Or instance."""
        super().__init__(subformula1=subformula1, subformula2=subformula2)

    @property
    def operation_name(self):
        return "or"

    @property
    def operation(self):
        return operations.wstlMax()


class Always(UnaryOperator):
    """
    Defines Always operator. Always operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).

    Attributes:
        subformula (WSTL_Formula): The subformula associated with the Always operator.
        interval (Interval): The interval over which the Always operator is evaluated.
        operation (Minish): The operation used for computation.
    """

    def __init__(self, subformula: WSTLFormula, interval: list = None):
        super().__init__(subformula=subformula, interval=interval)

    @property
    def operation_name(self):
        return "G"

    @property
    def operation(self):
        return operations.wstlMin()


class Eventually(UnaryOperator):
    """
    Defines Eventually operator. Eventually operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).

    Attributes:
        subformula: The subformula associated with the Eventually operator.
        interval: The interval over which the Eventually operator is evaluated.
        operation: The operation used for computation.
    """

    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    @property
    def operation_name(self):
        return "F"

    @property
    def operation(self):
        return operations.wstlMax()
