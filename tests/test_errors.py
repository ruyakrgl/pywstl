#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the Validator class in the utils.errors module.

This module tests all validation methods in the Validator class to ensure
proper error handling and validation logic for WSTL-related operations.
"""

import numpy as np
import pytest

from pywstl.signals import Signal
from pywstl.validation import Validator

import pywstl.wstlpy as wstlpy


class TestValidatorSignalAttributes:
    """Test _validate_signal_attributes method."""

    def test_valid_attributes(self):
        """Test validation passes for valid name and value."""
        name = "signal_name"
        value = np.array([[1, 2, 3], [4, 5, 6]])
        Validator._validate_signal_attributes(name, value)

    def test_invalid_name_type(self):
        """Test validation fails for non-string name."""
        name = 123
        value = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(TypeError, match="Name must be a string"):
            Validator._validate_signal_attributes(name, value)

    def test_invalid_value_type(self):
        """Test validation fails for non-numpy array value."""
        name = "signal_name"
        value = [[1, 2, 3], [4, 5, 6]]

        with pytest.raises(TypeError, match="Value must be a numpy array"):
            Validator._validate_signal_attributes(name, value)

    def test_invalid_value_dimension(self):
        """Test validation fails for value with less than 2 dimensions."""
        name = "signal_name"
        value = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Value must be at least 2-dimensional"):
            Validator._validate_signal_attributes(name, value)


class TestValidatorSignalInput:
    """Test _validate_signal_input method."""

    def test_valid_signal_instance(self):
        """Test validation passes for valid Signal instance."""
        mock_signal = Signal("mock_signal", np.array([1, 2, 3]))

        # Should not raise any exception
        Validator._validate_signal_input(mock_signal)

    def test_valid_signal_tuple(self):
        """Test validation passes for tuple of Signal instances."""
        mock_signal1 = Signal("mock_signal", np.array([1, 2, 3]))
        mock_signal2 = Signal("mock_signal_2", np.array([4, 5, 6]))
        signals_tuple = (mock_signal1, mock_signal2)

        # Should not raise any exception
        Validator._validate_signal_input(signals_tuple)

    def test_valid_object_with_value_attribute(self):
        """Test validation passes for object with 'value' attribute."""

        class MockSignal:
            def __init__(self, value):
                self.value = value
                self.name = "mock_signal"

        mock_obj = MockSignal(np.array([1, 2, 3]))
        # Should not raise any exception
        with pytest.raises(TypeError, match="Signal must be a tuple or Signal instance"):
            Validator._validate_signal_input(mock_obj)

    def test_invalid_signal_type(self):
        """Test validation fails for invalid signal type."""
        with pytest.raises(TypeError, match="Signal must be a tuple or Signal instance"):
            Validator._validate_signal_input("invalid_signal")

    def test_invalid_tuple_contents(self):
        """Test validation fails for tuple with non-Signal contents."""
        invalid_tuple = ("not_signal", 123)

        with pytest.raises(TypeError, match="Signal must be a tuple or Signal instance"):
            Validator._validate_signal_input(invalid_tuple)

    def test_empty_tuple(self):
        """Test validation fails for empty tuple."""
        with pytest.raises(TypeError, match="Signal must be a tuple or Signal instance"):
            Validator._validate_signal_input(())


class TestValidatorPredicateThreshold:
    """Test _validate_predicate_threshold method."""

    def test_valid_threshold(self):
        """Test validation passes for valid numeric threshold."""
        Validator._validate_predicate_threshold(0.5)
        Validator._validate_predicate_threshold(-1.0)
        Validator._validate_predicate_threshold(0)

    def test_invalid_threshold_type(self):
        """Test validation fails for non-numeric threshold."""
        with pytest.raises(TypeError, match="value should be a numeric value, not <class 'str'>."):
            Validator._validate_predicate_threshold("invalid_threshold")

        with pytest.raises(TypeError, match="value should be a numeric value, not <class 'list'>."):
            Validator._validate_predicate_threshold([1, 2, 3])


class TestValidatorWeightsDict:
    """Test _validate_weights_dict method."""

    def test_valid_weights_dict(self):
        """Test validation passes for valid weights dictionary."""
        valid_weights = {"weight1": [1.0, 2.0, 3.0], "weight2": [0.5, 1.5, 2.5]}

        Validator._validate_weights_dict(valid_weights)

    def test_valid_single_weight(self):
        """Test validation passes for single weight value."""
        valid_weights = {"weight1": [1.0]}

        Validator._validate_weights_dict(valid_weights)

    def test_invalid_dict_type(self):
        """Test validation fails for non-dictionary type."""
        with pytest.raises(TypeError, match="'weights_dict' should be a dictionary"):
            Validator._validate_weights_dict("not_a_dict")

    def test_empty_weights_dict(self):
        """Test validation fails for empty dictionary."""
        with pytest.raises(ValueError, match="'weights_dict' should not be empty"):
            Validator._validate_weights_dict({})

    def test_negative_weights(self):
        """Test validation fails for negative weights."""
        invalid_weights = {"weight1": [1.0, -2.0, 3.0]}

        with pytest.raises(
            ValueError, match="All weights in `weights_dict` must be strictly positive"
        ):
            Validator._validate_weights_dict(invalid_weights)

    def test_zero_weights(self):
        """Test validation fails for zero weights."""
        invalid_weights = {"weight1": [1.0, 0.0, 3.0]}

        with pytest.raises(
            ValueError, match="All weights in `weights_dict` must be strictly positive"
        ):
            Validator._validate_weights_dict(invalid_weights)

    def test_valid_keys(self):
        formula = wstlpy.And(
            Signal("x", np.array([1, 2, 3])) >= 0, Signal("y", np.array([4, 5, 6])) <= 10
        )

        weights = {"(x>=0)and(y<=10)": np.array([1, 2])}
        Validator._validate_weights_dict(weights, formula)

    def test_invalid_keys(self):
        formula = wstlpy.And(
            Signal("x", np.array([1, 2, 3])) >= 0, Signal("y", np.array([4, 5, 6])) <= 10
        )

        weights = {"invalid_key": np.array([1, 2])}
        with pytest.raises(
            ValueError, match="not found in weights dictionary keys \\['invalid_key'\\]."
        ):
            Validator._validate_weights_dict(weights, formula)


class TestValidatorTimeParameter:
    """Test _validate_time_parameter method."""

    def test_valid_time_parameter(self):
        """Test validation passes for valid time parameter."""
        Validator._validate_time_parameter(5, 10)

    def test_valid_zero_time(self):
        """Test validation passes for zero time."""
        Validator._validate_time_parameter(0, 10)

    def test_invalid_time_type(self):
        """Test validation fails for non-integer time."""
        with pytest.raises(TypeError, match="Time should be an integer"):
            Validator._validate_time_parameter(5.5, 10)

    def test_negative_time(self):
        """Test validation fails for negative time."""
        with pytest.raises(ValueError, match="Time should be non-negative"):
            Validator._validate_time_parameter(-1, 10)

    def test_time_exceeds_signal_length(self):
        """Test validation fails when time exceeds signal length."""
        with pytest.raises(ValueError, match="Time 10 exceeds the maximum allowed time length 5"):
            Validator._validate_time_parameter(10, 5)

    def test_time_equals_signal_length(self):
        """Test validation fails when time equals signal length."""
        with pytest.raises(ValueError, match="Time 5 exceeds the maximum allowed time length 5"):
            Validator._validate_time_parameter(5, 5)


class TestValidatorWeightRange:
    """Test _validate_weight_range method."""

    def test_valid_weight_range(self):
        """Test validation passes for valid weight range."""
        Validator._validate_weight_range([0.1, 1.0])

    def test_valid_equal_bounds(self):
        """Test validation passes for equal lower and upper bounds."""
        Validator._validate_weight_range([1.0, 1.0])

    def test_invalid_range_type(self):
        """Test validation fails for non-list type."""
        with pytest.raises(TypeError, match="'w_range' should be a list of two elements"):
            Validator._validate_weight_range((0.1, 1.0))

    def test_invalid_range_length(self):
        """Test validation fails for wrong number of elements."""
        with pytest.raises(TypeError, match="'w_range' should be a list of two elements"):
            Validator._validate_weight_range([0.1, 1.0, 2.0])

    def test_non_numeric_values(self):
        """Test validation fails for non-numeric values."""
        with pytest.raises(TypeError, match="'w_range' should contain numeric values only"):
            Validator._validate_weight_range([0.1, "invalid"])

    def test_descending_range(self):
        """Test validation fails for descending range."""
        with pytest.raises(ValueError, match="'w_range' should be in the form \\[min, max\\]"):
            Validator._validate_weight_range([1.0, 0.1])

    def test_negative_lower_bound(self):
        """Test validation fails for negative lower bound."""
        with pytest.raises(ValueError, match="'w_range' should be strictly positive"):
            Validator._validate_weight_range([-0.1, 1.0])

    def test_zero_lower_bound(self):
        """Test validation fails for zero lower bound."""
        with pytest.raises(ValueError, match="'w_range' should be strictly positive"):
            Validator._validate_weight_range([0.0, 1.0])


class TestValidatorNoSamples:
    """Test _validate_no_samples method."""

    def test_valid_no_samples(self):
        """Test validation passes for valid number of samples."""
        Validator._validate_no_samples(100)

    def test_valid_single_sample(self):
        """Test validation passes for single sample."""
        Validator._validate_no_samples(1)

    def test_invalid_samples_type(self):
        """Test validation fails for non-integer type."""
        with pytest.raises(TypeError, match="'no_samples' should be an integer"):
            Validator._validate_no_samples(100.5)

    def test_zero_samples(self):
        """Test validation fails for zero samples."""
        with pytest.raises(ValueError, match="'no_samples' should be a positive integer"):
            Validator._validate_no_samples(0)

    def test_negative_samples(self):
        """Test validation fails for negative samples."""
        with pytest.raises(ValueError, match="'no_samples' should be a positive integer"):
            Validator._validate_no_samples(-5)

    def test_invalid_type(self):
        """Test validation fails for invalid type."""
        with pytest.raises(TypeError, match="'no_samples' should be an integer"):
            Validator._validate_no_samples("100")


class TestValidatorRandomFlag:
    """Test _validate_random_flag method."""

    def test_valid_true_flag(self):
        """Test validation passes for True flag."""
        Validator._validate_random_flag(True)

    def test_valid_false_flag(self):
        """Test validation passes for False flag."""
        Validator._validate_random_flag(False)

    def test_invalid_flag_type(self):
        """Test validation fails for non-boolean type."""
        with pytest.raises(TypeError, match="'random' should be a boolean"):
            Validator._validate_random_flag("true")

    def test_invalid_integer_flag(self):
        """Test validation fails for integer type."""
        with pytest.raises(TypeError, match="'random' should be a boolean"):
            Validator._validate_random_flag(1)


class TestValidatorSeed:
    """Test _validate_seed method."""

    def test_valid_integer_seed(self):
        """Test validation passes for integer seed."""
        Validator._validate_seed(42)

    def test_valid_none_seed(self):
        """Test validation passes for None seed."""
        Validator._validate_seed(None)

    def test_valid_zero_seed(self):
        """Test validation passes for zero seed."""
        Validator._validate_seed(0)

    def test_valid_negative_seed(self):
        """Test validation passes for negative seed."""
        Validator._validate_seed(-1)

    def test_invalid_seed_type(self):
        """Test validation fails for non-integer, non-None type."""
        with pytest.raises(TypeError, match="'seed' should be an integer or None"):
            Validator._validate_seed(42.5)

    def test_invalid_string_seed(self):
        """Test validation fails for string type."""
        with pytest.raises(TypeError, match="'seed' should be an integer or None"):
            Validator._validate_seed("42")


class TestValidatorFormula:
    """Test _validate_formula method."""

    def test_valid_wstlpy_formula(self):
        """Test validation passes for pywstl.wstlpy.WSTLFormula instance."""
        mock_formula = wstlpy.And(
            Signal("x", np.array([1, 2, 3])) >= 0, Signal("y", np.array([4, 5, 6])) <= 10
        )
        # Should not raise any exception
        Validator._validate_formula(mock_formula)

    def test_valid_duck_typed_formula(self):
        """Test validation passes for object with required methods."""

        class MockFormula:
            def robustness(self):
                return 1.0

        mock_formula = MockFormula()

        # Should not raise any exception
        with pytest.raises(TypeError, match="'formula' should be a WSTLFormula"):
            Validator._validate_formula(mock_formula)

    def test_invalid_formula_type(self):
        """Test validation fails for invalid formula type."""
        with pytest.raises(TypeError, match="'formula' should be a WSTLFormula"):
            Validator._validate_formula("invalid_formula")

    def test_invalid_formula_missing_methods(self):
        """Test validation fails for object missing required methods."""

        class MockFormula:
            pass

        mock_formula = MockFormula()

        with pytest.raises(TypeError, match="'formula' should be a WSTLFormula"):
            Validator._validate_formula(mock_formula)

    def test_invalid_formula_non_callable_method(self):
        """Test validation fails for object with non-callable required attribute."""

        class MockFormula:
            def __init__(self):
                self.robustness = "not_callable"

        mock_formula = MockFormula()

        with pytest.raises(TypeError, match="'formula' should be a WSTLFormula"):
            Validator._validate_formula(mock_formula)


class TestValidatorInterval:
    """Test _validate_interval method."""

    def test_valid_interval(self):
        """Test validation passes for valid interval."""
        Validator._validate_interval([0, 5])

    def test_valid_single_point_interval(self):
        """Test validation passes for single point interval."""
        Validator._validate_interval([3, 3])

    def test_invalid_negative_interval(self):
        """Test validation fails for negative interval."""
        with pytest.raises(ValueError, match="'interval' should contain non-negative integers"):
            Validator._validate_interval([-1, 1])

    def test_invalid_interval_type(self):
        """Test validation fails for non-list/tuple type."""
        with pytest.raises(TypeError, match="'interval' should be a list/tuple of two elements"):
            Validator._validate_interval("not_a_list_or_tuple")

    def test_invalid_interval_length(self):
        """Test validation fails for wrong number of elements."""
        with pytest.raises(TypeError, match="'interval' should be a list/tuple of two elements"):
            Validator._validate_interval([0, 5, 10])

    def test_non_integer_values(self):
        """Test validation fails for non-integer values."""
        with pytest.raises(TypeError, match="'interval' should contain integers only"):
            Validator._validate_interval([0.5, 5])

    def test_descending_interval(self):
        """Test validation fails for descending interval."""
        with pytest.raises(ValueError, match="'interval' should be in the form \\[start, end\\]"):
            Validator._validate_interval([5, 0])


class TestValidatorDictNames:
    """Test _validate_dict_names method."""

    def test_valid_dict_names(self):
        """Test validation passes for valid dictionary names."""
        valid_names = ["weight1", "weight2", "param3"]

        Validator._validate_dict_names(valid_names)

    def test_valid_single_name(self):
        """Test validation passes for single name."""
        valid_names = ["weight1"]

        Validator._validate_dict_names(valid_names)

    def test_invalid_names_type(self):
        """Test validation fails for non-list type."""
        with pytest.raises(TypeError, match="'dict_names' should be a list"):
            Validator._validate_dict_names("not_a_list")

    def test_empty_dict_names(self):
        """Test validation fails for empty list."""
        with pytest.raises(ValueError, match="'dict_names' should not be empty"):
            Validator._validate_dict_names([])

    def test_non_string_names(self):
        """Test validation fails for non-string elements."""
        invalid_names = ["weight1", 123, "weight3"]

        with pytest.raises(TypeError, match="'dict_names' should contain strings only"):
            Validator._validate_dict_names(invalid_names)


class TestValidatorSignalLength:
    """Test _validate_signal_length method."""

    def test_valid_signal_length(self):
        """Test validation passes for valid signal length."""
        Validator._validate_signal_length(10)

    def test_valid_single_length(self):
        """Test validation passes for length of 1."""
        Validator._validate_signal_length(1)

    def test_invalid_length_type(self):
        """Test validation fails for non-integer type."""
        with pytest.raises(TypeError, match="'signal_length' should be an integer"):
            Validator._validate_signal_length(10.5)

    def test_zero_signal_length(self):
        """Test validation fails for zero length."""
        with pytest.raises(ValueError, match="'signal_length' should be a positive integer"):
            Validator._validate_signal_length(0)

    def test_negative_signal_length(self):
        """Test validation fails for negative length."""
        with pytest.raises(ValueError, match="'signal_length' should be a positive integer"):
            Validator._validate_signal_length(-5)


class TestValidatorTimeLength:
    """Test _validate_time_length method."""

    def test_valid_time_length(self):
        """Test validation passes for valid time length."""
        Validator._validate_time_length(10)
        Validator._validate_time_length(1)

    def test_invalid_time_length_type(self):
        """Test validation fails for non-integer type."""
        with pytest.raises(TypeError, match="'time_length' should be an integer"):
            Validator._validate_time_length(10.5)

    def test_zero_time_length(self):
        """Test validation fails for zero length."""
        with pytest.raises(ValueError, match="'time_length' should be a positive integer"):
            Validator._validate_time_length(0)

    def test_negative_time_length(self):
        """Test validation fails for negative length."""
        with pytest.raises(ValueError, match="'time_length' should be a positive integer"):
            Validator._validate_time_length(-5)


if __name__ == "__main__":
    pytest.main([__file__])
