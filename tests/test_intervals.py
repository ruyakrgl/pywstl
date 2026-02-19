#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the Interval class in the intervals module.
"""

import numpy as np
import pytest

from pywstl.intervals import Interval


class TestIntervalInit:
    """Test Interval initialization."""

    def test_finite_interval_list(self):
        """Test creating a finite interval from a list."""
        iv = Interval([0, 10])
        assert iv.interval == (0, 10)

    def test_finite_interval_tuple(self):
        """Test creating a finite interval from a tuple."""
        iv = Interval((2, 8))
        assert iv.interval == (2, 8)

    def test_none_interval(self):
        """Test creating an interval with None (unbounded)."""
        iv = Interval(None)
        assert iv.interval is None

    def test_default_interval_is_none(self):
        """Test that default interval is None."""
        iv = Interval()
        assert iv.interval is None

    def test_infinite_interval(self):
        """Test creating an interval with np.inf upper bound."""
        iv = Interval([5, np.inf])
        assert iv.interval == (5, np.inf)

    def test_single_point_interval(self):
        """Test creating a single-point interval (start == end)."""
        iv = Interval([3, 3])
        assert iv.interval == (3, 3)

    def test_zero_start_interval(self):
        """Test interval starting at zero."""
        iv = Interval([0, 0])
        assert iv.interval == (0, 0)

    def test_invalid_start_greater_than_end(self):
        """Test that start > end raises ValueError."""
        with pytest.raises(ValueError, match="Interval start must be less than or equal to end"):
            Interval([10, 5])

    def test_invalid_format_single_element(self):
        """Test that a single-element list raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            Interval([5])

    def test_invalid_format_three_elements(self):
        """Test that a three-element list raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            Interval([0, 5, 10])

    def test_invalid_format_scalar(self):
        """Test that a scalar value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            Interval(5)

    def test_invalid_format_string(self):
        """Test that a string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            Interval("0,10")

    def test_interval_stored_as_tuple(self):
        """Test that list input is stored as a tuple."""
        iv = Interval([1, 7])
        assert isinstance(iv.interval, tuple)


class TestIntervalSetInterval:
    """Test Interval.set_interval method."""

    def test_finite_interval_no_input_length(self):
        """Test set_interval for a finite interval without input_length."""
        iv = Interval([2, 6])
        iv.set_interval()
        assert iv.value == (2, 6)

    def test_finite_interval_with_input_length(self):
        """Test that input_length is ignored for finite intervals."""
        iv = Interval([2, 6])
        iv.set_interval(input_length=100)
        assert iv.value == (2, 6)

    def test_none_interval_with_input_length(self):
        """Test that None interval uses full signal range."""
        iv = Interval(None)
        iv.set_interval(input_length=20)
        assert iv.value == (0, 19)

    def test_none_interval_without_input_length_raises(self):
        """Test that None interval without input_length raises ValueError."""
        iv = Interval(None)
        with pytest.raises(ValueError, match="Input length is required"):
            iv.set_interval()

    def test_infinite_interval_with_input_length(self):
        """Test that inf upper bound is replaced by input_length - 1."""
        iv = Interval([5, np.inf])
        iv.set_interval(input_length=50)
        assert iv.value == (5, 49)

    def test_infinite_interval_without_input_length_raises(self):
        """Test that inf interval without input_length raises ValueError."""
        iv = Interval([5, np.inf])
        with pytest.raises(ValueError, match="Input length is required"):
            iv.set_interval()

    def test_single_point_finite(self):
        """Test set_interval for a single-point interval."""
        iv = Interval([4, 4])
        iv.set_interval()
        assert iv.value == (4, 4)

    def test_value_set_after_call(self):
        """Test that .value attribute is set after calling set_interval."""
        iv = Interval([0, 10])
        assert not hasattr(iv, "value")
        iv.set_interval()
        assert hasattr(iv, "value")
        assert iv.value == (0, 10)


class TestIntervalLen:
    """Test Interval.__len__ method."""

    def test_len_finite_interval_after_set(self):
        """Test length after set_interval on a finite interval."""
        iv = Interval([0, 10])
        iv.set_interval()
        assert len(iv) == 11  # inclusive both ends

    def test_len_finite_interval_before_set(self):
        """Test length before set_interval on a finite interval."""
        iv = Interval([3, 7])
        assert len(iv) == 5

    def test_len_single_point(self):
        """Test length of a single-point interval."""
        iv = Interval([5, 5])
        assert len(iv) == 1

    def test_len_none_interval_raises(self):
        """Test that None interval without set_interval raises ValueError."""
        iv = Interval(None)
        with pytest.raises(ValueError, match="Interval not properly initialized"):
            len(iv)

    def test_len_none_interval_after_set(self):
        """Test length of None interval after set_interval."""
        iv = Interval(None)
        iv.set_interval(input_length=11)
        assert len(iv) == 11  # (0, 10) -> 11 steps

    def test_len_infinite_interval_raises_before_set(self):
        """Test that infinite interval raises before set_interval."""
        iv = Interval([0, np.inf])
        with pytest.raises(ValueError, match="Cannot compute length of infinite interval"):
            len(iv)

    def test_len_infinite_interval_after_set(self):
        """Test length of infinite interval after set_interval."""
        iv = Interval([0, np.inf])
        iv.set_interval(input_length=100)
        assert len(iv) == 100  # (0, 99) -> 100 steps

    def test_len_partial_infinite_after_set(self):
        """Test length when infinite interval starts at non-zero."""
        iv = Interval([5, np.inf])
        iv.set_interval(input_length=100)
        assert len(iv) == 95  # (5, 99) -> 95 steps


class TestIntervalStr:
    """Test Interval.__str__ method."""

    def test_str_finite_interval(self):
        """Test string representation of a finite interval."""
        iv = Interval([0, 10])
        assert str(iv) == "[0,10]"

    def test_str_finite_non_zero_start(self):
        """Test string representation with non-zero start."""
        iv = Interval([3, 7])
        assert str(iv) == "[3,7]"

    def test_str_infinite_interval(self):
        """Test string representation of an infinite interval."""
        iv = Interval([5, np.inf])
        assert str(iv) == "[5,∞]"

    def test_str_none_interval(self):
        """Test string representation when interval is None."""
        iv = Interval(None)
        assert str(iv) == ""

    def test_repr_matches_str(self):
        """Test that repr matches str."""
        iv = Interval([2, 8])
        assert repr(iv) == str(iv)
