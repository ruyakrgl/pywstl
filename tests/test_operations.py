#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the wstlMax and wstlMin operations in the operations module.
"""

import numpy as np
import pytest

from pywstl.operations import wstlMax, wstlMin


class TestWstlMax:
    """Test wstlMax operation class."""

    def setup_method(self):
        self.op = wstlMax()

    def test_max_along_axis0(self):
        """Test max along axis 0."""
        x = np.array([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[2.0, 3.0]]))

    def test_max_along_axis1(self):
        """Test max along axis 1."""
        x = np.array([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=1)
        np.testing.assert_array_equal(result, np.array([[3.0], [2.0]]))

    def test_max_keepdims(self):
        """Test that keepdims=True is applied (dimension is preserved)."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = self.op(x, axis=1)
        assert result.ndim == x.ndim
        assert result.shape == (1, 1)

    def test_max_3d_array(self):
        """Test max on a 3D array."""
        x = np.arange(24, dtype=float).reshape(2, 3, 4)
        result = self.op(x, axis=1)
        expected = np.max(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(result, expected)

    def test_max_single_element(self):
        """Test max on a single-element array."""
        x = np.array([[5.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[5.0]]))

    def test_max_negative_values(self):
        """Test max with all negative values."""
        x = np.array([[-3.0, -1.0], [-4.0, -2.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[-3.0, -1.0]]))

    def test_max_invalid_input_raises(self):
        """Test that non-ndarray input raises AssertionError."""
        with pytest.raises(AssertionError):
            self.op([[1, 2], [3, 4]], axis=0)

    def test_max_invalid_list_raises(self):
        """Test that list input raises AssertionError."""
        with pytest.raises(AssertionError):
            self.op([1.0, 2.0, 3.0], axis=0)

    def test_max_returns_ndarray(self):
        """Test that the result is a numpy array."""
        x = np.array([[1.0, 2.0]])
        result = self.op(x, axis=1)
        assert isinstance(result, np.ndarray)

    def test_max_uniform_values(self):
        """Test max when all values are equal."""
        x = np.full((3, 4), 7.0)
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.full((1, 4), 7.0))

    def test_max_along_last_axis(self):
        """Test max along the last axis of a 4D array."""
        x = np.random.rand(2, 3, 4, 5)
        result = self.op(x, axis=3)
        expected = np.max(x, axis=3, keepdims=True)
        np.testing.assert_array_equal(result, expected)


class TestWstlMin:
    """Test wstlMin operation class."""

    def setup_method(self):
        self.op = wstlMin()

    def test_min_along_axis0(self):
        """Test min along axis 0."""
        x = np.array([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[1.0, 0.0]]))

    def test_min_along_axis1(self):
        """Test min along axis 1."""
        x = np.array([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=1)
        np.testing.assert_array_equal(result, np.array([[1.0], [0.0]]))

    def test_min_keepdims(self):
        """Test that keepdims=True is applied (dimension is preserved)."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = self.op(x, axis=1)
        assert result.ndim == x.ndim
        assert result.shape == (1, 1)

    def test_min_3d_array(self):
        """Test min on a 3D array."""
        x = np.arange(24, dtype=float).reshape(2, 3, 4)
        result = self.op(x, axis=1)
        expected = np.min(x, axis=1, keepdims=True)
        np.testing.assert_array_equal(result, expected)

    def test_min_single_element(self):
        """Test min on a single-element array."""
        x = np.array([[5.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[5.0]]))

    def test_min_negative_values(self):
        """Test min with all negative values."""
        x = np.array([[-3.0, -1.0], [-4.0, -2.0]])
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.array([[-4.0, -2.0]]))

    def test_min_invalid_input_raises(self):
        """Test that non-ndarray input raises AssertionError."""
        with pytest.raises(AssertionError):
            self.op([[1, 2], [3, 4]], axis=0)

    def test_min_invalid_list_raises(self):
        """Test that list input raises AssertionError."""
        with pytest.raises(AssertionError):
            self.op([1.0, 2.0, 3.0], axis=0)

    def test_min_returns_ndarray(self):
        """Test that the result is a numpy array."""
        x = np.array([[1.0, 2.0]])
        result = self.op(x, axis=1)
        assert isinstance(result, np.ndarray)

    def test_min_uniform_values(self):
        """Test min when all values are equal."""
        x = np.full((3, 4), 7.0)
        result = self.op(x, axis=0)
        np.testing.assert_array_equal(result, np.full((1, 4), 7.0))

    def test_min_along_last_axis(self):
        """Test min along the last axis of a 4D array."""
        x = np.random.rand(2, 3, 4, 5)
        result = self.op(x, axis=3)
        expected = np.min(x, axis=3, keepdims=True)
        np.testing.assert_array_equal(result, expected)


class TestWstlMaxVsMin:
    """Test that wstlMax and wstlMin give complementary results."""

    def test_max_geq_min(self):
        """Test that max result is always >= min result for any input."""
        x = np.random.rand(5, 6)
        op_max = wstlMax()
        op_min = wstlMin()
        result_max = op_max(x, axis=0)
        result_min = op_min(x, axis=0)
        assert np.all(result_max >= result_min)

    def test_max_min_same_for_uniform(self):
        """Test max == min when all values are equal."""
        x = np.full((4, 3), 2.5)
        op_max = wstlMax()
        op_min = wstlMin()
        np.testing.assert_array_equal(op_max(x, axis=0), op_min(x, axis=0))
