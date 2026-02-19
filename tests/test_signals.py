#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the Signal class in the wstlgp module.
"""

import numpy as np
import pytest
import pywstl

from pywstl.signals import Signal
import torch


class TestSignal:
    """Test Signal class functionality."""

    def setup_method(self):
        """Set up test data."""
        self.signal_data = np.random.rand(6, 2)  # Random signal data for testing
        self.signal = Signal("test_signal", self.signal_data)

    def test_signal_creation(self):
        """Test Signal object creation."""
        assert self.signal.name == "test_signal"
        np.testing.assert_array_equal(self.signal.value, self.signal_data)

    def test_signal_arithmetic(self):
        """Test Signal arithmetic operations."""
        add_expected = self.signal_data[:, 0] + self.signal_data[:, 1]
        sub_expected = self.signal_data[:, 0] - self.signal_data[:, 1]
        mul_expected = self.signal_data[:, 0] * 2.0

        signal1 = Signal("s1", self.signal_data[:, 0])
        signal2 = Signal("s2", self.signal_data[:, 1])

        # Addition
        result = signal1 + signal2
        np.testing.assert_array_equal(result.value, add_expected)
        assert result.name == "s1+s2"

        added = np.random.rand(6)
        result = signal1 + added
        np.testing.assert_array_equal(result.value, signal1.value + added)
        assert result.name == "s1+"

        result = added + signal1
        np.testing.assert_array_equal(result.value, signal1.value + added)
        assert result.name == "s1+"

        with pytest.raises(TypeError):
            result = signal1 + "string"

        with pytest.raises(ValueError):
            result = signal1 + np.random.rand(5, 2)

        # Subtraction
        result = signal1 - signal2
        np.testing.assert_array_equal(result.value, sub_expected)
        assert result.name == "s1-s2"

        result = signal1 - added
        np.testing.assert_array_equal(result.value, signal1.value - added)
        assert result.name == "s1-"

        result = added - signal1
        np.testing.assert_array_equal(result.value, added - signal1.value)
        assert result.name == "s1-"

        with pytest.raises(TypeError):
            result = signal1 - "string"

        with pytest.raises(ValueError):
            result = signal1 - np.random.rand(5, 2)

        result = -signal1
        np.testing.assert_array_equal(result.value, -signal1.value)
        assert result.name == "-s1"

        # Multiplication
        result = 2.0 * signal1
        np.testing.assert_array_equal(result.value, mul_expected)
        assert result.name == "s1*"

        result = signal1 * 2.0
        np.testing.assert_array_equal(result.value, signal1.value * 2.0)
        assert result.name == "s1*"

        result = signal1 * signal2
        np.testing.assert_array_equal(result.value, signal1.value * signal2.value)
        assert result.name == "s1*s2"

        result = signal1 / signal1
        np.testing.assert_array_equal(result.value, signal1.value / signal1.value)
        assert result.name == "s1/s1"

        result = signal1 / 2.0
        np.testing.assert_array_equal(result.value, signal1.value / 2.0)
        assert result.name == "s1/2.0"

    def test_signal_negation(self):
        """Test Signal negation."""
        signal = Signal("test", self.signal_data[:, 0])
        neg_signal = -signal
        np.testing.assert_array_equal(neg_signal.value, -self.signal_data[:, 0])

    def test_signal_comparison(self):
        """Test Signal comparison operations."""
        signal1 = Signal("s1", self.signal_data[:, 0].reshape(1, -1, 1, 1))
        comparison_value = np.random.rand(1)[0]

        # Greater than
        result = signal1 >= comparison_value
        for t in range(self.signal_data.shape[1]):
            value = result.robustness(signal1, t).flatten()[0]
            np.testing.assert_array_equal(value, self.signal_data[t, 0] - comparison_value)

        # Less than
        result = signal1 <= comparison_value
        for t in range(self.signal_data.shape[1]):
            value = result.robustness(signal1, t).flatten()[0]
            np.testing.assert_array_equal(value, comparison_value - self.signal_data[t, 0])

        # Equal to
        result = signal1 == comparison_value
        for t in range(self.signal_data.shape[1]):
            value = result.robustness(signal1, t).flatten()[0]
            np.testing.assert_array_equal(value, -np.abs(self.signal_data[t, 0] - comparison_value))

    def test_signal_errors(self):
        """Test Signal error handling."""
        with pytest.raises(TypeError):
            Signal(name=5, value=np.array([1, 2, 3]))  # Invalid name type

        with pytest.raises(TypeError):
            Signal("invalid", "not an array")

        with pytest.raises(TypeError):
            self.signal.set_name(5)

        with pytest.raises(TypeError):
            self.signal.set_value("not an array")

    def test_set_new_name(self):
        self.signal.set_name("new_name")
        assert self.signal.name == "new_name"

    def test_set_new_value(self):
        self.signal.set_value(np.array([4, 5, 6]))
        np.testing.assert_array_equal(self.signal.value, np.array([4, 5, 6]))


class TestSignalwithTorch:
    """Test Signal class functionality with PyTorch backend."""

    def teardown_method(self):
        """Reset backend to numpy after each test to avoid polluting other tests."""
        pywstl.reset_backend()

    def test_signal_creation(self):
        """Test Signal object creation."""
        pywstl.set_backend("torch")  # Ensure we're using the torch backend
        self.signal_data = torch.rand(6, 2)  # Random signal data for testing
        self.signal = Signal("test_signal", self.signal_data)
        assert self.signal.name == "test_signal"
        assert torch.equal(self.signal.value, self.signal_data)

    def test_array_data_type(self):
        """Test that the signal value is a torch tensor."""
        pywstl.set_backend("torch")  # Ensure we're using the torch backend
        self.signal_data = np.random.rand(6, 2)  # Random signal data for testing
        self.signal = Signal("test_signal", self.signal_data)
        assert isinstance(self.signal.value, torch.Tensor)


class TestSignalwithNumpy:
    def teardown_method(self):
        """Reset backend to numpy after each test to avoid polluting other tests."""
        pywstl.reset_backend()

    def test_signal_creation(self):
        """Test Signal object creation."""
        pywstl.set_backend("torch")  # Ensure we're using the torch backend
        self.signal_data = np.random.rand(6, 2)  # Random signal data for testing
        self.signal = Signal("test_signal", self.signal_data)
        assert self.signal.name == "test_signal"
        np.array_equal(self.signal.value, self.signal_data)

    def test_array_data_type(self):
        """Test that torch tensor input is converted to numpy array when using numpy backend."""
        # Backend is numpy (default, reset by teardown_method)
        self.signal_data = torch.rand(6, 2)  # Random signal data for testing
        self.signal = Signal("test_signal", self.signal_data)
        assert isinstance(self.signal.value, np.ndarray)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
