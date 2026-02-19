#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest

import pywstl
from pywstl import And, Eventually, Always, Or
from pywstl.intervals import Interval
from pywstl.signals import Signal


class TestPredicateRobustness:
    """Test robustness computation for atomic predicates."""

    def setup_method(self):
        """Set up test data for predicates."""
        pywstl.set_backend("torch")
        self.signal_data = np.random.rand(2, 4, 1, 1)
        self.signal = Signal("x", self.signal_data)

        self.comparison = np.random.rand(1)[0]

    def teardown_method(self):
        """Reset backend after each test."""
        pywstl.reset_backend()

    def test_greater_than_robustness(self):
        """Test GreaterThan predicate robustness computation."""
        # Test x >= value
        predicate = self.signal >= self.comparison

        # Expected robustness: signal_value - threshold
        expected = (self.signal_data - self.comparison)[:, 0, 0, 0]

        predicate.set_weights(signals=self.signal)
        computed = predicate.robustness(self.signal)

        np.testing.assert_allclose(np.asarray(computed.flatten()), np.asarray(expected), atol=1e-5)
        assert computed.shape == (
            2,
            1,
        ), "Robustness shape mismatch for Greater Than predicate"

        # assign random weights. Note that predicate does not take any weights.
        # the robustness values should not change.
        predicate.set_weights(signals=self.signal, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = predicate.robustness(self.signal)

        assert computed_random.shape == (
            2,
            1,
        ), "Robustness shape mismatch for Greater Than predicate with random weights"
        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected), atol=1e-5
        )

    def test_less_than_robustness(self):
        """Test LessThan predicate robustness computation."""
        # Test x <= 4.0
        predicate = self.signal <= self.comparison

        # Expected robustness: threshold - signal_value
        expected = (self.comparison - self.signal_data)[:, 0, 0, 0]

        predicate.set_weights(signals=self.signal)
        computed = predicate.robustness(self.signal)

        np.testing.assert_allclose(np.asarray(computed.flatten()), np.asarray(expected), atol=1e-5)
        assert computed.shape == (
            2,
            1,
        ), "Robustness shape mismatch for Less Than predicate"

        # assign random weights. Note that predicate does not take any weights.
        # the robustness values should not change.
        predicate.set_weights(signals=self.signal, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = predicate.robustness(self.signal)

        assert computed_random.shape == (
            2,
            1,
        ), "Robustness shape mismatch for Greater Than predicate with random weights"
        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected), atol=1e-5
        )

    def test_equal_robustness(self):
        """Test Equal predicate robustness computation."""
        # Test x == 3.0
        predicate = self.signal == self.comparison

        # Expected robustness: -|signal_value - target|
        expected = -np.abs(self.signal_data - self.comparison)[:, 0, 0, 0]

        predicate.set_weights(signals=self.signal)
        computed = predicate.robustness(self.signal)

        np.testing.assert_allclose(np.asarray(computed.flatten()), np.asarray(expected), atol=1e-5)
        assert computed.shape == (2, 1), "Robustness shape mismatch for Equal predicate"

        # assign random weights. Note that predicate does not take any weights.
        # the robustness values should not change.
        predicate.set_weights(signals=self.signal, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = predicate.robustness(self.signal)

        assert computed_random.shape == (
            2,
            1,
        ), "Robustness shape mismatch for Greater Than predicate with random weights"
        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected), atol=1e-5
        )


class TestTemporalOperator:
    """Test robustness computation for temporal operators (Always, Eventually)."""

    def setup_method(self):
        """Set up test data for temporal operators."""
        pywstl.set_backend("torch")
        # Create a signal with varying values for testing temporal properties
        self.N = 3
        self.T = 6
        self.Nsamples = 50

        self.signal_data = np.random.rand(self.N, self.T, 1, 1)  # Shape (N, T, 1, 1)
        self.signal = Signal("signal", self.signal_data)

        self.comparison = np.random.rand(1)[0]  # Threshold for temporal predicates

        # Create base predicate for temporal operations
        self.base_predicate = self.signal >= self.comparison

    def teardown_method(self):
        """Reset backend after each test."""
        pywstl.reset_backend()

    def test_always_operator_fixed_interval(self):
        """Test Always operator with fixed interval."""

        # Test Always[0,2] (x >= 1.0)
        always_formula = Always(self.base_predicate, [0, 2])

        # Set uniform weights for testing. i.e. stl robustness
        always_formula.set_weights(signals=self.signal)

        # Test interval setting
        assert always_formula.interval.interval == (0, 2)

    def test_eventually_operator_fixed_interval(self):
        """Test Eventually operator with basic interval."""

        # Test Eventually[0,2] (x >= 1.0)
        eventually_formula = Eventually(self.base_predicate, [0, 2])

        # Set uniform weights for testing
        eventually_formula.set_weights(signals=self.signal)

        # Test interval setting
        assert eventually_formula.interval.interval == (0, 2)

    def test_temporal_without_interval(self):
        """Test temporal operators without specified interval (default behavior)."""
        # Test Always without interval (should default)
        always_formula = Always(self.base_predicate)
        always_formula.set_weights(signals=self.signal)

        assert always_formula.interval.value == (0, self.T - 1)
        assert len(always_formula.interval) == self.T

        # Test Eventually without interval (should default)
        eventually_formula = Eventually(self.base_predicate)
        eventually_formula.set_weights(signals=self.signal)

        assert eventually_formula.interval.value == (0, self.T - 1)
        assert len(eventually_formula.interval) == self.T

    def test_always_weight_setting(self):
        """Test weight setting for temporal operators."""
        always_formula = Always(self.base_predicate, [0, 2])

        # Test uniform weights
        always_formula.set_weights(signals=self.signal)

        # Verify weights are set
        assert always_formula.weights is not None

        # Verify weights are set correctly
        for key in always_formula.weights.keys():
            assert always_formula.weights[key].shape == (
                len(always_formula.interval),
                1,
            ), f"Weight shape mismatch for {key}."
            assert np.all(
                np.asarray(always_formula.weights[key]) == 1
            ), f"Weights for {key} should be 1."

        # Test random weights
        always_formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )

        # Verify weights are set
        assert always_formula.weights is not None

        # Verify weights are set
        for key in always_formula.weights.keys():
            assert always_formula.weights[key].shape == (
                len(always_formula.interval),
                self.Nsamples,
            ), f"Weight shape mismatch for {key}."

        # Verify the range
        for key in always_formula.weights.keys():
            assert np.all(np.asarray(always_formula.weights[key]) >= 0.5) and np.all(
                np.asarray(always_formula.weights[key]) <= 2.0
            ), f"Weights for {key} should be in the range [0.5, 2.0]."

    def test_eventually_weight_setting(self):
        """Test weight setting for Eventually operator."""
        eventually_formula = Eventually(self.base_predicate, [0, 2])

        # Test uniform weights
        eventually_formula.set_weights(signals=self.signal)

        # Verify weights are set
        assert eventually_formula.weights is not None

        # Verify weights are set correctly
        for key in eventually_formula.weights.keys():
            assert eventually_formula.weights[key].shape == (
                len(eventually_formula.interval),
                1,
            ), f"Weight shape mismatch for {key}."
            assert np.all(
                np.asarray(eventually_formula.weights[key]) == 1
            ), f"Weights for {key} should be 1."

        # Test random weights
        eventually_formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )

        # Verify weights are set
        for key in eventually_formula.weights.keys():
            assert eventually_formula.weights[key].shape == (
                len(eventually_formula.interval),
                self.Nsamples,
            ), f"Weight shape mismatch for {key}."

        # Verify the range
        for key in eventually_formula.weights.keys():
            assert np.all(np.asarray(eventually_formula.weights[key]) >= 0.5) and np.all(
                np.asarray(eventually_formula.weights[key]) <= 2.0
            ), f"Weights for {key} should be in the range [0.5, 2.0]."

    def test_nested_temporal_operators(self):
        """Test nested temporal operators."""
        # Create nested formula: Always[0,1] Eventually[0,1] (x >= 1.0)
        inner_eventually = Eventually(self.base_predicate, [0, 3])
        outer_always = Always(inner_eventually, [0, 2])

        # Set weights for both operators
        outer_always.set_weights(signals=self.signal)

        assert outer_always.weights is not None
        assert outer_always.keys == [
            f"signal>={str(self.comparison).replace('.', '_')}",
            f"F[0,3](signal>={str(self.comparison).replace('.', '_')})",
            f"G[0,2](F[0,3](signal>={str(self.comparison).replace('.', '_')}))",
        ]

        outer_always.weights[
            f"G[0,2](F[0,3](signal>={str(self.comparison).replace('.', '_')}))"
        ].shape == (3, 1)
        outer_always.weights[f"F[0,3](signal>={str(self.comparison).replace('.', '_')})"].shape == (
            4,
            1,
        )

        assert outer_always.subformula == inner_eventually
        assert outer_always.subformula.interval.value == (0, 3)
        assert outer_always.interval.value == (0, 2)

        for key in outer_always.weights.keys():
            assert np.all(
                np.asarray(outer_always.weights[key]) == 1
            ), f"Weights for {key} should be 1."

    def test_always_robustness(
        self,
    ):
        """Helper function to compute and verify robustness."""
        # Compute robustness
        formula = Always(self.base_predicate, [0, 2])

        # Test uniform weights
        formula.set_weights(signals=self.signal)

        computed = formula.robustness(self.signal)
        expected = np.min(self.signal_data[:, 0:3, 0, 0] - self.comparison, axis=1)

        assert computed.shape == (
            self.signal_data.shape[0],
            1,
        ), "Robustness shape mismatch for Eventually operator"

        # Verify robustness matches expected values
        np.testing.assert_allclose(np.asarray(computed.flatten()), np.asarray(expected), atol=1e-5)

        # Test robustness with random weights
        formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )
        computed_random = formula.robustness(self.signal)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            self.Nsamples,
        ), "Robustness shape mismatch for Eventually operator with random weights"

        weights = formula.weights[f"G[0,2](signal>={str(self.comparison).replace('.', '_')})"]

        expected_random = np.zeros((self.signal_data.shape[0], self.Nsamples))
        for i in range(self.Nsamples):
            expected_random[:, i] = np.min(
                np.asarray(weights.T[i, :]) * (self.signal_data[:, 0:3, 0, 0] - self.comparison),
                axis=1,
            )

        np.testing.assert_allclose(
            np.asarray(computed_random), np.asarray(expected_random), atol=1e-5
        )

        # Compute robustness no time interval
        formula = Always(self.base_predicate)

        # Test uniform weights
        formula.set_weights(signals=self.signal)

        computed = formula.robustness(self.signal)
        expected = np.min(self.signal_data[:, :, 0, 0] - self.comparison, axis=1)

        assert computed.shape == (
            self.signal_data.shape[0],
            1,
        ), "Robustness shape mismatch for Eventually operator"

        # Verify robustness matches expected values
        np.testing.assert_allclose(np.asarray(computed.flatten()), np.asarray(expected), atol=1e-5)

        # Test robustness with random weights
        formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )
        computed_random = formula.robustness(self.signal)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            self.Nsamples,
        ), "Robustness shape mismatch for Eventually operator with random weights"

        weights = formula.weights[f"G(signal>={str(self.comparison).replace('.', '_')})"]

        expected_random = np.zeros((self.signal_data.shape[0], self.Nsamples))
        for i in range(self.Nsamples):
            expected_random[:, i] = np.min(
                np.asarray(weights.T[i, :]) * (self.signal_data[:, :, 0, 0] - self.comparison),
                axis=1,
            )

        np.testing.assert_allclose(
            np.asarray(computed_random), np.asarray(expected_random), atol=1e-5
        )

    def test_eventually_robustness(self):
        formula = Eventually(self.base_predicate, [0, 2])
        # Test uniform weights
        formula.set_weights(signals=self.signal)

        computed = formula.robustness(self.signal).flatten()
        expected = np.max(self.signal_data[:, 0:3, 0, 0] - self.comparison, axis=1)

        assert computed.shape == (
            self.signal_data.shape[0],
        ), "Robustness shape mismatch for Eventually operator"

        # Verify robustness matches expected values
        np.testing.assert_allclose(np.asarray(computed), np.asarray(expected), atol=1e-5)

        # Test robustness with random weights
        formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )
        computed_random = formula.robustness(self.signal)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            self.Nsamples,
        ), "Robustness shape mismatch for Eventually operator with random weights"

        weights = formula.weights[f"F[0,2](signal>={str(self.comparison).replace('.', '_')})"]

        expected_random = np.zeros((self.signal_data.shape[0], self.Nsamples))
        for i in range(self.Nsamples):
            expected_random[:, i] = np.max(
                np.asarray(weights.T[i, :]) * (self.signal_data[:, 0:3, 0, 0] - self.comparison),
                axis=1,
            )

        np.testing.assert_allclose(
            np.asarray(computed_random), np.asarray(expected_random), atol=1e-5
        )

        formula = Eventually(self.base_predicate)
        # Test uniform weights
        formula.set_weights(signals=self.signal)

        computed = formula.robustness(self.signal).flatten()
        expected = np.max(self.signal_data[:, :, 0, 0] - self.comparison, axis=1)

        assert computed.shape == (
            self.signal_data.shape[0],
        ), "Robustness shape mismatch for Eventually operator"

        # Verify robustness matches expected values
        np.testing.assert_allclose(np.asarray(computed), np.asarray(expected), atol=1e-5)

        # Test robustness with random weights
        formula.set_weights(
            signals=self.signal,
            w_range=[0.5, 2.0],
            no_samples=self.Nsamples,
            random=True,
        )
        computed_random = formula.robustness(self.signal)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            self.Nsamples,
        ), "Robustness shape mismatch for Eventually operator with random weights"

        weights = formula.weights[f"F(signal>={str(self.comparison).replace('.', '_')})"]

        expected_random = np.zeros((self.signal_data.shape[0], self.Nsamples))
        for i in range(self.Nsamples):
            expected_random[:, i] = np.max(
                np.asarray(weights.T[i, :]) * (self.signal_data[:, :, 0, 0] - self.comparison),
                axis=1,
            )

        np.testing.assert_allclose(
            np.asarray(computed_random), np.asarray(expected_random), atol=1e-5
        )

    def test_nested_temporal_robustness(self):
        """Test robustness for nested temporal operators."""
        # Create nested formula: Always[0,1] Eventually[0,1] (x >= 1.0)
        inner_eventually = Eventually(self.base_predicate, [0, 2])
        outer_always = Always(inner_eventually, [0, 3])

        # Set weights for both operators
        outer_always.set_weights(signals=self.signal)
        computed = outer_always.robustness(self.signal).flatten()

        intermediate = np.zeros((self.signal_data.shape[0], 4))
        for i in range(4):
            intermediate[:, i] = np.max(
                self.signal_data[:, i : i + 3, 0, 0] - self.comparison, axis=1
            )

        expected = np.min(intermediate, axis=1)

        # Verify robustness matches expected values
        np.testing.assert_allclose(np.asarray(computed), np.asarray(expected), atol=1e-5)


class TestLogicalOperatorRobustness:
    """Test robustness computation for logical operators (And, Or)."""

    def setup_method(self):
        """Set up test data for logical operators."""
        pywstl.set_backend("torch")
        self.signal1 = Signal("x1", np.random.rand(2, 3, 1, 1))
        self.signal2 = Signal("x2", np.random.rand(2, 3, 1, 1))

        self.comparison1 = np.random.rand(1)[0]  # Threshold for signal1
        self.comparison2 = np.random.rand(1)[0]  # Threshold for signal2

        # Create predicates for logical operations
        self.predicate1 = self.signal1 >= self.comparison1
        self.predicate2 = self.signal2 <= self.comparison2

        self.signals = (self.signal1, self.signal2)

    def teardown_method(self):
        """Reset backend after each test."""
        pywstl.reset_backend()

        self.signals = (self.signal1, self.signal2)

    def test_and_operator_basic(self):
        """Test And operator basic functionality."""
        and_formula = And(self.predicate1, self.predicate2)

        # Set weights
        and_formula.set_weights(self.signals)

        # Verify subformulas
        assert and_formula.subformula1 == self.predicate1
        assert and_formula.subformula2 == self.predicate2

        and_formula.set_weights(signals=self.signals)
        for key in and_formula.weights.keys():
            assert and_formula.weights[key].shape == (
                2,
                1,
            ), f"Weight shape mismatch for {key}."
            assert np.all(
                np.asarray(and_formula.weights[key]) == 1
            ), f"Weights for {key} should be 1."

        and_formula.set_weights(
            signals=self.signals, w_range=[0.5, 2.0], no_samples=10, random=True
        )
        # Verify weights are set
        for key in and_formula.weights.keys():
            assert and_formula.weights[key].shape == (
                2,
                10,
            ), f"Weight shape mismatch for {key}."

    def test_or_operator_basic(self):
        """Test Or operator basic functionality."""
        or_formula = Or(self.predicate1, self.predicate2)

        # Set weights
        or_formula.set_weights(self.signals)

        # Verify subformulas
        assert or_formula.subformula1 == self.predicate1
        assert or_formula.subformula2 == self.predicate2

        or_formula.set_weights(signals=self.signals)
        for key in or_formula.weights.keys():
            assert or_formula.weights[key].shape == (
                2,
                1,
            ), f"Weight shape mismatch for {key}."
            assert np.all(
                np.asarray(or_formula.weights[key]) == 1
            ), f"Weights for {key} should be 1."

        or_formula.set_weights(signals=self.signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        # Verify weights are set
        for key in or_formula.weights.keys():
            assert or_formula.weights[key].shape == (
                2,
                10,
            ), f"Weight shape mismatch for {key}."

    def test_and_robustness(self):
        """Test robustness computation for And operator."""
        and_formula = And(self.predicate1, self.predicate2)

        # Set weights
        and_formula.set_weights(self.signals)

        computed = and_formula.robustness(self.signals)

        values = np.concatenate(
            (
                (self.signal1.value[:, 0, 0, 0] - self.comparison1).reshape(-1, 1),
                (self.comparison2 - self.signal2.value[:, 0, 0, 0]).reshape(-1, 1),
            ),
            axis=1,
        )
        expected = np.min(values, axis=1)

        assert computed.shape == (
            self.signal1.value.shape[0],
            1,
        ), "Robustness shape mismatch for And operator"

        np.testing.assert_allclose(
            np.asarray(computed.flatten()), np.asarray(expected.flatten()), atol=1e-5
        )

        # random weights
        and_formula.set_weights(self.signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = and_formula.robustness(self.signals)
        assert computed_random.shape == (
            self.signal1.value.shape[0],
            10,
        ), "Robustness shape mismatch for And operator with random weights"

        expected_random = np.zeros((self.signal1.value.shape[0], 10))

        for i in range(10):
            w = and_formula.weights[
                f"(x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(self.comparison2).replace('.', '_')})"
            ].T[i, :]
            expected_random[:, i] = np.min(np.asarray(w) * values, axis=1)

        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected_random.flatten()), atol=1e-5
        )

    def test_or_robustness(self):
        """Test robustness computation for Or operator."""
        or_formula = Or(self.predicate1, self.predicate2)

        # Set weights
        or_formula.set_weights(self.signals)

        computed = or_formula.robustness(self.signals)

        values = np.concatenate(
            (
                (self.signal1.value[:, 0, 0, 0] - self.comparison1).reshape(-1, 1),
                (self.comparison2 - self.signal2.value[:, 0, 0, 0]).reshape(-1, 1),
            ),
            axis=1,
        )
        expected = np.max(values, axis=1)

        assert computed.shape == (
            self.signal1.value.shape[0],
            1,
        ), "Robustness shape mismatch for Or operator"

        np.testing.assert_allclose(
            np.asarray(computed.flatten()), np.asarray(expected.flatten()), atol=1e-5
        )

        # random weights
        or_formula.set_weights(self.signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = or_formula.robustness(self.signals)
        assert computed_random.shape == (
            self.signal1.value.shape[0],
            10,
        ), "Robustness shape mismatch for Or operator with random weights"

        expected_random = np.zeros((self.signal1.value.shape[0], 10))
        for i in range(10):
            w = or_formula.weights[
                f"(x1>={str(self.comparison1).replace('.', '_')})or(x2<={str(self.comparison2).replace('.', '_')})"
            ].T[i, :]
            expected_random[:, i] = np.max(np.asarray(w) * values, axis=1)

        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected_random.flatten()), atol=1e-5
        )


class TestCombinedOperators:
    def setup_method(self):
        """Set up test data for combined operators."""
        pywstl.set_backend("torch")
        self.signal_data = np.round(np.random.rand(2, 5, 1, 2), 2)  # Shape (T, K, 1, 1)
        self.signal1 = Signal("x1", self.signal_data[:, :, :, 0][:, :, :, None])
        self.signal2 = Signal("x2", self.signal_data[:, :, :, 1][:, :, :, None])

        self.comparison1 = np.round(np.random.rand(1)[0], 2)  # Threshold for signal1
        self.comparison2 = np.round(np.random.rand(1)[0], 2)  # Threshold for signal2

        # Create predicates for logical operations
        self.predicate1 = self.signal1 >= self.comparison1
        self.predicate2 = self.signal2 <= 2 * self.comparison2

    def teardown_method(self):
        """Reset backend after each test."""
        pywstl.reset_backend()

    def test_combined_and_or(self):
        """Test robustness computation for combined And and Or operators."""
        # Create a formula: (x1 >= 1.0) and (x2 <= 2.0) or (x1 <= 0.5)
        and_formula = And(self.predicate1, self.predicate2)
        pred3 = self.signal1 <= 0.5
        or_formula = Or(and_formula, pred3)

        # Set weights
        signals = ((self.signal1, self.signal2), self.signal1)
        or_formula.set_weights(signals)

        # Verify subformulas
        assert or_formula.subformula1 == and_formula
        assert or_formula.subformula2 == pred3

        computed = or_formula.robustness(signals)

        values = np.concatenate(
            (
                (self.signal1.value[:, 0, 0, 0] - self.comparison1).reshape(-1, 1),
                (2 * self.comparison2 - self.signal2.value[:, 0, 0, 0]).reshape(-1, 1),
            ),
            axis=1,
        )

        expected_and = np.min(values, axis=1)
        expected_or = np.max(
            np.column_stack((expected_and, 0.5 - self.signal1.value[:, 0, 0, 0])),
            axis=1,
        )

        assert computed.shape == (
            self.signal_data.shape[0],
            1,
        ), "Robustness shape mismatch for combined And/Or operator"

        np.testing.assert_allclose(
            np.asarray(computed.flatten()), np.asarray(expected_or.flatten()), atol=1e-5
        )

        # Random weights
        or_formula.set_weights(signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = or_formula.robustness(signals)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            10,
        ), "Robustness shape mismatch for combined And/Or operator with random weights"
        expected_random = np.zeros((self.signal_data.shape[0], 10))
        for i in range(10):
            w_and = or_formula.weights[
                f"(x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(2*self.comparison2).replace('.', '_')})"
            ].T[i, :]
            w_or = or_formula.weights[
                f"((x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(2*self.comparison2).replace('.', '_')}))or(x1<=0_5)"
            ].T[i, :]

            inner_value = np.min(np.asarray(w_and) * values, axis=1)
            outer_value = np.column_stack((inner_value, (0.5 - self.signal1.value[:, 0, 0, 0])))

            expected_random[:, i] = np.max(np.asarray(w_or) * outer_value, axis=1)

        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected_random.flatten()), atol=1e-5
        )

    def test_nested_temporal_logic_robustness(self):
        """Test robustness for nested temporal logic operators."""
        # Create nested formula: Always[0,1] Eventually[0,1] (x >= 1.0)

        inner_eventually = Eventually(self.predicate2, [2, np.inf])
        inner_always = Always(self.predicate1)

        formula = And(inner_always, inner_eventually)
        signals = (self.signal1, self.signal2)

        formula.set_weights(signals)

        # Verify subformulas
        assert formula.subformula1 == inner_always
        assert formula.subformula2 == inner_eventually

        # Verify interval
        assert formula.subformula1.interval.value == (0, 4)
        assert formula.subformula2.interval.value == (2, 4)

        # Set weights for both operators
        computed = formula.robustness(signals)

        expected = np.zeros((self.signal_data.shape[0], 1))
        for i in range(self.signal_data.shape[0]):
            # Compute robustness for inner always
            always_robustness = np.min(self.signal_data[i, 0:5, 0, 0] - self.comparison1)
            # Compute robustness for inner eventually
            eventually_robustness = np.max(2 * self.comparison2 - self.signal_data[i, 2:5, 0, 1])
            expected[i] = min(always_robustness, eventually_robustness)

        np.testing.assert_allclose(np.asarray(computed), np.asarray(expected), atol=1e-5)

        # Random weights
        formula.set_weights(signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = formula.robustness(signals)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            10,
        ), "Robustness shape mismatch for nested temporal logic operator with random weights"
        expected_random = np.zeros((self.signal_data.shape[0], 10))
        for i in range(10):
            w_always = formula.weights[f"G(x1>={str(self.comparison1).replace('.', '_')})"].T[i, :]
            w_eventually = formula.weights[
                f"F[2,∞](x2<={str(2*self.comparison2).replace('.', '_')})"
            ].T[i, :]
            w_and = formula.weights[
                f"(G(x1>={str(self.comparison1).replace('.', '_')}))and(F[2,∞](x2<={str(2*self.comparison2).replace('.', '_')}))"
            ].T[i, :]

            inner_value = np.min(
                np.asarray(w_always) * (self.signal_data[:, 0:5, 0, 0] - self.comparison1), axis=1
            )
            outer_value = np.max(
                np.asarray(w_eventually) * (2 * self.comparison2 - self.signal_data[:, 2:5, 0, 1]),
                axis=1,
            )

            and_value = np.column_stack((inner_value, outer_value))
            expected_random[:, i] = np.min(np.asarray(w_and) * and_value, axis=1)

        np.testing.assert_allclose(
            np.asarray(computed_random.flatten()), np.asarray(expected_random.flatten()), atol=1e-5
        )

    def test_logic_temporal_robustness(self):
        inner_and = And(self.predicate1, self.predicate2)
        inner_eventually = Eventually(inner_and, [0, 2])

        formula = Always(inner_eventually)

        signals = (self.signal1, self.signal2)

        formula.set_weights(signals)

        # Verify subformulas
        assert formula.subformula == inner_eventually
        assert formula.subformula.subformula == inner_and
        assert formula.subformula.subformula.subformula1 == self.predicate1
        assert formula.subformula.subformula.subformula2 == self.predicate2

        # Verify interval
        assert formula.interval.value == (0, 4)
        assert formula.subformula.interval.value == (0, 2)

        computed = formula.robustness(signals)
        expected = np.zeros((self.signal_data.shape[0], 1))

        for i in range(self.signal_data.shape[0]):
            eventually_robustness = np.zeros((5,))

            for t in range(5):
                # Compute robustness for inner eventually
                mint = np.minimum(t + 3, self.signal_data.shape[1])
                inters = np.zeros((mint - t,))
                for tt in range(t, mint):
                    inters[tt - t] = np.min(
                        (
                            self.signal_data[i, tt, 0, 0] - self.comparison1,
                            2 * self.comparison2 - self.signal_data[i, tt, 0, 1],
                        ),
                    )

                eventually_robustness[t] = np.max(inters)

            expected[i] = np.min(eventually_robustness)

        np.testing.assert_allclose(np.asarray(computed), np.asarray(expected), atol=1e-5)

        # random weights
        formula.set_weights(signals, w_range=[0.5, 2.0], no_samples=10, random=True)
        computed_random = formula.robustness(signals)
        assert computed_random.shape == (
            self.signal_data.shape[0],
            10,
        ), "Robustness shape mismatch for logic temporal operator with random weights"
        expected_random = np.zeros((self.signal_data.shape[0], 10))

        for i in range(10):
            w_eventually = formula.weights[
                f"F[0,2]((x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(2*self.comparison2).replace('.', '_')}))"
            ].T[i, :]
            w_always = formula.weights[
                f"G(F[0,2]((x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(2*self.comparison2).replace('.', '_')})))"
            ].T[i, :]

            w_and = formula.weights[
                f"(x1>={str(self.comparison1).replace('.', '_')})and(x2<={str(2*self.comparison2).replace('.', '_')})"
            ].T[i, :]

            for s in range(self.signal_data.shape[0]):
                eventually_robustness = np.zeros((5,))
                for t in range(5):
                    mint = np.minimum(t + 3, self.signal_data.shape[1])
                    inters = np.zeros((mint - t,))
                    for tt in range(t, mint):
                        inters[tt - t] = np.min(
                            np.asarray(w_and)
                            * [
                                self.signal_data[s, tt, 0, 0] - self.comparison1,
                                2 * self.comparison2 - self.signal_data[s, tt, 0, 1],
                            ]
                        )

                    eventually_robustness[t] = np.max(np.asarray(w_eventually)[: mint - t] * inters)

                expected_random[s, i] = np.min(np.asarray(w_always) * eventually_robustness)

        np.testing.assert_allclose(
            np.asarray(computed_random), np.asarray(expected_random), atol=1e-5
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test data for edge cases."""
        pywstl.set_backend("torch")
        self.signal_data = np.array([[0.0, 1.0, -1.0, 2.0]]).reshape(
            4, 1, 1, 1
        )  # Shape (T, K, 1, 1)
        self.signal = Signal("edge_test", self.signal_data)

    def teardown_method(self):
        """Reset backend after each test."""
        pywstl.reset_backend()

    def test_same_predicates(self):
        """Test robustness computation with zero values."""
        predicate = self.signal >= 0.0
        formula = Or(predicate, predicate)
        signals = (self.signal, self.signal)

        formula.set_weights(signals)
        robustness = formula.robustness(signals).flatten()

        expected = np.array([0.0, 1.0, -1.0, 2.0])

        np.testing.assert_allclose(np.asarray(robustness), np.asarray(expected), atol=1e-5)

        formula = And(predicate, predicate)
        signals = (self.signal, self.signal)

        formula.set_weights(signals)
        robustness = formula.robustness(signals).flatten()

        np.testing.assert_allclose(np.asarray(robustness), np.asarray(expected), atol=1e-5)

    def test_negative_values(self):
        """Test robustness computation with negative values."""
        predicate = self.signal <= -0.5
        predicate.set_weights(signals=self.signal)
        robustness = predicate.robustness(self.signal).flatten()
        expected = np.array([-0.5, -1.5, 0.5, -2.5])
        np.testing.assert_allclose(np.asarray(robustness), np.asarray(expected), atol=1e-5)

    def test_empty_intervals(self):
        """Test temporal operators with minimal intervals."""
        predicate = self.signal >= 0.5

        # Test minimal interval [0, 0]
        always_minimal = Always(predicate, [0, 0])
        always_minimal.set_weights(signals=self.signal)
        assert always_minimal.interval.value == (0, 0)

        # Test single point interval [1, 1]
        eventually_single = Eventually(predicate, [1, 1])
        eventually_single.set_weights(signals=self.signal)
        assert eventually_single.interval.value == (1, 1)


class TestIntervalClass:
    """Test the Interval class functionality."""

    def test_interval_creation(self):
        """Test Interval object creation."""
        # Test valid intervals
        interval1 = Interval([0, 5])
        assert interval1.interval == (0, 5)

        interval2 = Interval((1, 3))
        assert interval2.interval == (1, 3)

        # Test None interval
        interval_none = Interval(None)
        assert interval_none.interval is None

    def test_interval_validation(self):
        """Test interval validation."""
        # Test invalid interval (start > end)
        with pytest.raises(ValueError):
            Interval([5, 2])

        # Test invalid format
        with pytest.raises(ValueError):
            Interval([1, 2, 3])  # Too many elements

        with pytest.raises(ValueError):
            Interval("invalid")  # Wrong type

        with pytest.raises(ValueError):
            Interval([1])  # Not a valid interval

    def test_interval_setting(self):
        """Test interval value setting."""

        # Test setting a fixed interval with valid input length
        interval = Interval([0, 2])
        interval.set_interval(input_length=10)
        assert interval.value == (0, 2)

        # Test None interval with input length
        interval_none = Interval(None)
        interval_none.set_interval(input_length=5)
        assert interval_none.value == (0, 4)
        assert len(interval_none) == 5

        interval = Interval([2, np.inf])
        interval.set_interval(input_length=10)
        assert interval.value == (2, 9)  # Should use full length

        with pytest.raises(ValueError):
            interval_invalid = Interval(None)
            interval_invalid.set_interval()

        with pytest.raises(ValueError):
            interval_invalid = Interval([2, np.inf])
            interval_invalid.set_interval()

        with pytest.raises(ValueError):
            interval_invalid = Interval([5, 2])
            interval_invalid.set_interval(input_length=10)

    def test_interval_length(self):
        """Test interval length computation."""
        interval = Interval([0, 3])
        assert len(interval) == 4

        interval_inf = Interval([2, np.inf])
        interval_inf.set_interval(input_length=10)
        assert len(interval_inf) == 8

        interval_none = Interval(None)
        with pytest.raises(ValueError):
            len(interval_none)

        interval_inf = Interval([2, np.inf])
        with pytest.raises(ValueError):
            len(interval_inf)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
