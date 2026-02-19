#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the wstltorch module.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pywstl.signals import Signal  # noqa: E402
from pywstl.operations import wstlMaxTorch, wstlMinTorch  # noqa: E402
from pywstl.wstltorch import (  # noqa: E402
    AlwaysTorch,
    EventuallyTorch,
    AndTorch,
    OrTorch,
    NegationTorch,
)
from pywstl.signals import LessThanTorch, GreaterThanTorch, EqualTorch  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_x():
    """1-batch signal with 5 time steps: [3, 1, 4, 2, 5]."""
    return Signal("x", np.array([[3.0, 1.0, 4.0, 2.0, 5.0]]))


@pytest.fixture
def signal_y():
    """1-batch signal with 3 time steps: [2, 4, 0]."""
    return Signal("y", np.array([[2.0, 4.0, 0.0]]))


@pytest.fixture
def signal_x3():
    """1-batch signal with 3 time steps: [3, 1, 4]."""
    return Signal("x", np.array([[3.0, 1.0, 4.0]]))


@pytest.fixture
def pred_gt(signal_x):
    """GreaterThanTorch: x >= 2.0  ->  trace: [1, -1, 2, 0, 3]."""
    return GreaterThanTorch(signal_x, 2.0)


@pytest.fixture
def pred_lt(signal_x):
    """LessThanTorch: x <= 2.0  ->  trace: [-1, 1, -2, 0, -3]."""
    return LessThanTorch(signal_x, 2.0)


# ---------------------------------------------------------------------------
# Torch operations
# ---------------------------------------------------------------------------


class TestWstlMaxTorch:
    """Test wstlMaxTorch callable class."""

    def setup_method(self):
        self.op = wstlMaxTorch()

    def test_max_along_axis0(self):
        x = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=0)
        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0]]))

    def test_max_along_axis1(self):
        x = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=1)
        torch.testing.assert_close(result, torch.tensor([[3.0], [2.0]]))

    def test_max_keepdims(self):
        """Result should have same number of dimensions as input."""
        x = torch.rand(2, 3)
        result = self.op(x, axis=1)
        assert result.ndim == x.ndim
        assert result.shape == (2, 1)

    def test_max_returns_tensor(self):
        x = torch.rand(3, 4)
        result = self.op(x, axis=0)
        assert isinstance(result, torch.Tensor)

    def test_max_invalid_input_raises(self):
        """numpy arrays are rejected."""
        with pytest.raises(AssertionError):
            self.op(np.array([[1.0, 2.0]]), axis=0)

    def test_max_invalid_list_raises(self):
        with pytest.raises(AssertionError):
            self.op([[1.0, 2.0]], axis=0)

    def test_max_4d_array(self):
        x = torch.rand(2, 3, 4, 5)
        result = self.op(x, axis=2)
        expected = torch.max(x, dim=2, keepdim=True)[0]
        torch.testing.assert_close(result, expected)

    def test_max_single_element(self):
        x = torch.tensor([[5.0]])
        result = self.op(x, axis=0)
        torch.testing.assert_close(result, torch.tensor([[5.0]]))


class TestWstlMinTorch:
    """Test wstlMinTorch callable class."""

    def setup_method(self):
        self.op = wstlMinTorch()

    def test_min_along_axis0(self):
        x = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=0)
        torch.testing.assert_close(result, torch.tensor([[1.0, 0.0]]))

    def test_min_along_axis1(self):
        x = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
        result = self.op(x, axis=1)
        torch.testing.assert_close(result, torch.tensor([[1.0], [0.0]]))

    def test_min_keepdims(self):
        x = torch.rand(2, 3)
        result = self.op(x, axis=1)
        assert result.ndim == x.ndim
        assert result.shape == (2, 1)

    def test_min_returns_tensor(self):
        x = torch.rand(3, 4)
        result = self.op(x, axis=0)
        assert isinstance(result, torch.Tensor)

    def test_min_invalid_input_raises(self):
        with pytest.raises(AssertionError):
            self.op(np.array([[1.0, 2.0]]), axis=0)

    def test_min_invalid_list_raises(self):
        with pytest.raises(AssertionError):
            self.op([[1.0, 2.0]], axis=0)

    def test_min_4d_array(self):
        x = torch.rand(2, 3, 4, 5)
        result = self.op(x, axis=2)
        expected = torch.min(x, dim=2, keepdim=True)[0]
        torch.testing.assert_close(result, expected)

    def test_min_leq_max_for_any_input(self):
        x = torch.rand(4, 5)
        min_op, max_op = wstlMinTorch(), wstlMaxTorch()
        assert torch.all(min_op(x, 0) <= max_op(x, 0))


# ---------------------------------------------------------------------------
# Predicate classes
# ---------------------------------------------------------------------------


class TestGreaterThanTorch:
    """Test GreaterThanTorch predicate."""

    def test_construction(self, signal_x):
        pred = GreaterThanTorch(signal_x, 2.0)
        assert pred.val == 2.0
        assert pred.comparison_op == ">="
        assert pred.signal is signal_x

    def test_key_name(self, signal_x):
        pred = GreaterThanTorch(signal_x, 2.0)
        assert pred.key == "x>=2_0"

    def test_robustness_values(self, signal_x):
        """Robustness should be trace - val at each time step."""
        pred = GreaterThanTorch(signal_x, 2.0)
        # x = [3, 1, 4, 2, 5], val=2.0 -> expected: [1, -1, 2, 0, 3]
        for t, expected in enumerate([1.0, -1.0, 2.0, 0.0, 3.0]):
            rob = pred.robustness(signal_x, t=t)
            assert isinstance(rob, torch.Tensor)
            torch.testing.assert_close(rob.flatten()[0], torch.tensor(expected))

    def test_robustness_output_type(self, signal_x):
        pred = GreaterThanTorch(signal_x, 2.0)
        rob = pred.robustness(signal_x, t=0)
        assert isinstance(rob, torch.Tensor)

    def test_invalid_signal_type_raises(self):
        with pytest.raises(TypeError):
            GreaterThanTorch("not_a_signal", 2.0)

    def test_no_weights_stored(self, signal_x):
        """Predicates do not store any weights."""
        pred = GreaterThanTorch(signal_x, 2.0)
        pred.set_weights(signal_x)
        assert len(pred.weights) == 0

    def test_str_representation(self, signal_x):
        pred = GreaterThanTorch(signal_x, 2.0)
        assert "x" in str(pred)
        assert ">=" in str(pred)
        assert "2.0" in str(pred)


class TestLessThanTorch:
    """Test LessThanTorch predicate."""

    def test_construction(self, signal_x):
        pred = LessThanTorch(signal_x, 2.0)
        assert pred.val == 2.0
        assert pred.comparison_op == "<="

    def test_key_name(self, signal_x):
        pred = LessThanTorch(signal_x, 2.0)
        assert pred.key == "x<=2_0"

    def test_robustness_values(self, signal_x):
        """Robustness should be val - trace at each time step."""
        pred = LessThanTorch(signal_x, 2.0)
        # x = [3, 1, 4, 2, 5], val=2.0 -> expected: [-1, 1, -2, 0, -3]
        for t, expected in enumerate([-1.0, 1.0, -2.0, 0.0, -3.0]):
            rob = pred.robustness(signal_x, t=t)
            torch.testing.assert_close(rob.flatten()[0], torch.tensor(expected))

    def test_invalid_signal_type_raises(self):
        with pytest.raises(TypeError):
            LessThanTorch(42, 2.0)


class TestEqualTorch:
    """Test EqualTorch predicate."""

    def test_construction(self, signal_x):
        pred = EqualTorch(signal_x, 2.0)
        assert pred.val == 2.0
        assert pred.comparison_op == "=="

    def test_key_name(self, signal_x):
        pred = EqualTorch(signal_x, 2.0)
        assert pred.key == "x==2_0"

    def test_robustness_values(self, signal_x):
        """Robustness should be -|trace - val| at each time step."""
        pred = EqualTorch(signal_x, 2.0)
        # x = [3, 1, 4, 2, 5], val=2.0 -> expected: [-1, -1, -2, 0, -3]
        for t, expected in enumerate([-1.0, -1.0, -2.0, 0.0, -3.0]):
            rob = pred.robustness(signal_x, t=t)
            torch.testing.assert_close(rob.flatten()[0], torch.tensor(expected))

    def test_invalid_signal_type_raises(self):
        with pytest.raises(TypeError):
            EqualTorch(None, 2.0)


class TestPredicateTorchWeights:
    """Shared predicate weight behaviour."""

    def test_set_weights_is_noop(self, signal_x):
        """All predicate types ignore set_weights."""
        for cls in [GreaterThanTorch, LessThanTorch, EqualTorch]:
            pred = cls(signal_x, 1.0)
            pred.set_weights(signal_x)
            assert len(pred.weights) == 0

    def test_set_weights_from_dict_empty(self, signal_x):
        """An empty dict sets no weights."""
        pred = GreaterThanTorch(signal_x, 1.0)
        pred.set_weights_from_dict({})
        assert len(pred.weights) == 0

    def test_get_weight_array_empty(self, signal_x):
        """No weights -> empty tensor."""
        pred = GreaterThanTorch(signal_x, 1.0)
        arr = pred.get_weight_array()
        assert isinstance(arr, torch.Tensor)
        assert arr.numel() == 0


# ---------------------------------------------------------------------------
# AlwaysTorch
# ---------------------------------------------------------------------------


class TestAlwaysTorch:
    """Test AlwaysTorch temporal operator."""

    def test_construction_with_valid_subformula(self, signal_x, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        assert formula.operation is not None
        assert formula.operation_name == "G"

    def test_invalid_subformula_type_raises(self, signal_x):
        """Numpy-based predicate is not accepted."""
        numpy_pred = signal_x >= 2.0  # GreaterThan (numpy), not WSTLFormulaTorch
        with pytest.raises(TypeError):
            AlwaysTorch(numpy_pred, interval=[0, 2])

    def test_set_weights_uniform(self, signal_x, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        # Uniform weights: key is None but tensor is stored
        w = formula.weights[formula.key]
        assert isinstance(w, torch.Tensor)
        assert w.shape[0] == 3  # interval length [0,2] = 3

    def test_set_weights_random(self, signal_x, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x, w_range=[0.5, 2.0], random=True, seed=0)
        w = formula.weights[formula.key]
        assert torch.all(w >= 0.5) and torch.all(w <= 2.0)

    def test_robustness_returns_tensor(self, signal_x, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        rob = formula.robustness(signal_x, t=0)
        assert isinstance(rob, torch.Tensor)

    def test_robustness_shape(self, signal_x, pred_gt):
        """Robustness at one time step has 2D shape (batch, samples)."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        rob = formula.robustness(signal_x, t=0)
        assert rob.ndim == 2

    def test_robustness_values_uniform_weights(self, signal_x, pred_gt):
        """
        pred trace: [1, -1, 2, 0, 3]
        G[0,2] with uniform weights:
          t=0 -> min(1, -1, 2) = -1
          t=2 -> min(2, 0, 3)  =  0
          t=4 -> min(3)        =  3
        """
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        torch.testing.assert_close(
            formula.robustness(signal_x, t=0).flatten()[0], torch.tensor(-1.0)
        )
        torch.testing.assert_close(
            formula.robustness(signal_x, t=2).flatten()[0], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            formula.robustness(signal_x, t=4).flatten()[0], torch.tensor(3.0)
        )

    def test_always_no_interval(self, signal_x, pred_gt):
        """Without interval, should use full signal length [0, inf)."""
        formula = AlwaysTorch(pred_gt)
        formula.set_weights(signal_x)
        rob = formula.robustness(signal_x, t=0)
        assert isinstance(rob, torch.Tensor)

    def test_str_representation(self, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        s = str(formula)
        assert "G" in s
        assert "[0,2]" in s


# ---------------------------------------------------------------------------
# EventuallyTorch
# ---------------------------------------------------------------------------


class TestEventuallyTorch:
    """Test EventuallyTorch temporal operator."""

    def test_construction_with_valid_subformula(self, pred_gt):
        formula = EventuallyTorch(pred_gt, interval=[0, 2])
        assert formula.operation_name == "F"

    def test_invalid_subformula_type_raises(self, signal_x):
        numpy_pred = signal_x <= 2.0  # numpy-based LessThan
        with pytest.raises(TypeError):
            EventuallyTorch(numpy_pred, interval=[0, 2])

    def test_robustness_values_uniform_weights(self, signal_x, pred_gt):
        """
        pred trace: [1, -1, 2, 0, 3]
        F[0,2] with uniform weights:
          t=0 -> max(1, -1, 2) = 2
          t=2 -> max(2, 0, 3)  = 3
          t=4 -> max(3)        = 3
        """
        formula = EventuallyTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        torch.testing.assert_close(
            formula.robustness(signal_x, t=0).flatten()[0], torch.tensor(2.0)
        )
        torch.testing.assert_close(
            formula.robustness(signal_x, t=2).flatten()[0], torch.tensor(3.0)
        )
        torch.testing.assert_close(
            formula.robustness(signal_x, t=4).flatten()[0], torch.tensor(3.0)
        )

    def test_robustness_geq_always(self, signal_x, pred_gt):
        """Eventually robustness >= Always robustness at every time."""
        always = AlwaysTorch(pred_gt, interval=[0, 2])
        eventually = EventuallyTorch(pred_gt, interval=[0, 2])
        always.set_weights(signal_x)
        eventually.set_weights(signal_x)
        for t in range(3):  # only t=0..2 guaranteed non-empty interval
            rob_a = always.robustness(signal_x, t=t)
            rob_e = eventually.robustness(signal_x, t=t)
            assert torch.all(rob_e >= rob_a)

    def test_str_representation(self, pred_gt):
        formula = EventuallyTorch(pred_gt, interval=[0, 3])
        s = str(formula)
        assert "F" in s
        assert "[0,3]" in s


# ---------------------------------------------------------------------------
# AndTorch / OrTorch
# ---------------------------------------------------------------------------


class TestAndTorch:
    """Test AndTorch binary operator."""

    @pytest.fixture
    def pred_x(self, signal_x3):
        return GreaterThanTorch(signal_x3, 0.0)

    @pytest.fixture
    def pred_y(self, signal_y):
        return GreaterThanTorch(signal_y, 0.0)

    def test_construction(self, pred_x, pred_y):
        formula = AndTorch(pred_x, pred_y)
        assert formula.operation_name == "and"
        assert formula.operation is not None

    def test_invalid_sf1_type_raises(self, signal_x3, pred_y):
        numpy_pred = signal_x3 >= 0.0
        with pytest.raises(TypeError):
            AndTorch(numpy_pred, pred_y)

    def test_invalid_sf2_type_raises(self, pred_x, signal_y):
        numpy_pred = signal_y >= 0.0
        with pytest.raises(TypeError):
            AndTorch(pred_x, numpy_pred)

    def test_set_weights(self, signal_x3, signal_y, pred_x, pred_y):
        formula = AndTorch(pred_x, pred_y)
        formula.set_weights((signal_x3, signal_y))
        w = formula.weights[formula.key]
        assert isinstance(w, torch.Tensor)
        assert w.shape[0] == 2  # binary operator weight has 2 entries

    def test_robustness_returns_tensor(self, signal_x3, signal_y, pred_x, pred_y):
        formula = AndTorch(pred_x, pred_y)
        formula.set_weights((signal_x3, signal_y))
        rob = formula.robustness((signal_x3, signal_y), t=0)
        assert isinstance(rob, torch.Tensor)

    def test_robustness_values_uniform_weights(self, signal_x3, signal_y, pred_x, pred_y):
        """
        x=[3,1,4], y=[2,4,0]; pred_x: x>=0 -> [3,1,4]; pred_y: y>=0 -> [2,4,0]
        And(min) at t=0 -> min(3,2)=2, t=1 -> min(1,4)=1, t=2 -> min(4,0)=0
        """
        formula = AndTorch(pred_x, pred_y)
        formula.set_weights((signal_x3, signal_y))
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=0).flatten()[0], torch.tensor(2.0)
        )
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=1).flatten()[0], torch.tensor(1.0)
        )
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=2).flatten()[0], torch.tensor(0.0)
        )

    def test_and_leq_or(self, signal_x3, signal_y, pred_x, pred_y):
        """And(min) robustness <= Or(max) robustness at every time step."""
        and_f = AndTorch(pred_x, pred_y)
        or_f = OrTorch(pred_x, pred_y)
        and_f.set_weights((signal_x3, signal_y))
        or_f.set_weights((signal_x3, signal_y))
        for t in range(3):
            rob_and = and_f.robustness((signal_x3, signal_y), t=t)
            rob_or = or_f.robustness((signal_x3, signal_y), t=t)
            assert torch.all(rob_and <= rob_or)


class TestOrTorch:
    """Test OrTorch binary operator."""

    @pytest.fixture
    def pred_x(self, signal_x3):
        return GreaterThanTorch(signal_x3, 0.0)

    @pytest.fixture
    def pred_y(self, signal_y):
        return GreaterThanTorch(signal_y, 0.0)

    def test_construction(self, pred_x, pred_y):
        formula = OrTorch(pred_x, pred_y)
        assert formula.operation_name == "or"

    def test_robustness_values_uniform_weights(self, signal_x3, signal_y, pred_x, pred_y):
        """
        x=[3,1,4], y=[2,4,0]; pred_x: [3,1,4]; pred_y: [2,4,0]
        Or(max) at t=0 -> max(3,2)=3, t=1 -> max(1,4)=4, t=2 -> max(4,0)=4
        """
        formula = OrTorch(pred_x, pred_y)
        formula.set_weights((signal_x3, signal_y))
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=0).flatten()[0], torch.tensor(3.0)
        )
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=1).flatten()[0], torch.tensor(4.0)
        )
        torch.testing.assert_close(
            formula.robustness((signal_x3, signal_y), t=2).flatten()[0], torch.tensor(4.0)
        )


# ---------------------------------------------------------------------------
# NegationTorch
# ---------------------------------------------------------------------------


class TestNegationTorch:
    """Test NegationTorch operator."""

    def test_construction(self, signal_x, pred_gt):
        formula = NegationTorch(pred_gt)
        assert hasattr(formula, "subformula")

    def test_invalid_subformula_type_raises(self, signal_x):
        numpy_pred = signal_x >= 2.0
        with pytest.raises(TypeError):
            NegationTorch(numpy_pred)

    def test_key_set(self, pred_gt):
        """NegationTorch correctly sets its key after __init__."""
        formula = NegationTorch(pred_gt)
        assert formula.key is not None
        assert "not" in formula.key

    def test_robustness_negates_subformula(self, signal_x, pred_gt):
        """
        pred_gt trace: x>=2.0 -> [1, -1, 2, 0, 3]
        Negation: [-1, 1, -2, 0, -3]
        """
        formula = NegationTorch(pred_gt)
        for t, expected in enumerate([-1.0, 1.0, -2.0, 0.0, -3.0]):
            rob = formula.robustness(signal_x, t=t)
            torch.testing.assert_close(rob.flatten()[0], torch.tensor(expected))

    def test_negation_inverts_sign_for_all_times(self, signal_x, pred_gt):
        """Negation of predicate should always be negative of predicate."""
        neg = NegationTorch(pred_gt)
        for t in range(5):
            rob_pred = pred_gt.robustness(signal_x, t=t)
            rob_neg = neg.robustness(signal_x, t=t)
            torch.testing.assert_close(rob_neg, -rob_pred)

    def test_no_weights_for_predicate_negation(self, signal_x, pred_gt):
        """Negation of a predicate stores no weights."""
        formula = NegationTorch(pred_gt)
        formula.set_weights(signal_x)
        assert len(formula.weights) == 0

    def test_str_representation(self, pred_gt):
        formula = NegationTorch(pred_gt)
        assert "¬" in str(formula)


# ---------------------------------------------------------------------------
# WSTLFormulaTorch: weight dict / array helpers
# ---------------------------------------------------------------------------


class TestWSTLFormulaTorchWeights:
    """Test WSTLFormulaTorch weight management helpers."""

    def test_set_weights_from_dict_numpy(self, signal_x, pred_gt):
        """Numpy arrays are accepted and converted to torch tensors."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)  # initialise weights first
        key = formula.key
        numpy_w = np.ones((3, 1), dtype=np.float32)
        formula.set_weights_from_dict({key: numpy_w})
        assert isinstance(formula.weights[key], torch.Tensor)
        torch.testing.assert_close(formula.weights[key], torch.ones(3, 1))

    def test_set_weights_from_dict_torch(self, signal_x, pred_gt):
        """Torch tensors are accepted directly."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        key = formula.key
        torch_w = torch.full((3, 1), 0.5)
        formula.set_weights_from_dict({key: torch_w})
        torch.testing.assert_close(formula.weights[key], torch_w)

    def test_set_weights_from_dict_invalid_type_raises(self, signal_x, pred_gt):
        """Unsupported weight type raises TypeError."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        key = formula.key
        with pytest.raises(TypeError):
            formula.set_weights_from_dict({key: [1.0, 2.0, 3.0]})

    def test_get_weight_array_no_weights(self, signal_x):
        """Empty weights returns empty tensor."""
        pred = GreaterThanTorch(signal_x, 0.0)
        arr = pred.get_weight_array()
        assert isinstance(arr, torch.Tensor)
        assert arr.numel() == 0

    def test_get_weight_array_with_weights(self, signal_x, pred_gt):
        """Returns concatenated tensor of all weight values."""
        formula = AlwaysTorch(pred_gt, interval=[0, 4])
        formula.set_weights(signal_x)
        arr = formula.get_weight_array()
        assert isinstance(arr, torch.Tensor)
        assert arr.shape[0] == 5  # interval [0,4] has 5 time steps

    def test_set_weights_random_seed_reproducible(self, signal_x, pred_gt):
        """Same seed produces identical weights across two calls."""
        import copy

        f1 = AlwaysTorch(copy.deepcopy(pred_gt), interval=[0, 2])
        f1.set_weights(signal_x, w_range=[0.1, 2.0], random=True, seed=42)
        w1 = f1.weights[f1.key].clone()

        f2 = AlwaysTorch(copy.deepcopy(pred_gt), interval=[0, 2])
        f2.set_weights(signal_x, w_range=[0.1, 2.0], random=True, seed=42)
        w2 = f2.weights[f2.key].clone()

        torch.testing.assert_close(w1, w2)


# ---------------------------------------------------------------------------
# nn.Module compatibility
# ---------------------------------------------------------------------------


class TestWSTLFormulaTorchModule:
    """WSTLFormulaTorch inherits from torch.nn.Module."""

    def test_is_nn_module(self, pred_gt):
        assert isinstance(pred_gt, torch.nn.Module)

    def test_always_is_nn_module(self, pred_gt):
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        assert isinstance(formula, torch.nn.Module)

    def test_forward_method(self, signal_x, pred_gt):
        """forward() delegates to _robustness."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        result = formula.forward(signal_x)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Composite formulas
# ---------------------------------------------------------------------------


class TestCompositeTorch:
    """Test multi-level formula composition."""

    def test_always_of_and(self, signal_x3, signal_y):
        """G[0,1]( (x>=0) And (y>=0) ) should return a valid robustness."""
        pred_x = GreaterThanTorch(signal_x3, 0.0)
        pred_y = GreaterThanTorch(signal_y, 0.0)
        and_f = AndTorch(pred_x, pred_y)
        always_f = AlwaysTorch(and_f, interval=[0, 1])
        always_f.set_weights((signal_x3, signal_y))
        rob = always_f.robustness((signal_x3, signal_y), t=0)
        assert isinstance(rob, torch.Tensor)

    def test_negation_of_always(self, signal_x, pred_gt):
        """¬G[0,2](x>=2.0) should return negative of G robustness."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        neg_formula = NegationTorch(formula)
        neg_formula.set_weights(signal_x)
        rob_g = formula.robustness(signal_x, t=0)
        rob_neg = neg_formula.robustness(signal_x, t=0)
        torch.testing.assert_close(rob_neg, -rob_g)


# ---------------------------------------------------------------------------
# requires_grad
# ---------------------------------------------------------------------------


class TestRequiresGrad:
    """Test WSTLFormulaTorch.requires_grad()."""

    def test_weights_no_grad_by_default(self, signal_x, pred_gt):
        """Weights created by set_weights() do not require grad by default."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        for w in formula.weights.values():
            assert not w.requires_grad

    def test_enable_grad(self, signal_x, pred_gt):
        """requires_grad(True) sets requires_grad on all weights."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        formula.requires_grad(True)
        for w in formula.weights.values():
            assert w.requires_grad

    def test_disable_grad(self, signal_x, pred_gt):
        """requires_grad(False) clears requires_grad on all weights."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        formula.requires_grad(True)
        formula.requires_grad(False)
        for w in formula.weights.values():
            assert not w.requires_grad

    def test_robustness_has_grad_fn_when_enabled(self, signal_x, pred_gt):
        """Output of robustness() has a grad_fn after requires_grad(True)."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        formula.requires_grad(True)
        rob = formula.robustness(signal_x, t=0)
        assert rob.grad_fn is not None

    def test_robustness_no_grad_fn_when_disabled(self, signal_x, pred_gt):
        """Output of robustness() has no grad_fn when grad is off."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        rob = formula.robustness(signal_x, t=0)
        assert rob.grad_fn is None

    def test_nested_formula_all_weights_get_grad(self, signal_x, pred_gt):
        """requires_grad(True) walks the full tree — inner operator is also updated."""
        inner = AlwaysTorch(pred_gt, interval=[0, 1])
        outer = EventuallyTorch(inner, interval=[0, 2])
        outer.set_weights(signal_x)
        outer.requires_grad(True)
        for module in outer.modules():
            for w in module.weights.values():
                assert w.requires_grad

    def test_predicate_requires_grad_is_noop(self, signal_x, pred_gt):
        """Predicates have no weights, so requires_grad() is a silent no-op."""
        pred_gt.requires_grad(True)  # must not raise
        assert len(pred_gt.weights) == 0

    def test_returns_self_for_chaining(self, signal_x, pred_gt):
        """requires_grad() returns self, enabling method chaining."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        result = formula.requires_grad(True)
        assert result is formula

    def test_backward_runs_without_error(self, signal_x, pred_gt):
        """Calling .backward() on the robustness output should not raise."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        formula.requires_grad(True)
        rob = formula.robustness(signal_x, t=0)
        loss = rob.sum()
        loss.backward()  # must not raise
        w = formula.weights[formula.key]
        assert w.grad is not None

    def test_grad_accumulates_after_backward(self, signal_x, pred_gt):
        """Weight gradients are non-None and finite after backward."""
        formula = AlwaysTorch(pred_gt, interval=[0, 2])
        formula.set_weights(signal_x)
        formula.requires_grad(True)
        formula.robustness(signal_x, t=0).sum().backward()
        w = formula.weights[formula.key]
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()


# ---------------------------------------------------------------------------
# CUDA tests — skipped automatically when no GPU is available
# ---------------------------------------------------------------------------

cuda_available = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@cuda_available
class TestCUDADevice:
    """Tests verifying that formulas and weights move correctly to CUDA."""

    @pytest.fixture
    def signal_cuda(self):
        """Signal on CPU (torch backend active); CUDA move is handled by formula.to()."""
        import pywstl

        pywstl.set_backend("torch")
        sig = Signal("x", np.array([[3.0, 1.0, 4.0, 2.0, 5.0]]))
        yield sig
        pywstl.reset_backend()

    def test_device_starts_on_cpu(self, signal_cuda):
        """Default device is cpu before any .to() call."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        assert pred.device == torch.device("cpu")

        formula = AlwaysTorch(pred, interval=[0, 2])
        assert formula.device == torch.device("cpu")

    def test_to_cuda_updates_device(self, signal_cuda):
        """After .to('cuda'), self.device reflects cuda."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        formula = AlwaysTorch(pred, interval=[0, 2])
        formula.to("cuda")
        assert formula.device.type == "cuda"

    def test_weights_move_to_cuda(self, signal_cuda):
        """Weights set before .to('cuda') are moved to GPU."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        formula = AlwaysTorch(pred, interval=[0, 2])
        formula.set_weights(signal_cuda)  # created on CPU
        formula.to("cuda")
        for w in formula.weights.values():
            assert w.device.type == "cuda"

    def test_new_weights_created_on_cuda(self, signal_cuda):
        """Weights set after .to('cuda') are created directly on GPU."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        formula = AlwaysTorch(pred, interval=[0, 2])
        formula.to("cuda")
        formula.set_weights(signal_cuda, random=True, no_samples=5)
        for w in formula.weights.values():
            assert w.device.type == "cuda"

    def test_robustness_on_cuda(self, signal_cuda):
        """Robustness computation runs on GPU and returns a CUDA tensor."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        formula = AlwaysTorch(pred, interval=[0, 2])
        formula.to("cuda")
        formula.set_weights(signal_cuda)
        rob = formula.robustness(signal_cuda, t=0)
        assert isinstance(rob, torch.Tensor)
        assert rob.device.type == "cuda"

    def test_predicate_trace_moves_to_cuda(self, signal_cuda):
        """PredicateTorch._robustness moves input trace to self.device."""
        pred = GreaterThanTorch(signal_cuda, 2.0)
        pred.to("cuda")
        pred.set_weights(signal_cuda)
        rob = pred.robustness(signal_cuda)  # signal is on CPU, pred is on CUDA
        assert rob.device.type == "cuda"

    def test_nested_formula_on_cuda(self, signal_cuda):
        """Nested formula (Always of Eventually) works correctly on CUDA."""
        pred = GreaterThanTorch(signal_cuda, 1.5)
        inner = AlwaysTorch(pred, interval=[0, 1])
        outer = EventuallyTorch(inner, interval=[0, 2])
        outer.to("cuda")
        outer.set_weights(signal_cuda, random=True, no_samples=3)
        rob = outer.robustness(signal_cuda, t=0)
        assert rob.device.type == "cuda"
        assert rob.shape == (1, 3)

    def test_cpu_and_cuda_robustness_match(self, signal_cuda):
        """CPU and CUDA robustness values are numerically identical."""
        pred_cpu = GreaterThanTorch(signal_cuda, 2.0)
        formula_cpu = AlwaysTorch(pred_cpu, interval=[0, 2])
        formula_cpu.set_weights(signal_cuda, seed=42)
        rob_cpu = formula_cpu.robustness(signal_cuda, t=0)

        pred_cuda = GreaterThanTorch(signal_cuda, 2.0)
        formula_cuda = AlwaysTorch(pred_cuda, interval=[0, 2])
        formula_cuda.set_weights(signal_cuda, seed=42)
        formula_cuda.to("cuda")
        rob_cuda = formula_cuda.robustness(signal_cuda, t=0)

        torch.testing.assert_close(rob_cpu, rob_cuda.cpu())
