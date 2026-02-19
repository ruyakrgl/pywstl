#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for the backend selection module and factory classes.
"""

import numpy as np
import pytest

import pywstl
import pywstl.backend as backend_module
import pywstl.wstlpy as wstlpy
from pywstl.signals import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_backend_after_test():
    """Always reset the backend to numpy after each test."""
    yield
    pywstl.reset_backend()


@pytest.fixture
def simple_signal():
    return Signal("x", np.linspace(0, 1, 10).reshape(1, 10, 1, 1))


# ---------------------------------------------------------------------------
# Backend state management
# ---------------------------------------------------------------------------


class TestGetBackend:
    """Test get_backend function."""

    def test_default_backend_is_numpy(self):
        assert pywstl.get_backend() == "numpy"

    def test_get_backend_after_set(self):
        pywstl.set_backend("numpy")
        assert pywstl.get_backend() == "numpy"


class TestSetBackend:
    """Test set_backend function."""

    def test_set_numpy_backend(self):
        pywstl.set_backend("numpy")
        assert pywstl.get_backend() == "numpy"

    def test_set_numpy_backend_uppercase(self):
        """Backend selection is case-insensitive."""
        pywstl.set_backend("NUMPY")
        assert pywstl.get_backend() == "numpy"

    def test_set_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Backend must be 'numpy' or 'torch'"):
            pywstl.set_backend("jax")

    def test_set_empty_string_raises(self):
        with pytest.raises(ValueError):
            pywstl.set_backend("")

    def test_set_backend_torch_without_torch_raises(self):
        """If torch is not installed, setting torch backend raises RuntimeError."""
        if pywstl.is_torch_available():
            pytest.skip("pytorch is installed; skipping unavailability test")
        with pytest.raises(RuntimeError, match="pytorch backend not available"):
            pywstl.set_backend("torch")


class TestResetBackend:
    """Test reset_backend function."""

    def test_reset_restores_numpy(self):
        pywstl.set_backend("numpy")
        pywstl.reset_backend()
        assert pywstl.get_backend() == "numpy"

    def test_reset_is_idempotent(self):
        pywstl.reset_backend()
        pywstl.reset_backend()
        assert pywstl.get_backend() == "numpy"


class TestIsTorchAvailable:
    """Test is_torch_available function."""

    def test_returns_bool(self):
        result = pywstl.is_torch_available()
        assert isinstance(result, bool)

    def test_consistent_with_import(self):
        """Result should be consistent with whether torch can be imported."""
        try:
            import torch  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert pywstl.is_torch_available() == expected


class TestIsCudaAvailable:
    """Test is_cuda_available function."""

    def test_returns_bool(self):
        result = pywstl.is_cuda_available()
        assert isinstance(result, bool)

    def test_consistent_with_torch_cuda(self):
        """Result should be consistent with torch.cuda.is_available() if torch is available."""
        if pywstl.is_torch_available():
            import torch

            expected = torch.cuda.is_available()
            assert pywstl.is_cuda_available() == expected
        else:
            assert pywstl.is_cuda_available() is False


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


class TestGetImplementationModule:
    """Test _get_implementation_module internal helper."""

    def test_numpy_backend_returns_wstlpy(self):
        pywstl.set_backend("numpy")
        module = backend_module._get_implementation_module()
        import pywstl.wstlpy as wstlpy_mod

        assert module is wstlpy_mod

    def test_invalid_backend_raises(self):
        """Force an invalid backend state and check RuntimeError."""
        original = backend_module._BACKEND
        backend_module._BACKEND = "invalid"
        try:
            with pytest.raises(RuntimeError, match="Invalid backend"):
                backend_module._get_implementation_module()
        finally:
            backend_module._BACKEND = original


# ---------------------------------------------------------------------------
# Factory class: Always
# ---------------------------------------------------------------------------


class TestAlwaysFactory:
    """Test the Always factory class dispatches to numpy backend."""

    def test_always_returns_wstlpy_instance(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Always(pred, interval=[0, 5])
        assert isinstance(formula, wstlpy.Always)

    def test_always_no_interval(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Always(pred)
        assert isinstance(formula, wstlpy.Always)

    def test_always_computes_robustness(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Always(pred, interval=[0, 5])
        formula.set_weights(simple_signal)
        rob = formula.robustness(simple_signal, t=0)
        assert rob is not None


# ---------------------------------------------------------------------------
# Factory class: Eventually
# ---------------------------------------------------------------------------


class TestEventuallyFactory:
    """Test the Eventually factory class dispatches to numpy backend."""

    def test_eventually_returns_wstlpy_instance(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Eventually(pred, interval=[0, 5])
        assert isinstance(formula, wstlpy.Eventually)

    def test_eventually_no_interval(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Eventually(pred)
        assert isinstance(formula, wstlpy.Eventually)

    def test_eventually_computes_robustness(self, simple_signal):
        pred = simple_signal >= 0.0
        formula = pywstl.Eventually(pred, interval=[0, 5])
        formula.set_weights(simple_signal)
        rob = formula.robustness(simple_signal, t=0)
        assert rob is not None


# ---------------------------------------------------------------------------
# Factory class: And
# ---------------------------------------------------------------------------


class TestAndFactory:
    """Test the And factory class dispatches to numpy backend."""

    def test_and_returns_wstlpy_instance(self, simple_signal):
        pred1 = simple_signal >= 0.0
        pred2 = simple_signal <= 1.0
        formula = pywstl.And(pred1, pred2)
        assert isinstance(formula, wstlpy.And)

    def test_and_computes_robustness(self, simple_signal):
        pred1 = simple_signal >= 0.0
        pred2 = simple_signal <= 1.0
        formula = pywstl.And(pred1, pred2)
        formula.set_weights((simple_signal, simple_signal))
        rob = formula.robustness((simple_signal, simple_signal), t=0)
        assert rob is not None


# ---------------------------------------------------------------------------
# Factory class: Or
# ---------------------------------------------------------------------------


class TestOrFactory:
    """Test the Or factory class dispatches to numpy backend."""

    def test_or_returns_wstlpy_instance(self, simple_signal):
        pred1 = simple_signal >= 0.0
        pred2 = simple_signal <= 1.0
        formula = pywstl.Or(pred1, pred2)
        assert isinstance(formula, wstlpy.Or)

    def test_or_computes_robustness(self, simple_signal):
        pred1 = simple_signal >= 0.0
        pred2 = simple_signal <= 1.0
        formula = pywstl.Or(pred1, pred2)
        formula.set_weights((simple_signal, simple_signal))
        rob = formula.robustness((simple_signal, simple_signal), t=0)
        assert rob is not None


# ---------------------------------------------------------------------------
# Factory class: Negation
# ---------------------------------------------------------------------------


class TestNegationFactory:
    """Test the Negation factory class raises NotImplementedError (PNF requirement)."""

    def test_negation_raises_at_construction(self, simple_signal):
        """Negation raises NotImplementedError immediately (formulas must be in PNF)."""
        pred = simple_signal >= 0.0
        with pytest.raises(NotImplementedError):
            pywstl.Negation(pred)


# ---------------------------------------------------------------------------
# Predicate factory classes
# ---------------------------------------------------------------------------


class TestLessThanFactory:
    """Test the LessThan factory class."""

    def test_returns_signals_lessthan(self, simple_signal):
        from pywstl.signals import LessThan as LTSignal

        pred = pywstl.LessThan(simple_signal, 0.5)
        assert isinstance(pred, LTSignal)

    def test_robustness_value(self, simple_signal):
        pred = pywstl.LessThan(simple_signal, 0.5)
        pred.set_weights(simple_signal)
        rob = pred.robustness(simple_signal, t=0)
        # robustness of x <= 0.5 at t=0 should be 0.5 - x[t=0]
        x_val = simple_signal.value[0, 0, 0, 0]
        expected = 0.5 - x_val
        np.testing.assert_allclose(rob.flatten()[0], expected)


class TestGreaterThanFactory:
    """Test the GreaterThan factory class."""

    def test_returns_signals_greaterthan(self, simple_signal):
        from pywstl.signals import GreaterThan as GTSignal

        pred = pywstl.GreaterThan(simple_signal, 0.5)
        assert isinstance(pred, GTSignal)

    def test_robustness_value(self, simple_signal):
        pred = pywstl.GreaterThan(simple_signal, 0.5)
        pred.set_weights(simple_signal)
        rob = pred.robustness(simple_signal, t=0)
        # robustness of x >= 0.5 at t=0 should be x[t=0] - 0.5
        x_val = simple_signal.value[0, 0, 0, 0]
        expected = x_val - 0.5
        np.testing.assert_allclose(rob.flatten()[0], expected)


class TestEqualFactory:
    """Test the Equal factory class."""

    def test_returns_signals_equal(self, simple_signal):
        from pywstl.signals import Equal as EqSignal

        pred = pywstl.Equal(simple_signal, 0.5)
        assert isinstance(pred, EqSignal)

    def test_robustness_value(self, simple_signal):
        pred = pywstl.Equal(simple_signal, 0.5)
        pred.set_weights(simple_signal)
        rob = pred.robustness(simple_signal, t=0)
        # robustness of x == 0.5 at t=0 should be -|x[t=0] - 0.5|
        x_val = simple_signal.value[0, 0, 0, 0]
        expected = -np.abs(x_val - 0.5)
        np.testing.assert_allclose(rob.flatten()[0], expected)


# ---------------------------------------------------------------------------
# Integration: backend context switch
# ---------------------------------------------------------------------------


class TestBackendContextIntegration:
    """Integration tests verifying factory behaviour matches get_backend()."""

    def test_numpy_factory_gives_numpy_types(self, simple_signal):
        pywstl.set_backend("numpy")
        pred = simple_signal >= 0.0
        formula = pywstl.Always(pred, interval=[0, 5])
        assert isinstance(formula, wstlpy.WSTLFormula)

    def test_torch_backend_unavailable_raises_on_factory(self, simple_signal):
        """When torch is not installed, using torch factory raises RuntimeError."""
        if pywstl.is_torch_available():
            pytest.skip("pytorch is installed; skipping unavailability test")
        # Manually force torch backend to test factory guard
        backend_module._BACKEND = "torch"
        try:
            pred = simple_signal >= 0.0
            with pytest.raises(RuntimeError):
                pywstl.Always(pred, interval=[0, 5])
        finally:
            backend_module._BACKEND = "numpy"
