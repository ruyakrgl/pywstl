#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test suite for utility functions in the utils module.
"""

import numpy as np
import pytest

from pywstl.signals import Signal
from pywstl.utils import get_signal_shape, get_subformulas


class TestGetSignalShape:
    """Test get_signal_shape utility function."""

    def setup_method(self):
        """Set up test data."""
        # Create signals with known shapes
        self.signal_3x5 = Signal("test1", np.random.rand(3, 5))  # 3 signals, 5 timesteps
        self.signal_2x10 = Signal("test2", np.random.rand(2, 10))  # 2 signals, 10 timesteps
        self.signal_1x20 = Signal("test3", np.random.rand(1, 20))  # 1 signal, 20 timesteps

    def test_single_signal(self):
        """Test get_signal_shape with a single Signal object."""
        signal_no, time_no = get_signal_shape(self.signal_3x5)
        assert signal_no == 3
        assert time_no == 5

        signal_no, time_no = get_signal_shape(self.signal_2x10)
        assert signal_no == 2
        assert time_no == 10

        signal_no, time_no = get_signal_shape(self.signal_1x20)
        assert signal_no == 1
        assert time_no == 20

    def test_tuple_of_signals(self):
        """Test get_signal_shape with a tuple of Signal objects."""
        # Single element tuple
        signal_no, time_no = get_signal_shape((self.signal_3x5,))
        assert signal_no == 3
        assert time_no == 5

        # Multiple element tuple - should use first signal
        signal_no, time_no = get_signal_shape((self.signal_2x10, self.signal_3x5))
        assert signal_no == 2
        assert time_no == 10

    def test_invalid_input_type(self):
        """Test get_signal_shape with invalid input types."""
        with pytest.raises(TypeError, match="Signal must be a tuple or Signal instance"):
            get_signal_shape("not a signal")

        with pytest.raises(TypeError):
            get_signal_shape(123)

        with pytest.raises(TypeError):
            get_signal_shape([self.signal_3x5])  # list instead of tuple

        with pytest.raises(TypeError):
            get_signal_shape(np.array([1, 2, 3]))  # raw array

    def test_invalid_tuple_content(self):
        """Test get_signal_shape with tuple containing invalid objects."""
        with pytest.raises(TypeError):
            get_signal_shape(("not a signal",))

        with pytest.raises(TypeError):
            get_signal_shape((123,))

    def test_multidimensional_signals(self):
        """Test get_signal_shape with various signal dimensions."""
        # 1D time series (1 signal, multiple timesteps)
        signal_1d = Signal("1d", np.random.rand(1, 100))
        signal_no, time_no = get_signal_shape(signal_1d)
        assert signal_no == 1
        assert time_no == 100

        # Large number of signals
        signal_large = Signal("large", np.random.rand(50, 10))
        signal_no, time_no = get_signal_shape(signal_large)
        assert signal_no == 50
        assert time_no == 10


class TestGetSubformulas:
    """Test get_subformulas utility function using actual pywstl formulas."""

    def setup_method(self):
        """Set up test signals and predicates."""
        # Create signals
        self.x = Signal("x", np.array([[1.0, 2.0, 3.0, 4.0]]))
        self.y = Signal("y", np.array([[0.5, 1.5, 2.5, 3.5]]))
        self.z = Signal("z", np.array([[2.0, 3.0, 4.0, 5.0]]))
        self.w = Signal("w", np.array([[1.0, 2.0, 1.0, 2.0]]))

        # Create predicates
        self.phi1 = self.x >= 2.0
        self.phi2 = self.y <= 3.0
        self.phi3 = self.z >= 3.0
        self.phi4 = self.w <= 2.0

    def test_simple_and_formula(self):
        """Test parsing simple 'and' formulas."""
        import pywstl.wstlpy as wstlpy

        phi_and = wstlpy.And(self.phi1, self.phi2)
        formula_str = str(phi_and)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "x>=2.0"
        assert sf2 == "y<=3.0"

    def test_simple_or_formula(self):
        """Test parsing simple 'or' formulas."""
        import pywstl.wstlpy as wstlpy

        phi_or = wstlpy.Or(self.phi1, self.phi2)
        formula_str = str(phi_or)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "x>=2.0"
        assert sf2 == "y<=3.0"

    def test_nested_and_or(self):
        """Test parsing formulas with nested binary operators."""
        import pywstl.wstlpy as wstlpy

        # ((x>=2.0)and(y<=3.0))or((z>=3.0)and(w<=2.0))
        phi_inner1 = wstlpy.And(self.phi1, self.phi2)
        phi_inner2 = wstlpy.And(self.phi3, self.phi4)
        phi_or = wstlpy.Or(phi_inner1, phi_inner2)
        formula_str = str(phi_or)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "(x>=2.0)and(y<=3.0)"
        assert sf2 == "(z>=3.0)and(w<=2.0)"

    def test_temporal_and_predicate(self):
        """Test parsing formulas with temporal operators and predicates."""
        import pywstl.wstlpy as wstlpy

        # (G[0,5](x>=2.0))and(y<=3.0)
        phi_always = wstlpy.Always(self.phi1, interval=[0, 5])
        phi_and = wstlpy.And(phi_always, self.phi2)
        formula_str = str(phi_and)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "G[0,5](x>=2.0)"
        assert sf2 == "y<=3.0"

    def test_temporal_operators_and(self):
        """Test parsing formulas with two temporal operators."""
        import pywstl.wstlpy as wstlpy

        # (G[0,5](x>=2.0))and(F[1,10](y<=3.0))
        phi_always = wstlpy.Always(self.phi1, interval=[0, 5])
        phi_eventually = wstlpy.Eventually(self.phi2, interval=[1, 10])
        phi_and = wstlpy.And(phi_always, phi_eventually)
        formula_str = str(phi_and)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "G[0,5](x>=2.0)"
        assert sf2 == "F[1,10](y<=3.0)"

    def test_temporal_operators_or(self):
        """Test parsing formulas with temporal operators and 'or'."""
        import pywstl.wstlpy as wstlpy

        # (F[0,10](x>=2.0))or(G[5,15](y<=3.0))
        phi_eventually = wstlpy.Eventually(self.phi1, interval=[0, 10])
        phi_always = wstlpy.Always(self.phi2, interval=[5, 15])
        phi_or = wstlpy.Or(phi_eventually, phi_always)
        formula_str = str(phi_or)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "F[0,10](x>=2.0)"
        assert sf2 == "G[5,15](y<=3.0)"

    def test_complex_nested_temporal(self):
        """Test parsing complex nested formulas with temporal operators."""
        import pywstl.wstlpy as wstlpy

        # (F[0,10]((x>=2.0)and(y<=3.0)))or(G[5,15]((z>=3.0)or(w<=2.0)))
        inner_and = wstlpy.And(self.phi1, self.phi2)
        inner_or = wstlpy.Or(self.phi3, self.phi4)
        phi_eventually = wstlpy.Eventually(inner_and, interval=[0, 10])
        phi_always = wstlpy.Always(inner_or, interval=[5, 15])
        phi_complex = wstlpy.Or(phi_eventually, phi_always)
        formula_str = str(phi_complex)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "F[0,10]((x>=2.0)and(y<=3.0))"
        assert sf2 == "G[5,15]((z>=3.0)or(w<=2.0))"

    def test_whitespace_handling(self):
        """Test that whitespace in formulas is handled correctly."""
        import pywstl.wstlpy as wstlpy

        phi_and = wstlpy.And(self.phi1, self.phi2)
        formula_str = "  " + str(phi_and) + "  "  # Add whitespace

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "x>=2.0"
        assert sf2 == "y<=3.0"

    def test_invalid_formula_no_operator(self):
        """Test that formulas without operators raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse subformulas"):
            get_subformulas("(F[0,5](x>=2))")

        with pytest.raises(ValueError):
            get_subformulas("just_some_text")

        with pytest.raises(ValueError):
            get_subformulas("x>=2.0")  # Single predicate, no operator

    def test_invalid_formula_structure(self):
        """Test that malformed formulas raise ValueError."""
        with pytest.raises(ValueError):
            get_subformulas("")

        with pytest.raises(ValueError):
            get_subformulas("()")

        with pytest.raises(ValueError):
            get_subformulas("(x)")

    def test_unbalanced_parentheses(self):
        """Test formulas with unbalanced parentheses."""
        # These should raise ValueError because operator won't be found at paren_count==0
        with pytest.raises(ValueError):
            get_subformulas("((x>=1)and(y<=2)")  # Missing closing paren

        with pytest.raises(ValueError):
            get_subformulas("(x>=1))and((y<=2)")  # Extra closing paren at start

    def test_multiple_operators_chained(self):
        """Test formulas with multiple operators chained together."""
        import pywstl.wstlpy as wstlpy

        # Create three-way AND: (x>=2.0)and((y<=3.0)and(z>=3.0))
        inner_and = wstlpy.And(self.phi2, self.phi3)
        outer_and = wstlpy.And(self.phi1, inner_and)
        formula_str = str(outer_and)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "x>=2.0"
        assert sf2 == "(y<=3.0)and(z>=3.0)"

    def test_and_before_or_priority(self):
        """Test that 'and' is found before 'or' when parsing."""
        import pywstl.wstlpy as wstlpy

        # Test 'and' operator parsing
        phi_and = wstlpy.And(self.phi1, self.phi2)
        sf1, sf2 = get_subformulas(str(phi_and))
        assert sf1 == "x>=2.0"
        assert sf2 == "y<=3.0"

        # Test 'or' operator parsing
        phi_or = wstlpy.Or(self.phi3, self.phi4)
        sf1, sf2 = get_subformulas(str(phi_or))
        assert sf1 == "z>=3.0"
        assert sf2 == "w<=2.0"

    def test_deeply_nested_formulas(self):
        """Test parsing deeply nested formulas."""
        import pywstl.wstlpy as wstlpy

        # Create G[0,5](F[1,3]((x>=2.0)and(y<=3.0)))
        inner_and = wstlpy.And(self.phi1, self.phi2)
        eventually = wstlpy.Eventually(inner_and, interval=[1, 3])
        always = wstlpy.Always(eventually, interval=[0, 5])

        # Now combine with another formula
        phi_final = wstlpy.Or(always, self.phi3)
        formula_str = str(phi_final)

        sf1, sf2 = get_subformulas(formula_str)
        assert sf1 == "G[0,5](F[1,3]((x>=2.0)and(y<=3.0)))"
        assert sf2 == "z>=3.0"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
