#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging

import numpy as np
import pytest

import pywstl.wstlpy as wp  # NOQA
from pywstl.signals import Signal  # NOQA


class TestSetWeights:
    def setup_method(self):
        self.x = Signal("x", np.linspace(0, 10, 100).reshape(1, 100, 1, 1))

        self.logic_input = (self.x, self.x)
        self.logicformula = wp.And(self.x >= 0, self.x <= 100)

        self.temporalformula = wp.Always(self.x >= 0, [0, 5])
        self.temporalformula_inf = wp.Always(self.x >= 0)

        self.complexformula = wp.Always(self.logicformula, [0, 50])

        self.mostcomplex = wp.Always(
            wp.Or(wp.Eventually(wp.Always(wp.And(self.x >= 0, self.x <= 100))), self.x <= 150)
        )

        self.Nsamples = 10

    def test_set_weights_logical(self):
        formula = self.logicformula
        formula.set_weights(self.logic_input)

        assert "(x>=0)and(x<=100)" in formula.weights.keys()
        assert formula.weights["(x>=0)and(x<=100)"].shape == (2, 1)

    def test_set_weights_temporal(self):
        formula = self.temporalformula
        formula.set_weights(self.x)

        assert "G[0,5](x>=0)" in formula.weights.keys()
        assert formula.weights["G[0,5](x>=0)"].shape == (6, 1)

    def test_set_weights_temporal_inf(self):
        formula = self.temporalformula_inf
        formula.set_weights(self.x)

        assert "G(x>=0)" in formula.weights.keys()
        assert formula.weights["G(x>=0)"].shape == (100, 1)

    def test_set_weights_complex(self):
        formula = self.complexformula
        formula.set_weights(self.logic_input)

        assert "G[0,50]((x>=0)and(x<=100))" in formula.weights.keys()
        assert formula.weights["G[0,50]((x>=0)and(x<=100))"].shape == (51, 1)
        assert formula.weights["(x>=0)and(x<=100)"].shape == (2, 1)

    def test_set_seed(self):
        formula1 = copy.deepcopy(self.temporalformula)
        formula1.set_weights(self.x, w_range=[0.1, 1], random=True, seed=42)
        weights1 = formula1.weights["G[0,5](x>=0)"].copy()

        formula2 = copy.deepcopy(self.temporalformula)
        formula2.set_weights(self.x, w_range=[0.1, 1], random=True, seed=42)
        weights2 = formula2.weights["G[0,5](x>=0)"].copy()

        np.testing.assert_array_equal(weights1, weights2)

        formula3 = copy.deepcopy(self.temporalformula)
        formula3.set_weights(self.x, w_range=[0.1, 1], random=True, seed=0)
        weights3 = formula3.weights["G[0,5](x>=0)"].copy()
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(weights1, weights3)

    def test_set_weights_no_random_with_seed(self):
        formula = copy.deepcopy(self.temporalformula)
        formula.set_weights(self.x, random=False, seed=42)
        weights = formula.weights["G[0,5](x>=0)"].copy()
        expected_weights = np.full((6, 1), 1)
        np.testing.assert_array_equal(weights, expected_weights)

    def test_set_weight_random_wo_range(self, caplog):
        formula = copy.deepcopy(self.temporalformula)
        with caplog.at_level(logging.WARNING):
            formula.set_weights(self.x, random=True)

    def test_set_weights_from_dict(self):
        self.complexformula.set_weights(self.logic_input)
        key1 = "G[0,50]((x>=0)and(x<=100))"
        w1 = np.linspace(0.1, 1, 51).reshape(51, 1)

        key2 = "(x>=0)and(x<=100)"
        w2 = np.linspace(0.1, 2, 2).reshape(2, 1)
        weights = {key1: w1, key2: w2}
        self.complexformula.set_weights_from_dict(weights)
        np.testing.assert_array_equal(weights[key1], self.complexformula.weights[key1])
        np.testing.assert_array_equal(weights[key2], self.complexformula.weights[key2])

    def test_set_weights_from_dict_most_complex(self):
        f1 = copy.deepcopy(self.mostcomplex)
        f1.set_weights(((self.x, self.x), self.x), w_range=[0.1, 1], random=True, seed=42)
        weights1 = f1.weights.copy()

        f2 = copy.deepcopy(self.mostcomplex)
        f2.set_weights_from_dict(weights1)

        for key in weights1.keys():
            np.testing.assert_array_equal(weights1[key], f2.weights[key])

    def test_get_weight_array(self):
        w1 = np.linspace(0.1, 1, 51).reshape(51, 1)
        w2 = np.linspace(0.1, 2, 2).reshape(2, 1)

        w = np.vstack((w2, w1))

        self.complexformula.set_weights_from_dict(
            {"G[0,50]((x>=0)and(x<=100))": w1, "(x>=0)and(x<=100)": w2}
        )

        weight_array = self.complexformula.get_weight_array()
        np.testing.assert_array_equal(w, weight_array)

        formula1 = copy.deepcopy(self.temporalformula)
        with pytest.raises(ValueError):
            formula1.get_weight_array()


class TestSetDictNames:
    def setup_method(self):
        self.x = Signal("x", np.linspace(0, 10, 100).reshape(1, 100, 1, 1))

        self.subf = wp.Always(self.x >= 0)
        self.same_name = wp.And(self.subf, copy.deepcopy(self.subf))
        self.same_name_or = wp.Or(self.same_name, copy.deepcopy(self.same_name))

    def test_set_keys(self):
        formula = self.same_name_or
        formula.set_keys()

        dict_names = [
            "x>=0",
            "x>=0_",
            "x>=0__",
            "x>=0___",
            "G(x>=0)",
            "G(x>=0_)",
            "G(x>=0__)",
            "G(x>=0___)",
            "(G(x>=0))and(G(x>=0_))",
            "(G(x>=0__))and(G(x>=0___))",
            "((G(x>=0))and(G(x>=0_)))or((G(x>=0__))and(G(x>=0___)))",
        ]

        for name in dict_names:
            assert name in formula.keys


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
