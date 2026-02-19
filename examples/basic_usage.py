#!/usr/bin/env python3
# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic Usage Example for PyWSTL

This example demonstrates:
1. Creating signals,
2. Building WSTL formulas with predicates and temporal operators,
3. Computing weighted robustness for STL and WSTL cases, and
4. Setting custom weights,
for different backends.
"""

import pywstl
from pywstl import Signal, Always, Eventually, And, Or
import numpy as np
import torch


def example_setup():
    # Step 1: Create signals
    print("\n1. Creating signals...")

    # backends automatically convert signal values to the appropriate
    # type (numpy array or torch tensor)
    x = Signal("x", np.array([[1.0, 2.0, 3.0, 4.0, 3.0, 2.0]]))
    y = Signal("y", np.array([[0.5, 1.5, 2.5, 3.5, 2.5, 1.5]]))
    print(f"   Signal x: {x.value}")
    print(f"   Signal y: {y.value}")

    # Step 2: Create predicates
    print("\n2. Creating predicates...")
    phi1 = x >= 2.0  # x should be greater than or equal to 2
    phi2 = y <= 3.0  # y should be less than or equal to 3
    print(f"   phi1: {phi1}")
    print(f"   phi2: {phi2}")

    # Step 3: Build STL formulas
    print("\n3. Building STL formulas...")

    # Always: G[0,2] (x >= 2)
    # x is always >= 2 for the next [0,2] time steps
    phi_always = Always(subformula=phi1, interval=[0, 2])
    print(f"   Always: {phi_always}")

    # Eventually: F[1,3] (y <= 3)
    # y is eventually <= 3 in the next [1,3] time steps
    phi_eventually = Eventually(subformula=phi2, interval=[1, 3])
    print(f"   Eventually: {phi_eventually}")

    # Conjunction: (x >= 2) AND (y <= 3)
    phi_and = And(phi1, phi2)
    print(f"   And: {phi_and}")

    # Disjunction: (x >= 2) OR (y <= 3)
    phi_or = Or(phi1, phi2)
    print(f"   Or: {phi_or}")

    # Step 4: Set weights and compute robustness
    print(f"\n4. Computing robustness for {phi_and}...")

    # Set uniform weights (all weights = 1.0)
    phi_and.set_weights(signals=(x, y), w_range=[1.0, 1.0])

    # Compute robustness at t=0
    rho_0 = phi_and.robustness((x, y), t=0)
    print(f"   Robustness of {phi_and} at t=0 when weights are 1: {rho_0}")

    # Compute robustness at t=2
    rho_2 = phi_and.robustness((x, y), t=2)
    print(f"   Robustness of {phi_and} at t=2 when weights are 1: {rho_2}")

    # Step 5: Custom weights
    print("\n5. Using custom weights...")

    # Get actual keys from the formula
    print(f"   Available weight keys: {list(phi_and.weights.keys())}")

    and_key = list(phi_and.weights.keys())[0]  # Get the first key

    if pywstl.get_backend() == "torch":
        custom_weights = {and_key: torch.tensor([1.5, 0.5])}
    else:
        custom_weights = {and_key: np.array([1.5, 0.5])}

    phi_and.set_weights_from_dict(custom_weights)
    rho_custom = phi_and.robustness((x, y))
    print(f"   Applied custom weights: {custom_weights}")
    print(f"   Robustness with custom weights at t=0: {rho_custom}")

    phi_always.set_weights(x, w_range=[1.0, 1.0])
    rho_always = phi_always.robustness(x, t=0)
    print(f"   G[0,2](x >= 2) at t=0 when weights are 1: {rho_always}")

    phi_eventually.set_weights(y, w_range=[1.0, 1.0])
    rho_eventually = phi_eventually.robustness(y, t=0)
    print(f"   F[1,3](y <= 3) at t=0 when weights are 1: {rho_eventually}")


def main():
    print("=" * 60)
    print("Py(W)STL Basic Usage Example")
    print("=" * 60)

    print("\nA. Using numpy backend (default)...")
    pywstl.set_backend("numpy")
    example_setup()
    print("\n" + "=" * 60)

    print("\nB. Examples to PyTorch backend...")
    pywstl.set_backend("torch")
    example_setup()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
