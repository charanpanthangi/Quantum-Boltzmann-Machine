"""
Dataset utilities for simple binary patterns.
This module provides a toy dataset of small bitstrings along with
an empirical distribution. These tiny examples are enough to
illustrate Quantum Boltzmann Machine training without heavy
computational cost.
"""
from typing import Tuple
import numpy as np


def load_binary_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a tiny binary dataset and its empirical distribution.

    Returns
    -------
    X_data: np.ndarray
        Array of binary samples shaped (n_samples, n_bits).
    p_data: np.ndarray
        Probability mass function over the observed bitstrings.
    """
    # Two-bit patterns with simple clustering structure.
    samples = np.array([
        [0, 0],  # likely state
        [1, 1],  # likely state
        [0, 1],  # less likely
        [1, 0],  # less likely
    ], dtype=int)

    # Assign probabilities so that [0,0] and [1,1] are more common.
    probabilities = np.array([0.35, 0.35, 0.15, 0.15], dtype=float)

    # Normalize for safety in case values change later.
    probabilities = probabilities / probabilities.sum()

    return samples, probabilities


__all__ = ["load_binary_dataset"]
