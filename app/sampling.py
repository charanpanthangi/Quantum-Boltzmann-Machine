"""
Sampling helpers for the Quantum Boltzmann Machine.
These utilities draw classical bitstring samples from the learned
probability distribution so we can visualize what the model generates.
"""
import numpy as np


def sample_from_qbm(model_probs: np.ndarray, n_samples: int = 10) -> np.ndarray:
    """Sample bitstrings according to model probabilities."""
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    choices = np.random.choice(len(states), size=n_samples, p=model_probs)
    return states[choices]


__all__ = ["sample_from_qbm"]
