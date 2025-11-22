"""
Training loop for the Quantum Boltzmann Machine.
This module performs simple gradient descent on the KL divergence
between model and data probabilities. Finite-difference gradients keep
the implementation transparent while remaining fast for two qubits.
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np

from app.qbm_model import free_energy_loss, model_probabilities


def _finite_difference_grad(params: np.ndarray, data_probs: np.ndarray, beta: float, eps: float = 1e-3) -> np.ndarray:
    """Estimate gradients using central finite differences."""
    grads = np.zeros_like(params)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = eps
        loss_plus = free_energy_loss(params + step, data_probs, beta)
        loss_minus = free_energy_loss(params - step, data_probs, beta)
        grads[i] = (loss_plus - loss_minus) / (2 * eps)
    return grads


def train_qbm(
    params: Sequence[float],
    data_probs: np.ndarray,
    n_epochs: int = 50,
    lr: float = 0.1,
    beta: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Optimize Hamiltonian parameters to match the data distribution.

    Parameters
    ----------
    params: Sequence[float]
        Initial parameters for the Hamiltonian.
    data_probs: np.ndarray
        Target empirical distribution from the dataset.
    n_epochs: int
        Number of optimization steps.
    lr: float
        Learning rate for gradient descent.
    beta: float
        Inverse temperature used when building the Gibbs state.

    Returns
    -------
    params: np.ndarray
        Trained parameters.
    history: Dict[str, List[float]]
        Recorded loss values per epoch for plotting.
    """
    params = np.array(params, dtype=float)
    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(n_epochs):
        loss = free_energy_loss(params, data_probs, beta)
        history["loss"].append(loss)

        grads = _finite_difference_grad(params, data_probs, beta)
        params -= lr * grads

    return params, history


def evaluate_model(params: Sequence[float], beta: float = 1.0) -> np.ndarray:
    """Helper to expose model probabilities for downstream tasks."""
    return model_probabilities(params, beta)


__all__ = ["train_qbm", "evaluate_model"]
