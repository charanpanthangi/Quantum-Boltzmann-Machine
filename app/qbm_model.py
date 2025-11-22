"""
Core routines for the Quantum Boltzmann Machine.
This module converts a parameterized Hamiltonian into a Gibbs state,
extracts classical probabilities, and defines a free-energy inspired
loss. The implementation favors clarity over physical completeness so
beginners can follow each step.
"""
from typing import Sequence, Tuple
import numpy as np
from scipy.linalg import expm
import pennylane as qml

from app.hamiltonian import build_hamiltonian


def _hamiltonian_matrix(params: Sequence[float]) -> np.ndarray:
    """Return the full matrix representation of the Hamiltonian.

    The matrix is computed from PennyLane operators and then converted
    to a NumPy array for classical processing.
    """
    hamiltonian = build_hamiltonian(params)
    # qml.matrix builds the dense array in wire order [0, 1].
    return np.array(qml.matrix(hamiltonian, wire_order=[0, 1]), dtype=np.complex128)


def compute_density_matrix(params: Sequence[float], beta: float = 1.0) -> np.ndarray:
    """Compute the Gibbs density matrix ρ = exp(-βH) / Z.

    Parameters
    ----------
    params: Sequence[float]
        Trainable Hamiltonian parameters.
    beta: float
        Inverse temperature β controlling sharpness of the distribution.

    Returns
    -------
    np.ndarray
        Positive semidefinite matrix with unit trace.
    """
    h_mat = _hamiltonian_matrix(params)
    # Matrix exponential gives the unnormalized Gibbs operator.
    gibbs = expm(-beta * h_mat)
    # Partition function is the trace of the exponential.
    Z = np.trace(gibbs)
    density = gibbs / Z
    # Force Hermiticity numerically.
    density = 0.5 * (density + density.conj().T)
    return density


def model_probabilities(params: Sequence[float], beta: float = 1.0) -> np.ndarray:
    """Extract classical probabilities from the diagonal of the density matrix."""
    rho = compute_density_matrix(params, beta=beta)
    diag = np.real(np.diag(rho))
    # Normalize any numerical drift.
    return diag / diag.sum()


def free_energy_loss(params: Sequence[float], data_probs: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute a simple KL divergence between model and data probabilities.

    This captures the spirit of free-energy minimization while avoiding
    heavy quantum sampling. Small epsilons keep the expression stable
    for zero entries.
    """
    model_probs = model_probabilities(params, beta=beta)
    eps = 1e-8
    model_safe = model_probs + eps
    data_safe = data_probs + eps
    return float(np.sum(data_safe * np.log(data_safe / model_safe)))


def summarize_state(params: Sequence[float], beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper returning both density matrix and classical probabilities."""
    rho = compute_density_matrix(params, beta)
    probs = model_probabilities(params, beta)
    return rho, probs


__all__ = [
    "compute_density_matrix",
    "model_probabilities",
    "free_energy_loss",
    "summarize_state",
]
