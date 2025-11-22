"""
Hamiltonian construction for the Quantum Boltzmann Machine example.
The Hamiltonian encodes the energy landscape that the model will
learn to match to the training data distribution. Lower energy states
are assigned higher probability in the Gibbs distribution.
"""
from typing import Sequence
import pennylane as qml


def build_hamiltonian(params: Sequence[float]) -> qml.Hamiltonian:
    """
    Create a simple two-qubit Hamiltonian with tunable coefficients.

    Parameters
    ----------
    params: Sequence[float]
        Iterable with four values controlling each term.

    Returns
    -------
    qml.Hamiltonian
        Hamiltonian H = θ1 Z0 + θ2 Z1 + θ3 Z0 Z1 + θ4 X0
    """
    if len(params) != 4:
        raise ValueError("Expected four parameters for the Hamiltonian.")

    # Pauli operators acting on specific wires.
    z0 = qml.PauliZ(0)
    z1 = qml.PauliZ(1)
    x0 = qml.PauliX(0)

    # Coefficients map directly to the provided parameters.
    coeffs = [params[0], params[1], params[2], params[3]]
    ops = [z0, z1, z0 @ z1, x0]

    return qml.Hamiltonian(coeffs, ops)


__all__ = ["build_hamiltonian"]
