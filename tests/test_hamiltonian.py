import pennylane as qml
import numpy as np
from app.hamiltonian import build_hamiltonian


def test_hamiltonian_builds():
    params = [0.1, -0.2, 0.3, 0.4]
    H = build_hamiltonian(params)
    assert isinstance(H, qml.Hamiltonian)
    mat = np.array(qml.matrix(H, wire_order=[0, 1]))
    assert mat.shape == (4, 4)
