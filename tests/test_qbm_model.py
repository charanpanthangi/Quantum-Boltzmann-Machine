import numpy as np
from app.qbm_model import compute_density_matrix, model_probabilities


def test_density_matrix_properties():
    params = [0.1, 0.1, 0.1, 0.1]
    rho = compute_density_matrix(params)
    # Hermitian
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)
    # Trace one
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-6)
    # Positive semidefinite (eigenvalues non-negative)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-8)


def test_probabilities_sum_to_one():
    params = [0.0, 0.0, 0.0, 0.0]
    probs = model_probabilities(params)
    np.testing.assert_allclose(probs.sum(), 1.0)
