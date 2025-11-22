import numpy as np
from app.dataset import load_binary_dataset


def test_dataset_shapes():
    samples, probs = load_binary_dataset()
    assert samples.shape[1] == 2
    assert probs.shape[0] == samples.shape[0]
    np.testing.assert_allclose(probs.sum(), 1.0)


def test_dataset_values():
    _, probs = load_binary_dataset()
    assert np.all(probs >= 0)
