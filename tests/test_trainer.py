import numpy as np
from app.dataset import load_binary_dataset
from app.trainer import train_qbm


def test_loss_decreases():
    _, data_probs = load_binary_dataset()
    init_params = np.array([0.3, -0.1, 0.05, 0.2])
    trained_params, history = train_qbm(init_params, data_probs, n_epochs=8, lr=0.3)
    assert history["loss"][0] > history["loss"][-1]
    assert trained_params.shape[0] == init_params.shape[0]
