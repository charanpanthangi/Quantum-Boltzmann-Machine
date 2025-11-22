"""
Plotting utilities for the Quantum Boltzmann Machine demo.
All visuals are saved as SVG files to remain text-based and
version-control friendly.
"""
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.dpi"] = 120


def plot_free_energy_curve(history, output_path: str) -> None:
    """Save an SVG plot showing how the loss changes over epochs."""
    losses = history.get("loss", [])
    epochs = np.arange(len(losses))
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("KL loss (approx free energy)")
    plt.title("QBM training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_sample_distribution(model_probs: np.ndarray, data_probs: np.ndarray, output_path: str) -> None:
    """Compare learned probabilities with the data distribution."""
    labels = ["00", "01", "10", "11"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, data_probs, width, label="Data")
    plt.bar(x + width / 2, model_probs, width, label="Model")
    plt.xticks(x, labels)
    plt.xlabel("Bitstring")
    plt.ylabel("Probability")
    plt.title("Learned vs data distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = ["plot_free_energy_curve", "plot_sample_distribution"]
