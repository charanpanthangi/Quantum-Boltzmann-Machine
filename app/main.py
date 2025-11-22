"""
Command-line entry point demonstrating QBM training.
The pipeline loads a tiny dataset, initializes parameters, trains the
model, samples from it, and saves SVG plots to the examples folder.
"""
import argparse
import numpy as np

from app.dataset import load_binary_dataset
from app.trainer import train_qbm, evaluate_model
from app.sampling import sample_from_qbm
from app.plots import plot_free_energy_curve, plot_sample_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Quantum Boltzmann Machine")
    parser.add_argument("--epochs", type=int, default=60, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="inverse temperature")
    args = parser.parse_args()

    # Load binary patterns and their probabilities.
    _, data_probs = load_binary_dataset()

    # Small random initialization keeps energies near zero initially.
    np.random.seed(0)
    init_params = np.random.uniform(low=-0.5, high=0.5, size=4)

    trained_params, history = train_qbm(init_params, data_probs, n_epochs=args.epochs, lr=args.lr, beta=args.beta)
    final_loss = history["loss"][-1]
    print(f"Final loss: {final_loss:.4f}")

    # Evaluate the learned distribution and draw a few samples.
    model_probs = evaluate_model(trained_params, beta=args.beta)
    samples = sample_from_qbm(model_probs, n_samples=8)
    print("Generated samples (rows = bitstrings):")
    print(samples)

    # Save SVG plots.
    plot_free_energy_curve(history, "examples/qbm_free_energy_curve.svg")
    plot_sample_distribution(model_probs, data_probs, "examples/qbm_sample_distribution.svg")
    print("SVG visuals saved to examples/")


if __name__ == "__main__":
    main()
