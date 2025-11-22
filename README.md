# Quantum Boltzmann Machine (QBM)

## What This Project Does
- Implements a tiny Quantum Boltzmann Machine with two qubits.
- A Hamiltonian assigns energies to bitstrings.
- The Gibbs distribution turns energies into probabilities.
- Training nudges the Hamiltonian parameters so the model matches the data distribution.

## Why Quantum EBMs Are Interesting
- Quantum Hamiltonians can express richer energy landscapes.
- Gibbs states may capture correlations classical EBMs struggle with.
- QBMs show how quantum systems can model probability distributions.

## Why SVG Instead of PNG
> GitHub’s CODEX interface cannot preview binary image files like PNG/JPG
> and shows “Binary files are not supported.”
> All images in this repository are lightweight SVG files to ensure clean
> rendering and diff-friendly behavior.

## How It Works (Plain English)
- Create a tiny binary dataset.
- Build a trainable Hamiltonian.
- Compute the Gibbs distribution from the Hamiltonian.
- Adjust parameters with gradient descent.
- Compare model and data distributions.

## How to Run
```
pip install -r requirements.txt
python app/main.py
```

## What You Should See
- Training loss (approx free energy) decreasing.
- Learned distribution approaching the data distribution.
- SVG plots saved to `examples/`.

## Future Extensions
- More qubits.
- Transverse-field QBMs.
- Quantum-assisted contrastive divergence.
