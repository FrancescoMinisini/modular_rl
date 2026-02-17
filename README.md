# Quantum Control with TRPO

This repository contains an implementation of **Trust Region Policy Optimization (TRPO)** for Quantum Control, specifically targeting gate optimization in a superconducting qubit system. The project aims to find optimal control pulses to implement high-fidelity quantum gates (e.g., CZ gate) while minimizing leakage and satisfying boundary conditions.

## Project Structure

The codebase is streamlined to focus on the core TRPO implementation for quantum control. The key files are:

- **`run_trpo_quantum.py`**: The main entry point for training. It sets up the environment, initializes the agent, and runs the training loop with a curriculum learning approach (varying the target gate parameter $\alpha$).
- **`quantum_env.py`**: Defines the `QuantumEnv`, a Gym-like environment that simulates the quantum dynamics. It includes:
    - `ExactExponentialFilter`: Simulates the effect of control electronics bandwidth.
    - **Reward/Cost Function**: Implements the "UFO" cost function (Unidelity, Fidelity, Leakage, Boundary conditions).
    - **Dynamics**: Solves the Schrödinger equation to evolve the system state.
- **`quantum_system.py`**: Handles the low-level quantum system Hamiltonian, evolution, and fidelity calculations.
- **`trpo_agent.py`**: Identifying the PyTorch implementation of the TRPO algorithm, including the Policy Network (Gaussian policy) and Value Network.
- **`plot_results.py`**: Scripts for visualizing training logs (fidelity, cost, etc.).

## Installation

To run this project, you need Python and the following dependencies:

```bash
pip install numpy torch gym scipy matplotlib
```

## Usage

To start training the quantum control agent:

```bash
python run_trpo_quantum.py
```

### Key Arguments

You can customize the training via command-line arguments:

- `--target`: Target gate (default: "CZ").
- `--n_iter`: Number of TRPO iterations per curriculum step (default: 100).
- `--timesteps_per_batch`: Number of episodes per batch (default: 100).
- `--max_steps`: Maximum steps per episode (default: 500).
- `--dt`: Time step duration in ns (default: 1.0).
- `--seed`: Random seed (default: 1).
- `--noise_optimized`: Enable optimized noise model (default: False).

**Example:**

```bash
python run_trpo_quantum.py --target CZ --n_iter 50 --timesteps_per_batch 50
```

## Methodology

1.  **Environment**: The agent interacts with a simulated 3-level quantum system (qutrit) to avoid leakage errors.
2.  **Control**: The agent outputs control signals (amplitudes and phases) which are filtered and applied to the Hamiltonian.
3.  **Objective**: Minimize a cost function composed of:
    - **In-fidelity**: $1 - F$
    - **Leakage**: Population leaving the computational subspace.
    - **Boundary Constraints**: ensuring controls start/end at zero.
    - **Robustness**: Handling noise in control parameters.

## Results

Training logs are saved in the `logs/` directory. You can inspect the `training_log.jsonl` file or use `plot_results.py` to visualize performance.
