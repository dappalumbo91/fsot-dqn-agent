# FSOT-DQN Agent

A Python-based reinforcement learning agent using Deep Q-Network (DQN) with custom enhancements like the FSOT Modulator for neural plasticity, curriculum learning for multi-dimensional environments, and meta-control for sub-goal selection.

## Features
- **FSOT Modulator**: Handles synaptic adjustments with mathematical constants (e.g., golden ratio, Euler's number) and oscillations.
- **N-Dimensional Environment**: Simulates movement in 3D+ spaces with rewards for progress toward a goal.
- **Memory Buffer**: Prioritized experience replay with offline pre-filling.
- **Policy Network**: Includes predictive coding and dynamic layer growth/pruning.
- **Meta Controller**: Hierarchical RL for high-level sub-goal decisions.
- **Curriculum**: Progressively increases environment complexity (dimensions).
- Built with PyTorch, NumPy, mpmath, and other libraries for reproducibility.

## Installation
1. Clone the repository:2. Install dependencies (Python 3.12+ recommended):Note: The code uses pre-installed libraries like PyTorch, NumPy, and mpmath. No additional pip installs are needed beyond these.

## Usage
1. Save the code as `fsot_dqn.py` (rename from "FSOT-DQN (3).py.txt" if needed).
2. Run the script:- It will train the agent over 40 episodes, logging rewards and progress.
- Output includes episode rewards, steps, and whether the goal was reached.
- Press Enter at the end to exit.

Example output snippet:
## Configuration and Customization
- Adjust hyperparameters in the code (e.g., `Curriculum` for max dimensions, `FSOT_DQNAgent` for learning rate/epsilon).
- For higher dimensions, modify `Curriculum(max_dims=...)`.
- The code includes garbage collection (`gc.collect()`) for memory efficiency during long runs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Feel free to open issues or pull requests for improvements, bug fixes, or extensions (e.g., adding visualization with Matplotlib).

## Acknowledgments
- Built with inspiration from reinforcement learning concepts like DQN and hierarchical RL.
- Uses mathematical libraries for precise computations in the FSOT Modulator.
