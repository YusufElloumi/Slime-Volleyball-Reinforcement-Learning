# Slime Volleyball: Reinforcement Learning in a Real Game

Turn a classic browser volleyball game into a compact reinforcement-learning testbed. This project preserves the original JavaScript game and its physics, then adds a headless training pipeline for **Naruto**, an agent learning to play against the built-in White Slime AI.

[Play the original browser game](http://marler8997.github.io/SlimeJavascript/SlimeVolleyballLegacy.html) · [View the original GitHub project](https://github.com/marler8997/SlimeJavascript)

## The Project

The goal is to connect a playable game to a complete, inspectable learning loop without replacing the game with a simplified simulator. The headless environment advances the existing game logic, observes the match, applies Naruto's action, and returns the next state and outcome. Training can run without a browser window; the original browser game remains available separately.

The result is a practical experiment in adapting an existing game: one codebase for gameplay, simulation, training, saved checkpoints, evaluation, and baseline comparisons.

## Learning Architecture

Naruto uses **tabular Q-learning** within a complete game-to-agent pipeline. Each decision cycle turns live simulation data into a compact observation, maps that observation to a reusable state key, selects a movement action, and updates the policy from the resulting game events:

- **Compact observations:** player position and velocity, relative ball position and velocity, and opponent position.
- **Discrete state:** continuous observations are bucketed into a tuple key so the agent can look up values in a Q-table.
- **Three actions:** stay neutral, move left, or move right. Jump control is outside this initial action set.
- **Epsilon-greedy exploration:** Naruto tries actions while learning, gradually reducing random exploration toward a configured minimum.
- **Reward shaping:** point outcomes and successful clears provide the main signals, with smaller movement and ball-progress rewards guiding behavior between points.
- **Reproducible evaluation:** run the learned policy greedily and compare it with random or heuristic agents in the same headless environment.

The design keeps the learning rule straightforward while giving the agent a meaningful control problem: it must interpret ball motion, position itself against an active opponent, and learn from both immediate progress and point outcomes. The full workflow includes state abstraction, reward shaping, exploration scheduling, resumable checkpoints, and controlled baseline evaluation, all connected to the original game's simulation.

## Run It

You need Python 3.10 or newer and Node.js. Install the Python test dependency with:

```powershell
python -m pip install -r requirements.txt
```

Train Naruto:

```powershell
python -m scripts.train --episodes 1000
```

Resume from the saved Q-table:

```powershell
python -m scripts.train --episodes 1000 --resume
```

Evaluate the learned agent, or compare a baseline:

```powershell
python -m scripts.evaluate --agent naruto --episodes 100
python -m scripts.evaluate --agent random --episodes 100
python -m scripts.evaluate --agent heuristic --episodes 100
```

Training checkpoints are written under `models/naruto/`; episode metrics are written to `results/naruto_training.csv`.

## Project Map

- `SlimeVolleyball.html`, `Slime.js`, and `SlimeAI.js`: browser game, physics, and built-in opponent.
- `RLEnvironment.js`: headless simulation and RL environment interface.
- `scripts/environment.py`: Python interface to the headless JavaScript game.
- `scripts/agent.py`: state discretization, Q-table agent, and baseline agents.
- `scripts/train.py` and `scripts/evaluate.py`: training and evaluation entry points.
- `models/naruto/` and `results/`: saved model and training metrics.

## Credit

The underlying **SlimeJavascript** game is the work of its original contributors, [Jonathan Marler (@marler8997)](https://github.com/marler8997) and [John Murphy (@chiefsmurph)](https://github.com/chiefsmurph), as listed in the [upstream repository](https://github.com/marler8997/SlimeJavascript). This project builds on that game; the reinforcement-learning environment, Naruto agent, and training/evaluation workflow are additions in this repository. Please refer to the upstream project for its license and original game attribution.
