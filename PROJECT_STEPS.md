# Slime Volleyball RL Project Steps

## Step 1: Understand the existing game

Before touching the RL layer, read the real game code and confirm the physics loop, scoring rules, and AI movement. In this project, the playable game already works in the browser and we keep that behavior intact.

Expect to see:
- the ball and slime objects stored as global variables
- the main update loop calling physics functions per frame
- the White Slime AI using `move()` and a movement state machine
- scoring when the ball falls below the court

## Step 2: Expose the simulation

The RL loop should not rely on a browser window. Instead, we expose the same physics logic through a headless environment that runs the real update functions and the real White Slime controller.

Expect to see:
- a reset that restores the starting positions
- a state snapshot from the left or right player perspective
- a `step(action)` call that advances the real game simulation

## Step 3: Create the RL environment

The environment wraps the underlying game and returns the next state, reward, and a `done` flag. This is the bridge between the real game and the learning agent.

Expect to see:
- one side controlled by Naruto
- the White Slime fixed on the other side
- the ball and slime objects updated by the same physics code used in the browser

## Step 4: Define Naruto's actions

Naruto gets exactly three actions:
- 0 = neutral
- 1 = left
- 2 = right

This keeps the learning problem simple and matches the tabular Q-learning setup.

Expect to see:
- no direct jump control in the first version
- all actions applied through the existing horizontal movement logic
- no fake physics invented for movement

## Step 5: Define the state

Use a compact state that captures the key tactical information: Naruto's position and velocity, the ball position and velocity, and the opponent's x-position.

Expect to see:
- a state vector like player_x, ball_x, ball_y, ball_velocity_x, ball_velocity_y, opponent_x
- enough information to judge where the ball is headed and whether Naruto should move left or right

## Step 6: Discretize the state

Tabular Q-learning cannot work on raw continuous values. We bucket the state so each measurement maps to a discrete index.

Expect to see:
- bucket sizes configured in the training config
- a tuple key used as the dictionary key for the Q-table
- the same key structure always reused for training and evaluation

## Step 7: Implement rewards

The initial reward setup is:
- +1 when Naruto clears the ball over the net
- +10 when Naruto wins the point
- -10 when Naruto loses the point
- 0 otherwise

A clear must trigger once per successful crossing event, not every frame after the ball has crossed.

Expect to see:
- event detection only when the ball actually crosses the net from Naruto's side to the opponent's side
- win/loss rewards at point completion
- no spammed clear rewards while the ball sits on the far side

## Step 8: Implement the random baseline

Before trusting the learner, compare it against a random policy.

Expect to see:
- equal probability over left, neutral, and right
- the same headless environment used for evaluation
- a baseline that proves whether the learned policy is actually beating random behavior

## Step 9: Implement Q-learning

Naruto updates a Q-table using the standard update rule:

Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

Expect to see:
- the state key used to read/write Q-values
- the action selected using the Q-table
- the update applied after each step, not after every frame in the browser

## Step 10: Implement epsilon-greedy exploration

At the start, Naruto explores broadly. Over time, epsilon decays until it reaches a minimum value.

Expect to see:
- `epsilon` set in the config
- decay applied after an episode or update cycle
- `min_epsilon` enforced so exploration never disappears entirely

## Step 11: Train Naruto

Training runs with a headless loop that repeatedly resets, chooses actions, steps the environment, and updates Q-values.

Expect to see:
- training output only every few episodes, not every frame
- rolling statistics such as win rate and average reward
- checkpoints written periodically

## Step 12: Save checkpoints

Naruto's Q-table should be saved to disk so the run is resumable.

Expect to see:
- a `models/naruto` directory
- a JSON checkpoint storing the Q-table and epsilon
- enough metadata to resume training later without starting over

## Step 13: Resume training

Use the saved checkpoint to continue learning from the same point.

Expect to see:
- `python train.py --resume`
- training picks up the last saved Q-table and epsilon
- any CLI episode limit continuing from that checkpoint

## Step 14: Evaluate Naruto with epsilon = 0

Evaluation must not explore. It should greedily select the best move from the learned Q-table.

Expect to see:
- the training config or evaluation logic forcing greedy action selection
- no random exploration during evaluation
- stats like win rate, average reward, and episode length

## Step 15: Compare Naruto against the random baseline

Evaluate Naruto and random under identical conditions.

Expect to see:
- Naruto winning more often if the learning is working
- average clears and reward improving over time
- a clear difference between learned performance and random play

## Step 16: Check whether the agent is learning

The agent is learning when training metrics improve over time instead of staying flat.

Expect to see:
- win rate trending upward
- average reward trending upward
- clear count increasing when the agent has found better movement patterns

If the model does not improve, revisit the state representation, reward function, or action timing.
