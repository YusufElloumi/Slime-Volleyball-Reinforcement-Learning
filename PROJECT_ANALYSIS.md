# Project Analysis

## 1. Game objects and state

The core game state is stored as global variables in the browser script in [SlimeVolleyball.html](SlimeVolleyball.html):

- `ball` stores position, radius, and velocity.
- `slime1` and `slime2` represent the left and right slimes.
- `slime1Score` and `slime2Score` track points.
- `initRound(server)` resets the ball and both players to their starting positions.
- `updateSlime(...)` applies horizontal movement and gravity-like vertical motion.
- `updateBall()` advances ball physics and handles wall and score logic.

The same state is also assumed by the headless runtime in [RLEnvironment.js](RLEnvironment.js), which is the foundation for the RL wrapper.

## 2. White Slime AI

The built-in AI is implemented in [SlimeAI.js](SlimeAI.js):

- `newSlimeAI(onLeft, name)` creates the AI state object.
- `setPatheticWhiteSlime(ai)` defines the actual decision logic.
- `ai.move()` decides the movement intent and sets:
  - `ai.movement` = `0`, `1`, or `2`
  - `ai.jumpSet` = whether to jump
- The browser loop calls `updateOpponentAI()` and converts the AI's intent into actual slime velocity changes.

This is critical because the RL environment must use the real AI logic rather than a simplified imitation.

## 3. Movement and input processing

Player input is handled by [Input.js](Input.js) and the browser code in [SlimeVolleyball.html](SlimeVolleyball.html):

- `keysDown[...]` stores keyboard state.
- `updateSlimeVelocities(s, left, right, up)` converts key presses into `velocityX` and `velocityY`.
- `updateFrame()` applies the left player's manual input and then the opponent AI update.
- `updateSlime(s, leftLimit, rightLimit)` applies movement and gravity to the slime object.

The RL path uses the same game physics by directly assigning `velocityX` to the target slime instead of using the browser keyboard layer.

## 4. Ball and collision behavior

The ball is controlled by `updateBall()` in the browser script and the matching logic in [RLEnvironment.js](RLEnvironment.js):

- Ball physics: position + velocity, then gravity is applied.
- Collision with slimes: `collisionBallSlime(s)` adjusts ball position and velocity based on impact.
- Wall collisions: if the ball hits the left or right edge, it reverses `velocityX`.
- Net collision: the central net acts like a wall for the ball.

This means the RL environment should operate on the real simulation rather than a custom approximation.

## 5. Points and round reset

A point is scored when the ball falls below the court:

- `if (ball.y < 0)`
- If `ball.x > 500`, the left player scores; otherwise the right player scores.
- The game calls `initRound(server)` to reset the fight.

The score variables are global and can be checked directly, which is useful for reward and termination logic.

## 6. Match reset and game loop

The browser loop in [SlimeVolleyball.html](SlimeVolleyball.html) calls `gameIteration()` every 20 ms with `setInterval(gameIteration, 20)`. That means the game is effectively running at about 50 Hz, but headless RL can still drive the same simulation step-by-step.

The RL environment in [RLEnvironment.js](RLEnvironment.js) wraps those same global functions and exposes:

- `reset()`
- `get_state()`
- `step(action)`

The environment is therefore a thin RL wrapper over the real physics system.

## 7. Relevant RL integration points

The project already contains a useful starting point in [RLEnvironment.js](RLEnvironment.js):

- `createEnvironment(options)` creates a headless game context.
- `getStateForSide(side)` collects a compact observation.
- `stateKeyFromState(state)` buckets the state for tabular Q-learning.
- `createOptimisticEpsilonGreedyAgent(env, options)` implements Q-learning with epsilon-greedy policy selection.

This implementation is intentionally adapted to the required Naruto-vs-White-Slime milestone rather than rewriting the core game.

## 8. Integration plan for Naruto

Naruto uses the left side of the court (`slime1`) and the built-in White Slime remains fixed on the right side (`slime2`).

The RL design in this repository keeps the real game simulation untouched and adds a narrow Python-facing layer around it:

- [agent.py](agent.py): Q-table logic, discretization, epsilon-greedy policy, update rule.
- [environment.py](environment.py): Python wrapper accessing the headless JS environment.
- [train.py](train.py): episode loop, checkpointing, and resume logic.
- [evaluate.py](evaluate.py): evaluation with greedy actions and optional random baseline.
- [config.py](config.py): hyperparameter and state bucket configuration.

This preserves the original browser gameplay while enabling headless training and evaluation.
