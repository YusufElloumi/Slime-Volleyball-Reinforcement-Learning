from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

try:
    from scripts.agent import TabularQAgent
    from scripts.config import TrainingConfig, ensure_dirs
    from scripts.environment import NodeGameEnvironment
except ImportError:
    from agent import TabularQAgent
    from config import TrainingConfig, ensure_dirs
    from environment import NodeGameEnvironment


def compute_shaped_reward(config: TrainingConfig, state, next_state, result, action=None):
    base_reward = float(result.get("reward", 0.0))
    if not state:
        return base_reward

    prev_ball_x = float(state.get("ball_x", 0.0))
    prev_ball_y = float(state.get("ball_y", 0.0))
    next_ball_x = float(next_state.get("ball_x", prev_ball_x))
    next_ball_y = float(next_state.get("ball_y", prev_ball_y))

    prev_distance = abs(prev_ball_x) + 0.3 * abs(prev_ball_y)
    next_distance = abs(next_ball_x) + 0.3 * abs(next_ball_y)
    movement_bonus = 0.0
    if next_distance < prev_distance:
        movement_bonus += config.movement_reward_scale
    else:
        movement_bonus -= config.reward_distance_scale

    if next_ball_x > prev_ball_x:
        movement_bonus += config.ball_progress_reward
    else:
        movement_bonus -= config.ball_progress_penalty

    if next_ball_x > 200:
        movement_bonus += config.score_lane_bonus
    elif next_ball_x < -200:
        movement_bonus -= config.score_lane_penalty

    if action == 0 and abs(next_ball_x) < 20 and not result.get("clear_detected"):
        movement_bonus -= config.idle_penalty

    if result.get("clear_detected"):
        movement_bonus += config.clear_reward

    winner = result.get("winner")
    if winner == "left":
        movement_bonus += config.win_reward
    elif winner == "right":
        movement_bonus += config.loss_penalty

    return base_reward + movement_bonus


def log_action_summary(action_counts, last_ball_x, last_ball_y):
    action_text = ", ".join(f"{k}:{v}" for k, v in sorted(action_counts.items()))
    print(f"Action counts: {action_text} | last_ball_x={last_ball_x:.1f} | last_ball_y={last_ball_y:.1f}")


def log_state_before_action(ep, step, state, action):
    compact = {
        "player_x": state.get("player_x"),
        "player_y": state.get("player_y"),
        "ball_x": state.get("ball_x"),
        "ball_y": state.get("ball_y"),
        "ball_vx": state.get("ball_velocity_x", state.get("relative_ball_vx")),
        "ball_vy": state.get("ball_velocity_y", state.get("relative_ball_vy")),
        "side": state.get("side"),
        "player_on_ground": state.get("player_on_ground"),
        "action": action,
    }
    print(f"Episode {ep} step {step}: action={action} state={compact}")


def train(config: TrainingConfig):
    ensure_dirs(config.model_dir, config.results_path.rsplit("/", 1)[0] if "/" in config.results_path else ".")

    env = NodeGameEnvironment()
    if config.resume and Path(config.model_dir, "naruto_q_table.json").exists():
        agent = TabularQAgent.load(Path(config.model_dir, "naruto_q_table.json"), config=config)
        print(f"Loaded checkpoint from {config.model_dir}/naruto_q_table.json")
    else:
        agent = TabularQAgent(config=config)

    wins = 0
    losses = 0
    total_reward = 0.0
    total_clears = 0
    episode_length = 0
    results_rows = []

    for ep in range(1, config.episodes + 1):
        state = env.reset(server_side=(ep % 2 == 0), reset_scores=True)
        done = False
        episode_reward = 0.0
        clear_count = 0
        steps = 0
        action = agent.choose_action(state, epsilon=agent.epsilon)
        action_frames = 0
        result = None
        action_counts = {0: 0, 1: 0, 2: 0}
        last_ball_x = float(state.get("ball_x", 0.0))
        last_ball_y = float(state.get("ball_y", 0.0))

        while not done and steps < config.max_steps_per_episode:
            if config.debug_state_logging and (ep <= 3 or ep % config.log_every == 0):
                log_state_before_action(ep, steps, state, action)
            result = env.step(action)
            next_state = result.get("next_state", state)
            reward = compute_shaped_reward(config, state, next_state, result, action)
            episode_reward += float(reward)
            if result.get("clear_detected"):
                clear_count += 1
            agent.update(state, action, float(reward), next_state, bool(result.get("done", False)))
            state = next_state
            action_counts[int(action)] = action_counts.get(int(action), 0) + 1
            last_ball_x = float(state.get("ball_x", 0.0))
            last_ball_y = float(state.get("ball_y", 0.0))
            steps += 1
            done = bool(result.get("done", False))
            action_frames += 1

            if not done and action_frames >= config.frames_per_action:
                action = agent.choose_action(state, epsilon=agent.epsilon)
                action_frames = 0

        if result is None:
            result = {"winner": None, "reward": 0.0, "clear_detected": False}

        if ep % config.log_every == 0 or ep == 1:
            log_action_summary(action_counts, last_ball_x, last_ball_y)

        if result.get("winner") == "naruto":
            wins += 1
        elif result.get("winner") == "white":
            losses += 1

        total_reward += episode_reward
        total_clears += clear_count
        episode_length += steps
        agent.decay_epsilon()
        agent.episodes += 1

        if ep % config.log_every == 0 or ep == 1:
            win_rate = (wins / ep) * 100 if ep else 0.0
            avg_reward = total_reward / ep if ep else 0.0
            avg_clears = total_clears / ep if ep else 0.0
            print(
                f"Episode: {ep} | Wins: {wins} | Losses: {losses} | "
                f"Win Rate: {win_rate:.1f}% | Avg Reward: {avg_reward:.2f} | "
                f"Avg Clears: {avg_clears:.2f} | Epsilon: {agent.epsilon:.3f}"
            )

        if ep % config.checkpoint_every == 0 or ep == 1:
            saved = agent.save(config.model_dir)
            print(f"Checkpoint saved: {saved}")

        results_rows.append({
            "episode": ep,
            "reward": episode_reward,
            "wins": 1 if result.get("winner") == "naruto" else 0,
            "losses": 1 if result.get("winner") == "white" else 0,
            "clears": clear_count,
            "episode_length": steps,
            "epsilon": agent.epsilon,
        })

    agent.save(config.model_dir)
    print(
        f"Training finished. Final Wins: {wins} | Final Losses: {losses} | "
        f"Final Win Rate: {(wins / max(config.episodes,1)) * 100:.1f}% | final epsilon: {agent.epsilon:.3f}"
    )

    try:
        with open(config.results_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode", "reward", "wins", "losses", "clears", "episode_length", "epsilon"])
            writer.writeheader()
            writer.writerows(results_rows)
    except Exception:
        pass

    env.close()
    return agent


def main():
    config = TrainingConfig.from_args()
    train(config)


if __name__ == "__main__":
    main()
