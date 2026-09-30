from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.agent import HeuristicAgent, RandomAgent, TabularQAgent
    from scripts.config import TrainingConfig
    from scripts.environment import NodeGameEnvironment
except ImportError:
    from agent import HeuristicAgent, RandomAgent, TabularQAgent
    from config import TrainingConfig
    from environment import NodeGameEnvironment


def evaluate(model_path: str, episodes: int = 100, mode: str = "naruto"):
    config = TrainingConfig()
    env = NodeGameEnvironment()
    if mode == "naruto":
        agent = TabularQAgent.load(model_path, config=config)
    elif mode == "heuristic":
        agent = HeuristicAgent(config=config)
    else:
        agent = RandomAgent(config=config)

    wins = 0
    losses = 0
    total_reward = 0.0
    total_clears = 0
    total_length = 0

    for ep in range(1, episodes + 1):
        state = env.reset(server_side=(ep % 2 == 0), reset_scores=True)
        done = False
        reward_total = 0.0
        clears = 0
        steps = 0

        while not done:
            action = agent.choose_action(state, epsilon=0.0, force_greedy=True)
            result = env.step(action)
            reward_total += float(result.get("reward", 0.0))
            total_reward += float(result.get("reward", 0.0))
            if result.get("clear_detected"):
                clears += 1
            total_clears += 1 if result.get("clear_detected") else 0
            state = result.get("next_state", state)
            steps += 1
            done = bool(result.get("done", False))

        if result.get("winner") == "naruto":
            wins += 1
        elif result.get("winner") == "white":
            losses += 1
        total_length += steps

    avg_reward = total_reward / episodes if episodes else 0.0
    avg_clears = total_clears / episodes if episodes else 0.0
    avg_length = total_length / episodes if episodes else 0.0
    win_rate = (wins / episodes) * 100 if episodes else 0.0

    print(f"Agent: {mode}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f} | Average Clears: {avg_clears:.2f} | Average Episode Length: {avg_length:.1f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Naruto, a random baseline, or a heuristic baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--agent", choices=["naruto", "random", "heuristic"], default="naruto")
    parser.add_argument("--model-path", type=str, default="models/naruto/naruto_q_table.json")
    args = parser.parse_args()
    evaluate(args.model_path, episodes=args.episodes, mode=args.agent)


if __name__ == "__main__":
    main()
