from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from scripts.config import TrainingConfig
except ImportError:
    from config import TrainingConfig

ACTIONS = (0, 1, 2)


class Discretizer:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()

    def bucketize(self, value, bucket_size):
        if value is None or not isinstance(value, (int, float)):
            return 0
        if bucket_size <= 0:
            return 0
        return math.floor(float(value) / float(bucket_size))

    def state_key(self, state: dict) -> tuple:
        ball_x = state.get("relative_ball_x", state.get("ball_x", 0))
        ball_y = state.get("relative_ball_y", state.get("ball_y", 0))
        ball_vx = state.get("relative_ball_vx", state.get("ball_velocity_x", 0))
        ball_vy = state.get("relative_ball_vy", state.get("ball_velocity_y", 0))
        opponent_x = state.get("opponent_x", state.get("relative_opponent_x", 0))
        return (
            self.bucketize(state.get("player_x", 0), self.config.bucket_player_x),
            self.bucketize(state.get("player_velocity_x", 0), 2.0),
            self.bucketize(ball_x, self.config.bucket_ball_x),
            self.bucketize(ball_y, self.config.bucket_ball_y),
            self.bucketize(ball_vx, self.config.bucket_ball_vx),
            self.bucketize(ball_vy, self.config.bucket_ball_vy),
            self.bucketize(opponent_x, self.config.bucket_opponent_x),
        )


class TabularQAgent:
    def __init__(self, config: TrainingConfig | None = None, q_table: dict | None = None):
        self.config = config or TrainingConfig()
        self.discretizer = Discretizer(self.config)
        self.q_table: Dict[tuple, Dict[int, float]] = q_table.copy() if q_table else {}
        self.epsilon = self.config.epsilon
        self.episodes = 0

    @staticmethod
    def _serialize_key(key):
        return json.dumps(key)

    @staticmethod
    def _deserialize_key(raw):
        return tuple(json.loads(raw))

    def state_key(self, state: dict) -> tuple:
        return self.discretizer.state_key(state)

    def ensure_row(self, state_key):
        row = self.q_table.setdefault(state_key, {})
        for action in ACTIONS:
            row.setdefault(action, self.config.optimistic_value)
        return row

    def get_q_value(self, state_key, action):
        row = self.q_table.get(state_key)
        if row is None:
            return self.config.optimistic_value
        return row.get(action, self.config.optimistic_value)

    def random_action(self, state=None):
        return random.choice(ACTIONS)

    def directional_action(self, state: dict) -> int:
        ball_x = float(state.get("relative_ball_x", state.get("ball_x", 0.0)))
        side = str(state.get("side", "left")).lower()
        if side == "right":
            ball_x *= -1.0
        if ball_x > 30:
            return 2
        if ball_x < -30:
            return 1
        if state.get("player_on_ground") is False and state.get("ball_y", 0) > 50:
            return 2 if ball_x >= 0 else 1
        return 0

    def choose_action(self, state: dict, epsilon: float | None = None, force_greedy: bool = False):
        if epsilon is None:
            epsilon = self.epsilon

        directional = self.directional_action(state)
        if force_greedy or random.random() >= epsilon:
            state_key = self.state_key(state)
            row = self.ensure_row(state_key)
            best_value = max(row.values())
            best_actions = [action for action, value in row.items() if value == best_value]
            if directional in row and row[directional] >= best_value - 0.01:
                return directional
            return random.choice(best_actions)
        return self.random_action(state)

    def update(self, state, action, reward, next_state, done):
        current_key = self.state_key(state)
        next_key = self.state_key(next_state)
        row = self.ensure_row(current_key)
        current_value = row.get(action, self.config.optimistic_value)
        target = reward
        if not done:
            next_row = self.ensure_row(next_key)
            target += self.config.gamma * max(next_row.values())
        row[action] = current_value + self.config.alpha * (target - current_value)
        self.q_table[current_key] = row

    def decay_epsilon(self):
        if self.epsilon <= self.config.min_epsilon:
            self.epsilon = self.config.min_epsilon
            return
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def save(self, path):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "epsilon": self.epsilon,
            "episodes": self.episodes,
            "config": self.config.to_dict(),
            "q_table": {
                self._serialize_key(key): {str(action): float(value) for action, value in row.items()}
                for key, row in self.q_table.items()
            },
        }
        checkpoint_path = root / "naruto_q_table.json"
        checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(checkpoint_path)

    @classmethod
    def load(cls, path, config: TrainingConfig | None = None):
        file_path = Path(path)
        if not file_path.exists():
            return cls(config=config)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        q_table = {}
        for raw_key, row in payload.get("q_table", {}).items():
            parsed = cls._deserialize_key(raw_key)
            q_table[parsed] = {int(action): float(value) for action, value in row.items()}
        agent = cls(config=config or TrainingConfig(**payload.get("config", {})), q_table=q_table)
        agent.epsilon = payload.get("epsilon", agent.config.epsilon)
        agent.episodes = payload.get("episodes", 0)
        return agent


class RandomAgent:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()

    def choose_action(self, state: dict, epsilon: float | None = None, force_greedy: bool = False):
        return random.choice(ACTIONS)

    def save(self, path):
        return path

    @classmethod
    def load(cls, path, config: TrainingConfig | None = None):
        return cls(config=config)


class HeuristicAgent:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()

    def choose_action(self, state: dict, epsilon: float | None = None, force_greedy: bool = False):
        ball_x = float(state.get("relative_ball_x", state.get("ball_x", 0.0)))
        side = str(state.get("side", "left")).lower()
        player_x = float(state.get("player_x", 0.0))
        player_on_ground = bool(state.get("player_on_ground", True))

        if side == "right":
            ball_x *= -1.0

        if ball_x > 35:
            return 2
        if ball_x < -35:
            return 1
        if not player_on_ground and float(state.get("ball_y", 0.0)) > 30:
            return 2 if ball_x >= 0 else 1
        if abs(ball_x) < 20 and player_on_ground and float(state.get("ball_y", 0.0)) <= 30:
            return 0
        return 0

    def save(self, path):
        return path

    @classmethod
    def load(cls, path, config: TrainingConfig | None = None):
        return cls(config=config)
