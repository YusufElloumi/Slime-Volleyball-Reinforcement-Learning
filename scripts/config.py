from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.1
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.999
    optimistic_value: float = 1.5
    resume: bool = False
    frames_per_action: int = 5
    max_steps_per_episode: int = 300
    movement_reward_scale: float = 0.2
    reward_distance_scale: float = 0.1
    win_reward: float = 100.0
    clear_reward: float = 8.0
    loss_penalty: float = -80.0
    idle_penalty: float = 0.5
    ball_progress_reward: float = 1.5
    ball_progress_penalty: float = 0.5
    score_lane_bonus: float = 2.5
    score_lane_penalty: float = 1.0
    debug_state_logging: bool = True
    episodes: int = 1000
    log_every: int = 50
    checkpoint_every: int = 200
    model_dir: str = "models/naruto"
    results_path: str = "results/naruto_training.csv"
    bucket_player_x: float = 50.0
    bucket_ball_x: float = 25.0
    bucket_ball_y: float = 20.0
    bucket_ball_vx: float = 2.0
    bucket_ball_vy: float = 2.0
    bucket_opponent_x: float = 50.0
    bucket_player_y: float = 20.0
    bucket_score_diff: float = 1.0

    @classmethod
    def from_args(cls, argv=None):
        parser = argparse.ArgumentParser(description="Train Naruto against the existing White Slime AI.")
        parser.add_argument("--episodes", type=int, default=cls().episodes)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--alpha", type=float, default=cls().alpha)
        parser.add_argument("--gamma", type=float, default=cls().gamma)
        parser.add_argument("--epsilon", type=float, default=cls().epsilon)
        parser.add_argument("--min-epsilon", type=float, default=cls().min_epsilon)
        parser.add_argument("--epsilon-decay", type=float, default=cls().epsilon_decay)
        parser.add_argument("--log-every", type=int, default=cls().log_every)
        parser.add_argument("--checkpoint-every", type=int, default=cls().checkpoint_every)
        parser.add_argument("--frames-per-action", type=int, default=cls().frames_per_action)
        parser.add_argument("--model-dir", type=str, default=cls().model_dir)
        parser.add_argument("--results-path", type=str, default=cls().results_path)
        args = parser.parse_args(argv)

        return cls(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            min_epsilon=args.min_epsilon,
            epsilon_decay=args.epsilon_decay,
            optimistic_value=cls().optimistic_value,
            resume=args.resume,
            frames_per_action=args.frames_per_action,
            episodes=args.episodes,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            model_dir=args.model_dir,
            results_path=args.results_path,
            bucket_player_x=cls().bucket_player_x,
            bucket_ball_x=cls().bucket_ball_x,
            bucket_ball_y=cls().bucket_ball_y,
            bucket_ball_vx=cls().bucket_ball_vx,
            bucket_ball_vy=cls().bucket_ball_vy,
            bucket_opponent_x=cls().bucket_opponent_x,
            bucket_player_y=cls().bucket_player_y,
            bucket_score_diff=cls().bucket_score_diff,
            )

    def to_dict(self):
        return asdict(self)


def ensure_dirs(*paths):
    for raw in paths:
        Path(raw).mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = TrainingConfig()
