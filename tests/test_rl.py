import unittest

try:
    from scripts.agent import HeuristicAgent, TabularQAgent
    from scripts.config import TrainingConfig
    from scripts.environment import Discretizer
    from scripts.train import compute_shaped_reward
except ImportError:
    from agent import HeuristicAgent, TabularQAgent
    from config import TrainingConfig
    from environment import Discretizer
    from train import compute_shaped_reward


class TestTabularRL(unittest.TestCase):
    def test_q_learning_update(self):
        config = TrainingConfig(alpha=0.2, gamma=0.9, epsilon=0.0, min_epsilon=0.0, epsilon_decay=0.99)
        agent = TabularQAgent(config)
        state = {"player_x": 200, "player_velocity_x": 0, "ball_x": 100, "ball_y": 200, "ball_velocity_x": 4, "ball_velocity_y": 2, "opponent_x": 200}
        next_state = {"player_x": 210, "player_velocity_x": 0, "ball_x": 150, "ball_y": 150, "ball_velocity_x": 3, "ball_velocity_y": 1, "opponent_x": 200}
        action = 2

        q_before = agent.get_q_value(agent.state_key(state), action)
        agent.update(state, action, 1.5, next_state, False)
        q_after = agent.get_q_value(agent.state_key(state), action)

        self.assertIsInstance(q_after, float)
        self.assertNotEqual(q_before, q_after)

    def test_epsilon_decay_never_below_min(self):
        config = TrainingConfig(alpha=0.2, gamma=0.9, epsilon=0.3, min_epsilon=0.1, epsilon_decay=0.5)
        agent = TabularQAgent(config)
        agent.epsilon = 0.05
        agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.1)

    def test_discretizer_buckets_state(self):
        state = {"player_x": 200, "player_velocity_x": 0, "ball_x": 350, "ball_y": 180, "ball_velocity_x": 4, "ball_velocity_y": -2, "opponent_x": 700}
        key = Discretizer().state_key(state)
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 7)

    def test_random_agent_behavior(self):
        config = TrainingConfig()
        agent = TabularQAgent(config)
        state = {"player_x": 250, "player_velocity_x": 0, "ball_x": 320, "ball_y": 30, "ball_velocity_x": 2, "ball_velocity_y": 0, "opponent_x": 680}
        action = agent.random_action(state)
        self.assertIn(action, (0, 1, 2))

    def test_greedy_action_uses_best_q(self):
        config = TrainingConfig(alpha=0.2, gamma=0.9, epsilon=0.0)
        agent = TabularQAgent(config)
        state = {"player_x": 250, "player_velocity_x": 0, "ball_x": 320, "ball_y": 30, "ball_velocity_x": 2, "ball_velocity_y": 0, "opponent_x": 680}
        key = agent.state_key(state)
        agent.q_table[key] = {0: 0.1, 1: 0.4, 2: 0.4}
        action = agent.choose_action(state, force_greedy=True)
        self.assertIn(action, (1, 2))

    def test_movement_reward_encourages_closer_ball(self):
        config = TrainingConfig()
        state = {"player_x": 200, "ball_x": -80, "ball_y": 30}
        next_state = {"player_x": 210, "ball_x": -30, "ball_y": 20}
        result = {"reward": 0.0, "clear_detected": False}
        reward = compute_shaped_reward(config, state, next_state, result)
        self.assertGreater(reward, 0.0)

    def test_win_reward_is_large(self):
        config = TrainingConfig()
        state = {"player_x": 200, "ball_x": -10, "ball_y": 10}
        next_state = {"player_x": 200, "ball_x": -5, "ball_y": 5}
        result = {"reward": 0.0, "clear_detected": False, "winner": "left"}
        reward = compute_shaped_reward(config, state, next_state, result)
        self.assertGreater(reward, 40.0)

    def test_directional_action_moves_toward_ball(self):
        config = TrainingConfig()
        agent = TabularQAgent(config)
        left_state = {"player_x": 200, "side": "left", "relative_ball_x": 80, "player_on_ground": True}
        right_state = {"player_x": 800, "side": "right", "relative_ball_x": -80, "player_on_ground": True}
        self.assertEqual(agent.directional_action(left_state), 2)
        self.assertEqual(agent.directional_action(right_state), 2)

    def test_heuristic_agent_moves_toward_ball(self):
        config = TrainingConfig()
        agent = HeuristicAgent(config)
        left_state = {"player_x": 200, "side": "left", "relative_ball_x": 80, "player_on_ground": True}
        right_state = {"player_x": 800, "side": "right", "relative_ball_x": -80, "player_on_ground": True}
        self.assertEqual(agent.choose_action(left_state), 2)
        self.assertEqual(agent.choose_action(right_state), 2)


if __name__ == "__main__":
    unittest.main()
