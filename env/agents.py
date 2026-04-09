"""
Agents — PPO (optional), RuleBased, Random
All agents: act(observation) -> int action
Fallback chain: PPO -> RuleBased -> Random
"""

import os
import random

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    pass


class RandomAgent:
    """Uniform random action. Zero dependencies."""
    name = "RandomAgent"

    def __init__(self, num_actions=7, seed=42):
        self.num_actions = max(num_actions, 1)
        self.rng = random.Random(seed)

    def act(self, obs):
        return self.rng.randint(0, self.num_actions - 1)


class RuleBasedAgent:
    """Heuristic: quarantine most-infected city, vaccinate most-susceptible."""
    name = "RuleBasedAgent"

    def __init__(self):
        self._step = 0

    def act(self, obs):
        self._step += 1
        try:
            if not obs or len(obs) < 12:
                return 0
            infections = [obs[1], obs[5], obs[9]]
            worst = infections.index(max(infections))
            if infections[worst] > 0.005:
                return worst + 1
            susceptibles = [obs[0], obs[4], obs[8]]
            best = susceptibles.index(max(susceptibles))
            return best + 4
        except Exception:
            return 0


class PPOAgent:
    """PPO agent. Requires torch + checkpoint. Fails fast for caller to catch."""
    name = "PPOAgent"

    def __init__(self, obs_size=12, num_actions=7, checkpoint_path=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")

        self.model = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            key = next(
                (k for k in ("model_state_dict", "state_dict", "model") if k in state),
                None,
            )
            self.model.load_state_dict(state[key] if key else state)
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def act(self, obs):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            return torch.argmax(self.model(t), dim=-1).item()


def create_agent(checkpoint_path=None, num_actions=7, seed=42):
    """Create best available agent with automatic fallback. Returns (agent, name)."""
    if checkpoint_path:
        try:
            agent = PPOAgent(obs_size=12, num_actions=num_actions, checkpoint_path=checkpoint_path)
            return agent, "PPOAgent"
        except Exception:
            pass

    try:
        return RuleBasedAgent(), "RuleBasedAgent"
    except Exception:
        pass

    return RandomAgent(num_actions, seed=seed), "RandomAgent"
