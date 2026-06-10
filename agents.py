"""
Policy Networks and REINFORCE Agents
=====================================
Each agent uses a simple MLP policy trained with REINFORCE (policy gradient).
Shared reward signal encourages emergent communication / grounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple MLP policy: obs -> action probabilities."""

    def __init__(self, obs_size, action_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def select_action(self, obs):
        """Sample action from policy; return action and log-prob."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    Accumulates (log_prob, reward) pairs and updates at episode end.
    """

    def __init__(self, obs_size, action_size, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(obs_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def act(self, obs):
        action, log_prob = self.policy.select_action(obs)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        """Compute discounted returns and update policy with REINFORCE loss."""
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G  # gradient ascent on expected return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode buffer
        self.log_probs = []
        self.rewards = []

        return loss.item()
