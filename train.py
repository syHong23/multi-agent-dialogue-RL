"""
Training Loop — Batched REINFORCE
===================================
Batch updates (32 episodes per update) for stable multi-agent training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# ── Environment ──────────────────────────────────────────────────────────────

class DialogueGroundingEnv:
    """
    Referential communication: Speaker sees target, sends message, Listener picks item.
    Shared reward: +1 if correct, 0 otherwise.
    """
    def __init__(self, num_candidates=4, vocab_size=8):
        self.num_candidates = num_candidates
        self.vocab_size = vocab_size

    def sample(self, batch_size):
        """Sample a batch of episodes. Returns target indices."""
        targets = np.random.randint(0, self.num_candidates, size=batch_size)
        return targets


# ── Policy Networks ───────────────────────────────────────────────────────────

class SpeakerPolicy(nn.Module):
    """Speaker: sees target index → produces discrete message token."""
    def __init__(self, num_candidates, vocab_size, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(num_candidates, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, vocab_size)
        )

    def forward(self, target_idx):
        x = self.embed(target_idx)
        return F.softmax(self.fc(x), dim=-1)


class ListenerPolicy(nn.Module):
    """Listener: sees message token → selects candidate."""
    def __init__(self, vocab_size, num_candidates, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_candidates)
        )

    def forward(self, message):
        x = self.embed(message)
        return F.softmax(self.fc(x), dim=-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(num_episodes=10000, batch_size=64, lr=3e-3,
          num_candidates=4, vocab_size=8, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = DialogueGroundingEnv(num_candidates, vocab_size)
    speaker = SpeakerPolicy(num_candidates, vocab_size)
    listener = ListenerPolicy(vocab_size, num_candidates)

    opt_s = torch.optim.Adam(speaker.parameters(), lr=lr)
    opt_l = torch.optim.Adam(listener.parameters(), lr=lr)

    reward_history = []
    num_batches = num_episodes // batch_size

    print("Multi-Agent Dialogue Grounding — Batched REINFORCE")
    print(f"Candidates: {num_candidates} | Vocab: {vocab_size} | Batch: {batch_size}")
    print(f"Random baseline: {1/num_candidates:.0%}\n")

    for batch_idx in range(num_batches):
        targets = torch.LongTensor(env.sample(batch_size))

        # Speaker selects message
        s_probs = speaker(targets)
        s_dist = Categorical(s_probs)
        messages = s_dist.sample()
        s_log_probs = s_dist.log_prob(messages)

        # Listener selects candidate
        l_probs = listener(messages)
        l_dist = Categorical(l_probs)
        choices = l_dist.sample()
        l_log_probs = l_dist.log_prob(choices)

        # Shared reward
        rewards = (choices == targets).float()
        acc = rewards.mean().item()
        reward_history.append(acc)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # REINFORCE loss (both agents)
        loss_s = -(s_log_probs * rewards).mean()
        loss_l = -(l_log_probs * rewards).mean()
        loss = loss_s + loss_l

        opt_s.zero_grad()
        opt_l.zero_grad()
        loss.backward()
        opt_s.step()
        opt_l.step()

        if (batch_idx + 1) % 50 == 0:
            recent = np.mean(reward_history[-50:])
            print(f"Batch {batch_idx+1:4d}/{num_batches} | "
                  f"Accuracy: {recent:.2%}")

    final_acc = np.mean(reward_history[-50:])
    print(f"\nFinal accuracy: {final_acc:.2%} "
          f"(random baseline: {1/num_candidates:.0%})")

    plot_results(reward_history, num_candidates)
    return speaker, listener


def plot_results(history, num_candidates):
    window = 20
    smoothed = [np.mean(history[max(0, i-window):i+1]) for i in range(len(history))]
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, color="steelblue", label=f"Rolling accuracy (w={window})")
    plt.axhline(1/num_candidates, color="red", linestyle="--",
                label=f"Random baseline ({1/num_candidates:.0%})")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Multi-Agent Dialogue Grounding — REINFORCE Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("Saved training_curve.png")


if __name__ == "__main__":
    train()
