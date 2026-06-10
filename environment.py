"""
Multi-Agent Dialogue Grounding Environment
==========================================
Two agents (Speaker and Listener) collaborate to solve a referential
communication task — a classic grounding task in dialogue RL research.

Task: Speaker sees a target item; Listener must identify it from a set
of candidates based on a message produced by the Speaker.
This directly mirrors the 'grounding' sub-task in the Orange PhD thesis topic.
"""

import numpy as np
import torch


class DialogueGroundingEnv:
    """
    Referential communication environment for two agents.
    
    - Speaker: observes target item + candidate set, produces a message (action)
    - Listener: observes candidate set + Speaker's message, selects an item (action)
    
    Reward: +1 if Listener selects the correct item, 0 otherwise (shared reward)
    """

    def __init__(self, vocab_size=10, num_candidates=4, message_len=2):
        self.vocab_size = vocab_size        # number of possible message tokens
        self.num_candidates = num_candidates  # items Listener chooses from
        self.message_len = message_len      # fixed message length

        # Action spaces
        self.speaker_action_size = vocab_size ** message_len  # all possible messages
        self.listener_action_size = num_candidates

        # Observation sizes
        # Speaker obs: one-hot target (num_candidates) + one-hot candidates flattened
        self.speaker_obs_size = num_candidates + num_candidates
        # Listener obs: message (message_len * vocab_size one-hot) + candidates
        self.listener_obs_size = message_len * vocab_size + num_candidates

    def reset(self):
        """Reset environment, sample new target and candidate set."""
        self.candidates = np.eye(self.num_candidates)  # identity = distinct items
        self.target_idx = np.random.randint(self.num_candidates)
        self.target = self.candidates[self.target_idx]
        self.last_message = None
        return self._speaker_obs(), self._listener_obs_no_message()

    def _speaker_obs(self):
        """Speaker sees: target item (one-hot) + candidate count indicator."""
        target_onehot = np.zeros(self.num_candidates)
        target_onehot[self.target_idx] = 1.0
        return np.concatenate([target_onehot, np.ones(self.num_candidates)])

    def _listener_obs(self, message_onehot):
        """Listener sees: message + all candidates."""
        return np.concatenate([message_onehot, np.ones(self.num_candidates)])

    def _listener_obs_no_message(self):
        return np.zeros(self.listener_obs_size)

    def message_to_onehot(self, action_idx):
        """Convert flat message action index to one-hot encoded tokens."""
        tokens = []
        idx = action_idx
        for _ in range(self.message_len):
            token = idx % self.vocab_size
            one_hot = np.zeros(self.vocab_size)
            one_hot[token] = 1.0
            tokens.append(one_hot)
            idx //= self.vocab_size
        return np.concatenate(tokens)

    def step(self, speaker_action, listener_action):
        """
        Execute one full dialogue turn.
        Returns: listener_obs, reward, done, info
        """
        message_onehot = self.message_to_onehot(speaker_action)
        self.last_message = message_onehot

        listener_obs = self._listener_obs(message_onehot)
        reward = 1.0 if listener_action == self.target_idx else 0.0
        done = True  # single-turn episode
        info = {"target_idx": self.target_idx, "listener_choice": listener_action}
        return listener_obs, reward, done, info
