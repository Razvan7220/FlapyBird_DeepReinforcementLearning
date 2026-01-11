"""
Prioritized Experience Replay Buffer

Implements Prioritized Experience Replay from "Prioritized Experience Replay" (Schaul et al., 2016)
Uses a sum tree data structure for efficient sampling based on TD-error priorities.
"""

import numpy as np
import random
from collections import namedtuple
import torch


# Transition tuple
Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    The tree structure allows O(log n) updates and O(log n) sampling,
    compared to O(n) for naive implementation.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Retrieve sample index based on priority value s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new data with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority of node at idx."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get data and index for priority value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD-error, giving more weight to
    surprising transitions that the agent can learn more from.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: int - maximum size of buffer
            alpha: float - how much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: float - initial importance sampling weight
            beta_frames: int - number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        self.max_priority = 1.0
    
    def _get_beta(self):
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, *args):
        """Add transition to buffer with maximum priority."""
        transition = Transition(*args)
        self.tree.add(self.max_priority ** self.alpha, transition)
    
    def sample(self, batch_size):
        """
        Sample batch of transitions based on priorities.
        
        Returns:
            batch: list of Transitions
            indices: list of tree indices (for updating priorities)
            weights: importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        
        # Divide priority range into segments
        segment = self.tree.total() / batch_size
        
        beta = self._get_beta()
        self.frame += 1
        
        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        min_prob = np.min(priorities) / self.tree.total()
        max_weight = (min_prob * len(self)) ** (-beta)
        
        weights = (priorities / self.tree.total() * len(self)) ** (-beta)
        weights /= max_weight
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: list of tree indices
            td_errors: TD-errors (priorities will be |td_error| + epsilon)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries


class SimpleReplayBuffer:
    """
    Simple uniform replay buffer for comparison.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, *args):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch uniformly."""
        batch = random.sample(self.buffer, batch_size)
        indices = [0] * batch_size  # Dummy indices
        weights = np.ones(batch_size)  # Uniform weights
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """No-op for uniform buffer."""
        pass
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Test Prioritized Replay Buffer
    print("Testing PrioritizedReplayBuffer...")
    
    buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(50):
        state = np.random.randn(4, 84, 84)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4, 84, 84)
        done = bool(np.random.randint(2))
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample batch
    batch, indices, weights = buffer.sample(8)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Indices: {indices}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights: {weights}")
    
    # Update priorities
    td_errors = np.random.randn(8)
    buffer.update_priorities(indices, td_errors)
    print("Priorities updated successfully!")
    
    # Sample again to see if priorities changed
    batch2, indices2, weights2 = buffer.sample(8)
    print(f"New weights: {weights2}")
