"""
DQN Agent Implementation

Implements Deep Q-Network with:
- Double DQN (reduces overestimation bias)
- Dueling DQN (separate value and advantage streams)
- Prioritized Experience Replay
- Target network with periodic updates
- Epsilon-greedy exploration

Based on:
- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
- "Prioritized Experience Replay" (Schaul et al., 2016)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from .network import DuelingDQN
from .replay_buffer import PrioritizedReplayBuffer, SimpleReplayBuffer
from .utils import FramePreprocessor, FrameStack, AdaptiveEpsilonScheduler, EpsilonScheduler
from . import config  # Import config for USE_BATCH_NORM


class DQNAgent:
    """
    Complete DQN Agent with all modern improvements.
    """
    
    def __init__(self, 
                 input_shape,
                 n_actions,
                 device='cuda',
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay_steps=100000,
                 buffer_size=100000,
                 batch_size=64,
                 target_update_freq=1000,
                 use_double_dqn=True,
                 use_dueling_dqn=True,
                 use_per=True,
                 per_alpha=0.6,
                 per_beta_start=0.4,
                 per_beta_frames=100000,
                 grad_clip=10.0,
                 dropout_rate=0.0):  # Anti-plateau regularization
        """
        Initialize DQN Agent.
        
        Args:
            input_shape: tuple - (channels, height, width)
            n_actions: int - number of possible actions
            device: torch.device - cuda or cpu
            learning_rate: float - optimizer learning rate
            gamma: float - discount factor
            epsilon_start/min/decay_steps: epsilon-greedy parameters
            buffer_size: int - replay buffer capacity
            batch_size: int - training batch size
            target_update_freq: int - steps between target network updates
            use_double_dqn: bool - use Double DQN
            use_dueling_dqn: bool - use Dueling architecture
            use_per: bool - use Prioritized Experience Replay
            per_alpha/beta_start/beta_frames: PER parameters
            grad_clip: float - gradient clipping value
            dropout_rate: float - dropout probability for regularization
        """
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.grad_clip = grad_clip
        
        # Networks - use config for batch normalization setting
        self.q_network = DuelingDQN(input_shape, n_actions, use_dueling_dqn, 
                                    use_batch_norm=config.USE_BATCH_NORM, dropout_rate=dropout_rate).to(device)
        self.target_network = DuelingDQN(input_shape, n_actions, use_dueling_dqn,
                                         use_batch_norm=config.USE_BATCH_NORM, dropout_rate=dropout_rate).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Learning Rate Scheduler - SIMPLIFIED for stability
        # ReduceLROnPlateau adapts based on performance, not time
        # This is more stable than aggressive cosine annealing
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',          # Monitor reward (higher is better)
            factor=0.5,          # Reduce LR by half when plateauing
            patience=1000,       # Wait 1000 updates before reducing
            min_lr=learning_rate * 0.01  # Minimum LR (1% of initial)
        )
        
        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, per_alpha, per_beta_start, per_beta_frames
            )
        else:
            self.replay_buffer = SimpleReplayBuffer(buffer_size)
        
        # Exploration - ADAPTIVE with CURRICULUM LEARNING!
        # Automatically boosts epsilon when milestones achieved
        self.epsilon_scheduler = AdaptiveEpsilonScheduler(
            start=epsilon_start, 
            end=epsilon_min, 
            decay_steps=epsilon_decay_steps,
            enable_curriculum=True  # Enable milestone-based exploration boosts
        )
        
        # Training statistics
        self.steps_done = 0
        self.updates_done = 0
        
        print(f"🤖 DQN Agent initialized")
        print(f"   Network: {'Dueling ' if use_dueling_dqn else ''}{'Double ' if use_double_dqn else ''}DQN")
        print(f"   Replay: {'Prioritized' if use_per else 'Uniform'}")
        print(f"   Batch Norm: {config.USE_BATCH_NORM}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def select_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: np.ndarray - current state (C, H, W)
            epsilon: float - override epsilon (if None, use scheduler)
            
        Returns:
            int - selected action
        """
        if epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        
        if np.random.random() < epsilon:
            # Explore
            return np.random.randint(self.n_actions)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        
        Args:
            state, next_state: np.ndarray
            action: int
            reward: float
            done: bool
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """
        Perform one update step on the Q-network.
        
        Returns:
            dict - training metrics (loss, q_values, grad_norm) or None if not enough samples
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use Q-network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (weighted by importance sampling weights)
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping and tracking
        grad_norm = 0.0
        if self.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # LR Scheduler: ReduceLROnPlateau is called from training loop with reward metric
        # NOT called here (it needs a performance metric, not just time steps)
        # self.scheduler.step()  # Will be called manually with reward
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # HARD target network update (Nature DQN 2015)
        # Update every TARGET_UPDATE_FREQUENCY steps (NOT every step!)
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            # print(f"Target network updated @ step {self.steps_done}")
        
        # Update epsilon
        self.epsilon_scheduler.step()
        
        self.steps_done += 1
        self.updates_done += 1
        
        # Return debugging metrics
        return {
            'loss': loss.item(),
            'q_values': {
                'mean': current_q_values.mean().item(),
                'max': current_q_values.max().item(),
                'min': current_q_values.min().item()
            },
            'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item()
        }
    
    def save(self, path):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'updates_done': self.updates_done
        }, path)
    
    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.updates_done = checkpoint.get('updates_done', 0)
        print(f"Agent loaded from {path}")
    
    def train_mode(self):
        """Set networks to training mode."""
        self.q_network.train()
        # Target network always stays in eval mode
    
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.q_network.eval()


class DQNTrainer:
    """
    Trainer class to handle the training loop.
    """
    
    def __init__(self, agent, env, preprocessor, frame_stack, frame_skip=1):
        """
        Args:
            agent: DQNAgent
            env: gymnasium environment
            preprocessor: FramePreprocessor
            frame_stack: FrameStack
            frame_skip: int - number of frames to skip (repeat action)
        """
        self.agent = agent
        self.env = env
        self.preprocessor = preprocessor
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def train_episode(self, min_buffer_size=10000, update_frequency=4):
        """
        Train for one episode.
        
        Args:
            min_buffer_size: Minimum buffer size before training starts
            update_frequency: Update every N steps (batched updates)
        
        Returns:
            dict - episode statistics including debugging info
        """
        # Reset environment
        obs, _ = self.env.reset()
        frame = self.preprocessor.process(obs)
        state = self.frame_stack.reset(frame)
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_q_values = []
        episode_grad_norms = []
        action_counts = {0: 0, 1: 0}  # Track action distribution
        
        done = False
        step_count = 0
        
        while not done:
            # Select action
            action = self.agent.select_action(state)
            action_counts[action] += 1
            
            # Execute action with frame skipping for efficiency
            total_reward = 0
            for _ in range(self.frame_skip):
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                if done:
                    break
            
            # Apply reward shaping for better learning signal
            from .utils import shape_reward
            shaped_reward = shape_reward(
                total_reward, 
                done, 
                info
                # Use default params from utils.py
            )
            
            # NO reward clipping - let shaped rewards flow! (bug fix)
            
            # Preprocess next state
            next_frame = self.preprocessor.process(next_obs)
            next_state = self.frame_stack.push(next_frame)
            
            # Store transition (with shaped reward - NO clipping!)
            self.agent.store_transition(state, action, shaped_reward, next_state, done)
            
            # Update agent (only if buffer has enough samples AND at update frequency)
            step_count += 1
            if len(self.agent.replay_buffer) >= min_buffer_size and step_count % update_frequency == 0:
                metrics = self.agent.update()
                if metrics is not None:
                    episode_losses.append(metrics['loss'])
                    episode_q_values.append(metrics['q_values'])
                    episode_grad_norms.append(metrics['grad_norm'])
            
            # Update statistics (use original reward for tracking)
            episode_reward += total_reward
            episode_length += 1
            state = next_state
        
        # Record statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            self.losses.append(avg_loss)
        else:
            avg_loss = 0.0
        
        # Aggregate Q-values
        avg_q_values = None
        if episode_q_values:
            avg_q_values = {
                'mean': np.mean([q['mean'] for q in episode_q_values]),
                'max': np.mean([q['max'] for q in episode_q_values]),
                'min': np.mean([q['min'] for q in episode_q_values])
            }
        
        
        # Update adaptive epsilon scheduler with episode performance
        if hasattr(self.agent.epsilon_scheduler, 'update_performance'):
            self.agent.epsilon_scheduler.update_performance(episode_reward)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'loss': avg_loss,
            'epsilon': self.agent.epsilon_scheduler.get_epsilon(),
            'q_values': avg_q_values,
            'action_counts': action_counts,
            'grad_norm': np.mean(episode_grad_norms) if episode_grad_norms else 0.0
        }
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate agent for n_episodes without exploration.
        
        Returns:
            dict - evaluation statistics
        """
        self.agent.eval_mode()
        eval_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            frame = self.preprocessor.process(obs)
            state = self.frame_stack.reset(frame)
            
            episode_reward = 0
            done = False
            
            while not done:
                # Greedy action (epsilon=0)
                action = self.agent.select_action(state, epsilon=0.0)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                frame = self.preprocessor.process(obs)
                state = self.frame_stack.push(frame)
                
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        self.agent.train_mode()
        
        return {
            'mean': np.mean(eval_rewards),
            'std': np.std(eval_rewards),
            'min': np.min(eval_rewards),
            'max': np.max(eval_rewards)
        }


if __name__ == "__main__":
    # Test agent initialization
    print("Testing DQN Agent...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (4, 84, 84)
    n_actions = 2
    
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        use_double_dqn=True,
        use_dueling_dqn=True,
        use_per=True
    )
    
    print("\nAgent created successfully!")
    print(f"   Steps done: {agent.steps_done}")
    print(f"   Epsilon: {agent.epsilon_scheduler.get_epsilon():.4f}")
