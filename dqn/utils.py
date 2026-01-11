"""
Utility functions for DQN training.

Includes preprocessing, frame stacking, plotting, and reward shaping.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os


def shape_reward(reward, done, info=None):
    """
    Pass through reward WITHOUT clipping!
    
    Original Flappy Bird rewards:
    - +1.0 per pipe passed
    - 0.0 for staying alive
    - -1.0 for death
    
    We preserve the full reward signal so the agent can learn:
    - 1 pipe = +1.0 reward
    - 2 pipes = +2.0 reward (cumulative!)
    - 3 pipes = +3.0 reward (even better!)
    
    This is CRITICAL for learning to pass multiple pipes!
    """
    # NO CLIPPING - return raw reward
    return float(reward)


class FramePreprocessor:
    """
    Preprocesses frames from the environment for neural network input.
    
    Typical preprocessing pipeline:
    1. Convert to grayscale (reduce channels from 3 to 1)
    2. Resize to smaller resolution (84x84 for efficiency)
    3. Normalize pixel values to [0, 1]
    4. Stack multiple frames for temporal information
    """
    
    def __init__(self, width=84, height=84, grayscale=True, normalize=True):
        """
        Args:
            width: int - target width
            height: int - target height
            grayscale: bool - convert to grayscale
            normalize: bool - normalize to [0, 1]
        """
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.normalize = normalize
    
    def process(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: np.ndarray - raw frame from environment (H, W, C)
            
        Returns:
            np.ndarray - processed frame (H, W) or (H, W, 1)
        """
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize
        frame = cv2.resize(frame, (self.width, self.height), 
                          interpolation=cv2.INTER_AREA)
        
        # Normalize
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame


class FrameStack:
    """
    Stacks multiple frames to capture temporal information.
    
    Useful for understanding velocity and direction of movement.
    """
    
    def __init__(self, num_frames=4):
        """
        Args:
            num_frames: int - number of frames to stack
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        """Reset with initial frame (repeat it num_frames times)."""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def push(self, frame):
        """Add new frame and return stacked state."""
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        """Return current stacked state as numpy array."""
        # Stack along first dimension: (num_frames, H, W)
        return np.stack(self.frames, axis=0)


class AdaptiveEpsilonScheduler:
    """
    ADAPTIVE epsilon decay with CURRICULUM LEARNING!
    
    Strategy:
    - Fast decay when performance improves (agent learning well)
    - Slow/pause decay when stuck (needs more exploration)
    - Boost epsilon if performance degrades (escape local minima)
    - **NEW**: Boost epsilon when milestone achieved (curriculum learning)
    
    Curriculum Learning:
    - Detects when agent masters a difficulty (e.g., 70% success at 1 pipe)
    - Automatically boosts epsilon to explore next difficulty level
    - Progressive: 1 pipe → 2 pipes → 3 pipes → 5 pipes → etc.
    """
    
    def __init__(self, start=1.0, end=0.001, decay_steps=500000, 
                 performance_window=100, boost_threshold=-0.1,
                 enable_curriculum=True):
        """
        Args:
            start: float - initial epsilon
            end: float - final epsilon
            decay_steps: int - base steps to decay (can be extended dynamically)
            performance_window: int - episodes to track for performance
            boost_threshold: float - if avg reward drops this much, boost epsilon
            enable_curriculum: bool - enable milestone-based curriculum learning
        """
        self.start = start
        self.end = end
        self.base_decay_steps = decay_steps
        self.current_step = 0
        
        # Adaptive parameters
        self.performance_window = performance_window
        self.boost_threshold = boost_threshold
        self.recent_rewards = []
        self.best_avg_reward = -float('inf')
        self.plateau_counter = 0
        self.decay_speed = 1.0  # Multiplier for decay speed (1.0 = normal)
        
        # Curriculum Learning - Milestone Tracking
        self.enable_curriculum = enable_curriculum
        self.milestones = [1, 2, 3, 5, 10, 15, 20, 30]  # Pipe count targets
        self.current_milestone_idx = 0
        self.milestone_window = 50  # Episodes to check for mastery
        self.milestone_threshold = 0.7  # 70% success rate to advance
        
    def get_epsilon(self):
        """Get current epsilon with adaptive decay."""
        # Effective decay steps (adjusted by decay_speed)
        effective_steps = self.base_decay_steps / self.decay_speed
        
        if self.current_step >= effective_steps:
            return self.end
        
        # Linear decay with adaptive speed
        progress = self.current_step / effective_steps
        epsilon = self.start - (self.start - self.end) * progress
        
        return max(self.end, epsilon)
    
    def update_performance(self, episode_reward):
        """
        Update performance tracking and adjust decay speed.
        
        NOW WITH CURRICULUM LEARNING!
        
        Args:
            episode_reward: float - reward from last episode
        """
        self.recent_rewards.append(episode_reward)
        
        # Keep only recent window
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)
        
        # Need enough data
        if len(self.recent_rewards) < self.performance_window // 2:
            return
        
        avg_reward = np.mean(self.recent_rewards)
        
        # CURRICULUM LEARNING: Check milestone progress
        if self.enable_curriculum and self.current_milestone_idx < len(self.milestones):
            self._check_milestone_progress(avg_reward)
        
        # Check for improvement
        if avg_reward > self.best_avg_reward + 0.5:  # Significant improvement
            self.best_avg_reward = avg_reward
            self.plateau_counter = 0
            # Speed up decay when learning well
            self.decay_speed = min(1.5, self.decay_speed * 1.05)
            
        # NOTE: Removed aggressive performance-drop boost
        # It was too aggressive (epsilon -> 1.0) and harmful to learning
        # Curriculum milestone boosts are sufficient for exploration
        
        # Track plateaus but don't boost aggressively
        elif avg_reward < self.best_avg_reward - abs(self.boost_threshold):
            self.plateau_counter += 1
            if self.plateau_counter > 10:
                # Just slow down decay, don't boost epsilon
                self.decay_speed = max(0.7, self.decay_speed * 0.98)
                self.plateau_counter = 0  # Reset
        else:
            # Mild plateau detected
            self.plateau_counter += 1
            if self.plateau_counter > 15:
                # Slow down decay during plateau
                self.decay_speed = max(0.7, self.decay_speed * 0.98)
    
    def _check_milestone_progress(self, avg_reward):
        """
        Check if current milestone has been mastered.
        If yes, boost epsilon and advance to next milestone.
        """
        current_milestone = self.milestones[self.current_milestone_idx]
        
        # Need enough recent data
        if len(self.recent_rewards) < self.milestone_window:
            return
        
        # Count successes in recent window
        recent_window = self.recent_rewards[-self.milestone_window:]
        successes = sum(1 for r in recent_window if r >= current_milestone)
        success_rate = successes / self.milestone_window
        
        # Milestone achieved!
        if success_rate >= self.milestone_threshold:
            next_milestone_idx = self.current_milestone_idx + 1
            
            if next_milestone_idx < len(self.milestones):
                next_milestone = self.milestones[next_milestone_idx]
                print(f"\n{'='*80}")
                print(f"MILESTONE ACHIEVED! Mastered {current_milestone} pipe(s)")
                print(f"   Success rate: {success_rate*100:.1f}% over last {self.milestone_window} episodes")
                print(f"   Next target: {next_milestone} pipe(s)")
                
                # Boost epsilon for exploration of next difficulty level
                self._boost_epsilon_for_next_level()
                
                print(f"   Epsilon boosted to: {self.get_epsilon():.3f}")
                print(f"{'='*80}\n")
                
                self.current_milestone_idx = next_milestone_idx
                self.plateau_counter = 0  # Reset plateau counter
    
    def _boost_epsilon_for_next_level(self):
        """
        Boost epsilon to explore next difficulty level.
        Steps back in the decay to increase epsilon.
        
        CONSERVATIVE boost to avoid too much random exploration!
        """
        # SMALLER boost: only 15% step back (was 30%)
        # This gives enough exploration without wasting too much time on random actions
        boost_steps = int(self.base_decay_steps * 0.15)
        self.current_step = max(0, self.current_step - boost_steps)
        
        # Slightly faster decay after boost (was 0.8, now 0.9)
        # Returns to exploitation faster after brief exploration
        self.decay_speed = max(0.7, self.decay_speed * 0.9)
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def get_stats(self):
        """Get debugging stats."""
        stats = {
            'epsilon': self.get_epsilon(),
            'decay_speed': self.decay_speed,
            'plateau_counter': self.plateau_counter,
            'avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'best_avg': self.best_avg_reward
        }
        
        # Add curriculum stats if enabled
        if self.enable_curriculum and self.current_milestone_idx < len(self.milestones):
            stats['current_milestone'] = self.milestones[self.current_milestone_idx]
            if len(self.recent_rewards) >= self.milestone_window:
                recent_window = self.recent_rewards[-self.milestone_window:]
                successes = sum(1 for r in recent_window 
                              if r >= self.milestones[self.current_milestone_idx])
                stats['milestone_progress'] = f"{successes}/{self.milestone_window}"
        
        return stats


# Keep old scheduler for compatibility
class EpsilonScheduler:
    """
    Basic epsilon decay (linear/exponential).
    """
    
    def __init__(self, start=1.0, end=0.01, decay_steps=100000, decay_type='linear'):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0
    
    def get_epsilon(self):
        """Get current epsilon value."""
        if self.current_step >= self.decay_steps:
            return self.end
        
        if self.decay_type == 'linear':
            epsilon = self.start - (self.start - self.end) * (self.current_step / self.decay_steps)
        else:  # exponential
            epsilon = self.end + (self.start - self.end) * np.exp(-self.current_step / self.decay_steps)
        
        return epsilon
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1


def plot_training_curves(rewards, losses, epsilons, save_path=None):
    """
    Plot training curves: rewards, losses, and epsilon over time.
    
    Args:
        rewards: list - episode rewards
        losses: list - training losses
        epsilons: list - epsilon values
        save_path: str - path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) > 50:
        # Moving average
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                    color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Losses
    axes[1].plot(losses, alpha=0.6, color='green')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Epsilon
    axes[2].plot(epsilons, color='purple')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Exploration Rate (Epsilon)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_checkpoint(agent, episode, rewards, losses, path):
    """
    Save training checkpoint.
    
    Args:
        agent: DQNAgent - agent to save
        episode: int - current episode
        rewards: list - episode rewards
        losses: list - training losses
        path: str - save path
    """
    checkpoint = {
        'episode': episode,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards': rewards,
        'losses': losses,
        'epsilon': agent.epsilon_scheduler.get_epsilon()
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(agent, path):
    """
    Load training checkpoint.
    
    Args:
        agent: DQNAgent - agent to load into
        path: str - checkpoint path
        
    Returns:
        dict - checkpoint data (episode, rewards, losses)
    """
    checkpoint = torch.load(path)
    
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from: {path}")
    
    return {
        'episode': checkpoint['episode'],
        'rewards': checkpoint['rewards'],
        'losses': checkpoint['losses'],
        'epsilon': checkpoint['epsilon']
    }


def calculate_statistics(rewards):
    """
    Calculate statistics from list of rewards.
    
    Returns:
        dict - mean, std, min, max, median
    """
    rewards = np.array(rewards)
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards)
    }


def print_training_stats(episode, total_episodes, reward, epsilon, loss, 
                        avg_reward_100, steps, total_steps, episode_rewards=None, 
                        q_values=None, action_counts=None):
    """
    Print comprehensive training statistics with debugging info.
    
    Args:
        episode: current episode number
        total_episodes: total episodes to train
        reward: current episode reward
        epsilon: current epsilon value
        loss: current loss
        avg_reward_100: average reward over last 100 episodes
        steps: steps in current episode
        total_steps: total steps so far
        episode_rewards: list of all episode rewards (optional, for best/worst)
        q_values: recent Q-values (optional, for debugging)
        action_counts: dict with action distribution (optional)
    """
    print(f"━" * 100)
    print(f"Episode {episode}/{total_episodes} ({100*episode/total_episodes:.1f}%)")
    
    # Main metrics
    print(f"   Current Reward: {reward:.2f} | Steps: {steps}")
    print(f"   Avg(100): {avg_reward_100:.2f}", end="")
    
    # Best and worst performance
    if episode_rewards and len(episode_rewards) >= 100:
        recent_100 = episode_rewards[-100:]
        best_100 = max(recent_100)
        worst_100 = min(recent_100)
        print(f" | Best: {best_100:.2f} | Worst: {worst_100:.2f}")
        
        # Overall best
        overall_best = max(episode_rewards)
        print(f"   🏆 All-time Best: {overall_best:.2f} (Episode {episode_rewards.index(overall_best) + 1})")
    else:
        print()
    
    # Recent trend (last 10 episodes)
    if episode_rewards and len(episode_rewards) >= 10:
        recent_10 = episode_rewards[-10:]
        trend = "UP" if recent_10[-1] > recent_10[0] else "DOWN"
        print(f"   Last 10: {trend} [{', '.join([f'{r:.1f}' for r in recent_10[-5:]])}...]")
    
    # Learning metrics
    print(f"   Epsilon: {epsilon:.4f} | Loss: {loss:.4f} | Total Steps: {total_steps:,}")
    
    # Q-values debugging (if available)
    if q_values is not None:
        print(f"   Q-values: mean={q_values['mean']:.3f}, max={q_values['max']:.3f}, min={q_values['min']:.3f}")
    
    # Action distribution (if available)
    if action_counts is not None:
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_dist = {k: f"{100*v/total_actions:.1f}%" for k, v in action_counts.items()}
            print(f"   Actions: {action_dist}")
    
    print(f"━" * 100)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing FramePreprocessor...")
    preprocessor = FramePreprocessor(84, 84)
    
    # Fake frame
    frame = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
    processed = preprocessor.process(frame)
    print(f"Original shape: {frame.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Test frame stacking
    print("\nTesting FrameStack...")
    stack = FrameStack(4)
    state = stack.reset(processed)
    print(f"Stacked state shape: {state.shape}")
    
    # Test epsilon scheduler
    print("\nTesting EpsilonScheduler...")
    scheduler = EpsilonScheduler(1.0, 0.01, 1000)
    epsilons = [scheduler.get_epsilon() for _ in range(1000)]
    for _ in range(1000):
        scheduler.step()
    print(f"Initial epsilon: {epsilons[0]:.4f}")
    print(f"Final epsilon: {epsilons[-1]:.4f}")
