"""
Configuration file for DQN Flappy Bird Training

Contains all hyperparameters and settings for optimal performance.
"""

import torch

# ==================== ENVIRONMENT ====================
ENV_NAME = "FlappyBird-v0"
USE_LIDAR = False  # We use pixels for 30 points
RENDER_TRAINING = False
RENDER_EVALUATION = True

# ==================== NETWORK ARCHITECTURE ====================
# Input preprocessing
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_STACK = 4  # Stack 4 frames for temporal information
FRAME_SKIP = 2   # Execute action every N frames for speed

# Convolutional layers (similar to DQN Nature paper)
CONV_LAYERS = [
    {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
    {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
    {'out_channels': 64, 'kernel_size': 3, 'stride': 1}
]

# Fully connected layers
FC_HIDDEN_SIZE = 512

# ==================== DQN ALGORITHM ====================
# Learning - NATURE DQN (2015) PROVEN CONFIG
LEARNING_RATE = 0.00025  # Exact Nature paper value
GAMMA = 0.99             # Standard - works for ALL tasks
BATCH_SIZE = 32          # Nature paper standard
OPTIMIZER = 'Adam'       # Keep Adam (Nature used RMSprop)

# Exploration - BALANCED LINEAR DECAY (FIXED!)
EPSILON_START = 1.0
EPSILON_MIN = 0.01  # Minimum exploration
EPSILON_DECAY_STEPS = 200000  # Slower decay: epsilon=0.50 @ 5k episodes, 0.01 @ 20k

# Regularization - MINIMAL (baseline configuration)
DROPOUT_RATE = 0.0  # No dropout for baseline
USE_BATCH_NORM = True  # CRITICAL: Enable for stable training!

# Target network - HARD UPDATES (Nature DQN 2015)
# We use HARD updates every N steps (copy all weights at once)
# This is the original Nature DQN approach, more stable than soft updates
TARGET_UPDATE_FREQUENCY = 1000  # Update target network every 1000 steps

# Gradient clipping
GRAD_CLIP = 10.0  # Prevent exploding gradients

# ==================== EXPERIENCE REPLAY ====================
BUFFER_SIZE = 100000  # Large buffer for diverse experiences
MIN_BUFFER_SIZE = 1000  # Very fast start for quicker initial learning (was 5000)

# Prioritized Experience Replay (PER)
USE_PER = True
PER_ALPHA = 0.6  # How much prioritization to use
PER_BETA_START = 0.4  # Importance sampling weight
PER_BETA_FRAMES = 100000  # Anneal beta to 1.0 over this many frames
PER_EPSILON = 1e-6  # Small constant to avoid zero priority

# ==================== TRAINING ====================
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 10000  # Max steps per episode
UPDATE_FREQUENCY = 1  # Update every step for maximum learning speed (was 2)

# Reward processing - DISABLED for better learning signal
# Clipping can hide important differences between good/bad states
REWARD_CLIP_MIN = None  # No clipping - preserve full reward signal
REWARD_CLIP_MAX = None  # Agent needs to learn actual reward magnitudes

# Checkpointing
SAVE_FREQUENCY = 500  # Save checkpoint every N episodes
EVAL_FREQUENCY = 100  # Evaluate every N episodes
EVAL_EPISODES = 10  # Number of episodes for evaluation

# Early stopping
EARLY_STOP_REWARD = 1000  # Stop if average reward exceeds this
EARLY_STOP_EPISODES = 100  # Average over this many episodes

# ==================== ADVANCED TECHNIQUES ====================
USE_DOUBLE_DQN = True   # Reduces overestimation bias - proven effective
USE_DUELING_DQN = False # Start with baseline, can enable later
# NOTE: USE_PER is defined above at line 62 (USE_PER = True)

# Reward shaping (optional, can improve learning)
REWARD_SCALE = 1.0
REWARD_ALIVE = 0.1  # Small reward for staying alive
REWARD_PASSED_PIPE = 10.0  # Bonus for passing pipe (if we can detect it)

# ==================== DEVICE ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== LOGGING ====================
LOG_INTERVAL = 10  # Print stats every N episodes
TENSORBOARD_LOG = True
TENSORBOARD_DIR = "./results/tensorboard"

# ==================== REPRODUCIBILITY ====================
RANDOM_SEED = 42

# ==================== PERFORMANCE OPTIMIZATIONS ====================
# Frame preprocessing options
GRAYSCALE = True
NORMALIZE = True  # Scale pixels to [0, 1]
BACKGROUND_THRESHOLD = None  # Optional: threshold for background removal

# Multi-step learning (optional)
USE_NSTEP = False
NSTEP_SIZE = 3

print(f"DQN Configuration Loaded")
print(f"Device: {DEVICE}")
print(f"Architecture: {'Dueling ' if USE_DUELING_DQN else ''}{'Double ' if USE_DOUBLE_DQN else ''}DQN")
print(f"Prioritized Replay: {USE_PER}")
print(f"Frame Stack: {FRAME_STACK}, Frame Skip: {FRAME_SKIP}")
