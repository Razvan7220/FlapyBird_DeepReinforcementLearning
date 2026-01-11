"""
Training script for DQN Flappy Bird Agent

This script trains a DQN agent from scratch using pixel input.
Implements Double DQN, Dueling architecture, and Prioritized Experience Replay.
"""

import gymnasium
import flappy_bird_gymnasium
import torch
import numpy as np
import os
import sys
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.dqn_agent import DQNAgent, DQNTrainer
from dqn.utils import (
    FramePreprocessor, FrameStack, plot_training_curves, 
    save_checkpoint, print_training_stats
)
from dqn import config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main(args):
    # Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    print("=" * 80)
    print("DQN FLAPPY BIRD TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.DEVICE}")
    print(f"Training for {args.episodes} episodes")
    print("=" * 80)
    
    # Create environment
    env = gymnasium.make(
        config.ENV_NAME,
        render_mode=None if not config.RENDER_TRAINING else "human",
        use_lidar=config.USE_LIDAR
    )
    
    # Get environment info
    n_actions = env.action_space.n
    print(f"\n📋 Environment Info:")
    print(f"   Actions: {n_actions} (0=do nothing, 1=flap)")
    print(f"   Observation space: {env.observation_space}")
    
    # Initialize preprocessing
    preprocessor = FramePreprocessor(
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        grayscale=config.GRAYSCALE,
        normalize=config.NORMALIZE
    )
    
    frame_stack = FrameStack(num_frames=config.FRAME_STACK)
    
    # Input shape for network
    input_shape = (config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH)
    
    # Create agent with anti-plateau configuration
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay_steps=config.EPSILON_DECAY_STEPS,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQUENCY,
        use_double_dqn=config.USE_DOUBLE_DQN,
        use_dueling_dqn=config.USE_DUELING_DQN,
        use_per=config.USE_PER,
        per_alpha=config.PER_ALPHA,
        per_beta_start=config.PER_BETA_START,
        per_beta_frames=config.PER_BETA_FRAMES,
        grad_clip=config.GRAD_CLIP,
        dropout_rate=config.DROPOUT_RATE  # Anti-plateau regularization
    )
    
    # Create trainer
    trainer = DQNTrainer(agent, env, preprocessor, frame_stack, frame_skip=config.FRAME_SKIP)
    
    # Create results directories
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Load checkpoint if resuming
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
        # Try to load training stats
        # (In a full implementation, these would be saved in checkpoint)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    # Training loop
    best_avg_reward = -float('inf')
    recent_rewards = []
    
    try:
        for episode in range(start_episode, args.episodes):
            # Train one episode with optimizations
            stats = trainer.train_episode(
                min_buffer_size=config.MIN_BUFFER_SIZE,
                update_frequency=config.UPDATE_FREQUENCY
            )
            
            # Track recent rewards for early stopping
            recent_rewards.append(stats['reward'])
            if len(recent_rewards) > config.EARLY_STOP_EPISODES:
                recent_rewards.pop(0)
            
            avg_reward_100 = np.mean(trainer.episode_rewards[-100:]) if len(trainer.episode_rewards) >= 100 else np.mean(trainer.episode_rewards)
            
            # Print stats periodically
            if (episode + 1) % config.LOG_INTERVAL == 0:
                print_training_stats(
                    episode + 1,
                    args.episodes,
                    stats['reward'],
                    stats['epsilon'],
                    stats['loss'],
                    avg_reward_100,
                    stats['length'],
                    agent.steps_done,
                    episode_rewards=trainer.episode_rewards,
                    q_values=stats.get('q_values'),
                    action_counts=stats.get('action_counts')
                )
            
            # Evaluate periodically
            if (episode + 1) % config.EVAL_FREQUENCY == 0:
                print(f"\n🔍 Evaluating at episode {episode + 1}...")
                eval_stats = trainer.evaluate(config.EVAL_EPISODES)
                print(f"   Eval Mean: {eval_stats['mean']:.2f} ± {eval_stats['std']:.2f}")
                print(f"   Eval Range: [{eval_stats['min']:.2f}, {eval_stats['max']:.2f}]\n")
                
                # Save best model
                if eval_stats['mean'] > best_avg_reward:
                    best_avg_reward = eval_stats['mean']
                    best_path = "results/checkpoints/dqn_best.pth"
                    agent.save(best_path)
                    print(f"   💎 New best model saved! Avg reward: {best_avg_reward:.2f}\n")
            
            # Save checkpoint periodically
            if (episode + 1) % config.SAVE_FREQUENCY == 0:
                checkpoint_path = f"results/checkpoints/dqn_episode_{episode + 1}.pth"
                save_checkpoint(
                    agent, episode + 1,
                    trainer.episode_rewards,
                    trainer.losses,
                    checkpoint_path
                )
                
                # Plot training curves
                plot_path = f"results/plots/training_curves_{episode + 1}.png"
                plot_training_curves(
                    trainer.episode_rewards,
                    trainer.losses,
                    [agent.epsilon_scheduler.get_epsilon()] * len(trainer.losses),
                    save_path=plot_path
                )
            
            # Early stopping check
            if len(recent_rewards) >= config.EARLY_STOP_EPISODES:
                avg_recent = np.mean(recent_rewards)
                if avg_recent >= config.EARLY_STOP_REWARD:
                    print(f"\n🎉 Early stopping! Average reward {avg_recent:.2f} >= {config.EARLY_STOP_REWARD}")
                    break
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    finally:
        print("\n" + "=" * 80)
        print("SAVING FINAL MODEL")
        print("=" * 80)
        
        # Save final model
        final_path = "results/checkpoints/dqn_final.pth"
        agent.save(final_path)
        print(f"Final model saved to: {final_path}")
        
        # Save final training curves
        plot_path = "results/plots/training_curves_final.png"
        plot_training_curves(
            trainer.episode_rewards,
            trainer.losses,
            [agent.epsilon_scheduler.get_epsilon()] * len(trainer.losses),
            save_path=plot_path
        )
        
        # Final evaluation
        print(f"\n🏁 FINAL EVALUATION ({config.EVAL_EPISODES} episodes):")
        final_eval = trainer.evaluate(config.EVAL_EPISODES)
        print(f"   Mean: {final_eval['mean']:.2f} ± {final_eval['std']:.2f}")
        print(f"   Range: [{final_eval['min']:.2f}, {final_eval['max']:.2f}]")
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Episodes completed: {len(trainer.episode_rewards)}")
        print(f"Total steps: {agent.steps_done:,}")
        print(f"Total updates: {agent.updates_done:,}")
        print(f"Best avg reward (100 ep): {best_avg_reward:.2f}")
        print(f"Final epsilon: {agent.epsilon_scheduler.get_epsilon():.4f}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for Flappy Bird')
    parser.add_argument('--episodes', type=int, default=config.MAX_EPISODES,
                       help='Number of training episodes')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-mode', action='store_true',
                       help='Quick test mode (10 episodes)')
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.episodes = 10
        print("TEST MODE: Running only 10 episodes")
    
    main(args)
