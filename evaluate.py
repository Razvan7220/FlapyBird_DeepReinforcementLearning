"""
Evaluation script for trained DQN agent

Run trained agent and collect statistics over multiple episodes.
"""

import gymnasium
import flappy_bird_gymnasium
import torch
import numpy as np
import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn.dqn_agent import DQNAgent
from dqn.utils import FramePreprocessor, FrameStack, calculate_statistics
from dqn import config


def evaluate_agent(agent, env, preprocessor, frame_stack, n_episodes=50, render=True, verbose=True):
    """
    Evaluate agent over multiple episodes.
    
    Args:
        agent: DQNAgent - trained agent
        env: gymnasium environment
        preprocessor: FramePreprocessor
        frame_stack: FrameStack
        n_episodes: int - number of evaluation episodes
        render: bool - whether to render
        verbose: bool - print progress
        
    Returns:
        dict - evaluation statistics
    """
    agent.eval_mode()
    
    rewards = []
    lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        frame = preprocessor.process(obs)
        state = frame_stack.reset(frame)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select greedy action (no exploration)
            action = agent.select_action(state, epsilon=0.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = preprocessor.process(obs)
            state = frame_stack.push(frame)
            
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    agent.train_mode()
    
    # Calculate statistics
    stats = {
        'rewards': calculate_statistics(rewards),
        'lengths': calculate_statistics(lengths),
        'raw_rewards': rewards,
        'raw_lengths': lengths
    }
    
    return stats


def main(args):
    print("=" * 80)
    print("DQN FLAPPY BIRD EVALUATION")
    print("=" * 80)
    
    # Create environment
    env = gymnasium.make(
        config.ENV_NAME,
        render_mode="human" if args.render else None,
        use_lidar=config.USE_LIDAR
    )
    
    n_actions = env.action_space.n
    
    # Initialize preprocessing
    preprocessor = FramePreprocessor(
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        grayscale=config.GRAYSCALE,
        normalize=config.NORMALIZE
    )
    
    frame_stack = FrameStack(num_frames=config.FRAME_STACK)
    input_shape = (config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH)
    
    # Create agent
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=config.DEVICE,
        use_double_dqn=config.USE_DOUBLE_DQN,
        use_dueling_dqn=config.USE_DUELING_DQN,
        use_per=config.USE_PER
    )
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Please train a model first using train_dqn.py")
        return
    
    print(f"\n📂 Loading model from: {args.model}")
    agent.load(args.model)
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    print("=" * 80 + "\n")
    
    stats = evaluate_agent(
        agent, env, preprocessor, frame_stack,
        n_episodes=args.episodes,
        render=args.render,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nRewards:")
    print(f"   Mean:   {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
    print(f"   Median: {stats['rewards']['median']:.2f}")
    print(f"   Min:    {stats['rewards']['min']:.2f}")
    print(f"   Max:    {stats['rewards']['max']:.2f}")
    
    print(f"\n📏 Episode Lengths:")
    print(f"   Mean:   {stats['lengths']['mean']:.1f} ± {stats['lengths']['std']:.1f}")
    print(f"   Median: {stats['lengths']['median']:.1f}")
    print(f"   Min:    {stats['lengths']['min']:.0f}")
    print(f"   Max:    {stats['lengths']['max']:.0f}")
    
    # Distribution of rewards
    print(f"\nReward Distribution:")
    bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-10', '10-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    
    reward_array = np.array(stats['raw_rewards'])
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = np.sum((reward_array >= low) & (reward_array < high))
        pct = 100 * count / len(reward_array)
        print(f"   {labels[i]:>10}: {count:3d} episodes ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    
    # Save results
    if args.save_results:
        results_file = args.save_results
        np.savez(
            results_file,
            rewards=stats['raw_rewards'],
            lengths=stats['raw_lengths']
        )
        print(f"Results saved to: {results_file}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', type=str, default='results/checkpoints/dqn_best.pth',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save evaluation results (.npz)')
    
    args = parser.parse_args()
    
    main(args)
