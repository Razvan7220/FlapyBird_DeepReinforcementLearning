"""
DQN Network Architecture - OPTIMIZED VERSION

Implements Dueling DQN with modern improvements:
- Batch Normalization for training stability
- Dropout for regularization
- LeakyReLU to prevent dying neurons

Based on:
- "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Modern deep learning best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DuelingDQN(nn.Module):
    """
    OPTIMIZED Dueling DQN architecture with modern improvements.
    
    Improvements over vanilla DQN:
    1. Batch Normalization - stabilizes training, allows higher LR
    2. Dropout - prevents overfitting, improves generalization  
    3. LeakyReLU - prevents dying neurons, better gradient flow
    """
    
    def __init__(self, input_shape, n_actions, use_dueling=True, use_batch_norm=True, dropout_rate=0.0):
        """
        Args:
            input_shape: tuple (channels, height, width) - e.g., (4, 84, 84)
            n_actions: int - number of possible actions
            use_dueling: bool - whether to use dueling architecture
            use_batch_norm: bool - use batch normalization (recommended: True)
            dropout_rate: float - dropout probability (0.0 = no dropout)
        """
        super(DuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.use_dueling = use_dueling
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_output_size(input_shape)
        
        if use_dueling:
            # Dueling streams
            # Value stream
            self.value_fc1 = nn.Linear(conv_out_size, 512)
            self.value_fc2 = nn.Linear(512, 1)
            
            # Advantage stream
            self.advantage_fc1 = nn.Linear(conv_out_size, 512)
            self.advantage_fc2 = nn.Linear(512, n_actions)
        else:
            # Standard DQN
            self.fc1 = nn.Linear(conv_out_size, 512)
            self.fc2 = nn.Linear(512, n_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Network initialized with optimizations:")
        print(f"   Batch Normalization: {use_batch_norm}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   Activation: LeakyReLU (0.01)")
    
    def _get_conv_output_size(self, shape):
        """Calculate output size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.leaky_relu(self.bn1(self.conv1(dummy_input)), 0.01)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
            return int(np.prod(x.size()))
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for LeakyReLU."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # He initialization for LeakyReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    # Small positive bias to avoid dead neurons at initialization
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: torch.Tensor - input state (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor - Q-values for each action (batch_size, n_actions)
        """
        # Normalize input if not already normalized
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Convolutional layers with LeakyReLU + Batch Norm
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply dropout (only in training mode)
        x = self.dropout(x)
        
        if self.use_dueling:
            # Dueling architecture
            # Value stream
            value = F.leaky_relu(self.value_fc1(x), 0.01)
            value = self.dropout(value)  # Dropout before final layer
            value = self.value_fc2(value)
            
            # Advantage stream
            advantage = F.leaky_relu(self.advantage_fc1(x), 0.01)
            advantage = self.dropout(advantage)  # Dropout before final layer
            advantage = self.advantage_fc2(advantage)
            
            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Using mean instead of max for better gradient flow
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            x = F.leaky_relu(self.fc1(x), 0.01)
            x = self.dropout(x)
            q_values = self.fc2(x)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: np.ndarray or torch.Tensor - current state
            epsilon: float - exploration rate
            
        Returns:
            int - selected action
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).unsqueeze(0)
                
                # Set to eval mode for consistent behavior (disables dropout)
                was_training = self.training
                self.eval()
                q_values = self.forward(state)
                if was_training:
                    self.train()
                
                return q_values.argmax(dim=1).item()


class SimpleDQN(nn.Module):
    """
    Simpler DQN architecture for comparison/debugging.
    """
    
    def __init__(self, input_shape, n_actions):
        super(SimpleDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_output_size(input_shape)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        self.n_actions = n_actions
    
    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # Test the optimized network
    print("Testing Optimized DuelingDQN...")
    input_shape = (4, 84, 84)
    n_actions = 2
    
    net = DuelingDQN(input_shape, n_actions, use_dueling=True, use_batch_norm=True, dropout_rate=0.2)
    print(f"\nNetwork architecture:\n{net}")
    
    # Test forward pass
    dummy_input = torch.randn(8, *input_shape)  # Batch of 8
    
    # Training mode (with dropout)
    net.train()
    output_train = net(dummy_input)
    print(f"\n[Train Mode]")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_train.shape}")
    print(f"Output (Q-values): {output_train[0]}")
    
    # Eval mode (without dropout)
    net.eval()
    output_eval = net(dummy_input)
    print(f"\n[Eval Mode]")
    print(f"Output (Q-values): {output_eval[0]}")
    print(f"Difference (should be minimal): {(output_train[0] - output_eval[0]).abs().mean():.6f}")
    
    # Test action selection
    state = torch.randn(4, 84, 84)
    action = net.get_action(state, epsilon=0.1)
    print(f"\nSelected action: {action}")
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with vanilla
    print("\n" + "="*80)
    print("Comparison with vanilla DQN:")
    vanilla_net = DuelingDQN(input_shape, n_actions, use_dueling=True, use_batch_norm=False, dropout_rate=0.0)
    vanilla_params = sum(p.numel() for p in vanilla_net.parameters())
    print(f"Vanilla params: {vanilla_params:,}")
    print(f"Optimized params: {total_params:,}")
    print(f"Difference: +{total_params - vanilla_params:,} ({100*(total_params-vanilla_params)/vanilla_params:.1f}% increase)")
    print("Note: Batch Norm adds ~few hundred parameters, minimal overhead for big performance gain!")
