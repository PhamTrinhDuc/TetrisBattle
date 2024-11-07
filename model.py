import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat([s.unsqueeze(0) if len(s.shape) == 3 else s for s in states]), 
                torch.tensor(actions), 
                torch.tensor(rewards, dtype=torch.float), 
                torch.cat([s.unsqueeze(0) if len(s.shape) == 3 else s for s in next_states]),
                torch.tensor(dones, dtype=torch.float))
    
    def __len__(self):
        return len(self.buffer)

class TetrisAgent:
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_update = 10
        self.learning_rate = 1e-4
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(100000)
        
        self.eps = self.eps_start
        self.steps_done = 0
        
    def select_action(self, state):
        if random.random() > self.eps:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device))
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample transitions
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute next Q values
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.eps = checkpoint['eps']
            self.steps_done = checkpoint['steps_done']