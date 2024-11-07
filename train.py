import numpy as np
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from model import TetrisAgent
import torch
from datetime import datetime
import os

def preprocess_state(state):
    # Convert to float and normalize
    state = state.astype(np.float32) / 255.0  # Normalize if the input is in range [0, 255]
    
    # Add channel dimension if needed
    if len(state.shape) == 2:
        state = np.expand_dims(state, axis=0)
    
    return torch.FloatTensor(state)

def train_agent(episodes=10000, max_steps=10000):
    # Create environment
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    
    # Get initial state to determine shape
    initial_state = env.reset()
    print("Original state shape:", initial_state.shape)
    processed_state = preprocess_state(initial_state)
    print("Processed state shape:", processed_state.shape)
    
    # Initialize agent with correct state shape
    state_shape = processed_state.shape  # This should be (channels, height, width)
    n_actions = env.action_space.n
    agent = TetrisAgent(state_shape, n_actions)
    
    # Training variables
    best_reward = float('-inf')
    episode_rewards = []
    
    # Create directory for saving models
    save_dir = "tetris_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)
            episode_reward += reward
            
            # Store transition and train
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(save_dir, 'best_model.pth'))
        
        # Save checkpoint periodically
        if episode % 100 == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_{episode}.pth'))
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}")
            print(f"Average Reward (last 10): {avg_reward:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Epsilon: {agent.eps:.3f}")
            print("------------------------")
    
    return agent, episode_rewards

if __name__ == "__main__":
    # Train the agent
    agent, rewards = train_agent()
    
    # Plot training progress
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_progress.png')
    plt.close()