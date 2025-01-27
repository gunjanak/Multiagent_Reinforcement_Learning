#Altenative version of mppo_cnn
#This one contains three layers of cnn

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.init as init
import torch.nn.functional as F


from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.init as init

class ActorCritic(nn.Module):
    def __init__(self, height, width, hidden_dim, action_dim, num_agents):
        super(ActorCritic, self).__init__()

        # Helper function to calculate output size of convolutions
        def conv2d_size_out(size, kernel_size, padding=0, stride=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        # Define multi-scale CNN layers
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 3x3 filter
        self.conv1_2 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)  # 5x5 filter
        self.conv1_3 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3)  # 7x7 filter

        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)  # Combine feature maps

        # Calculate output sizes
        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=3, padding=1, stride=1), kernel_size=3, padding=1, stride=1)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=3, padding=1, stride=1), kernel_size=3, padding=1, stride=1)
        self.linear_input_size = convw * convh * 64  # Final output size from CNN

        # Fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for policy and value functions
        self.fc_pi = nn.Linear(hidden_dim, action_dim * num_agents)
        self.fc_v = nn.Linear(hidden_dim, num_agents)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)

        # Activation function
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        """Forward pass through the shared CNN layers."""
        # Multi-scale CNNs
        x1 = self.swish(self.conv1_1(x))
        x2 = self.swish(self.conv1_2(x))
        x3 = self.swish(self.conv1_3(x))

        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3), dim=1)

        # Further processing with a single CNN
        x = self.swish(self.conv2(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def pi(self, x):
        """Policy network for action logits."""
        x = self.forward(x)
        logits = self.fc_pi(x)
        return Categorical(logits=logits.view(x.size(0), -1, 4))  # Each agent gets its own logits

    def v(self, x):
        """Value network for state value predictions."""
        x = self.forward(x)
        value = self.fc_v(x)
        return value
    

class PPOAgent:
    def __init__(self, height, width, action_dim=4, buffer_size=10000, gamma=0.99,
                 K_epochs=4, eps_clip=0.2, hidden_dim=128, num_agents=2, device=None):
        self.policy = ActorCritic(height, width, hidden_dim, action_dim, num_agents)
        self.policy_old = ActorCritic(height, width, hidden_dim, action_dim, num_agents)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = self.policy.optimizer
        self.MseLoss = nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.rewards = []

    def update(self):
        states, actions, logprobs, rewards, is_terminals = zip(*self.memory)

        discounted_rewards = []
        discounted_reward = torch.zeros(len(rewards[0]))
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = torch.zeros_like(discounted_reward)
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.stack(discounted_rewards, dim=0).view(-1, 2)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0)) / (discounted_rewards.std(dim=0) + 1e-7)

        old_states = torch.cat(states).detach()
        old_actions = torch.cat(actions).detach()
        old_logprobs = torch.cat(logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, discounted_rewards) - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate(self, state, action):
        state_value = self.policy.v(state)
        dist = self.policy.pi(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def normalize_state(self, state):
        return (state - np.mean(state)) / (np.std(state) + 1e-8)
    

    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'rewards': self.rewards
        }
        torch.save(checkpoint, filename)
        print(f"Model and rewards saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.rewards = checkpoint.get('rewards', [])
        print(len(self.rewards))
        print(f"Model and rewards loaded from {filename}")




    def train(self, env, num_episodes, early_stopping=None, checkpoint_path=None):
        count = 0
        for episode in range(1, num_episodes + 1):
            total_rewards = np.zeros(2)
            state = env.reset()
            state = self.normalize_state(state)
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                dist = self.policy_old.pi(state_tensor)
                actions = dist.sample()  # Get actions for both agents
                # print(actions)
                
                # Correctly handle the tensor to scalar conversion
                action1 = actions[0][0].item()
                action2 = actions[0][1].item()

                # Execute actions in the environment
                next_state, reward, done, _ = env.step([action1, action2])

                self.memory.append((state_tensor, actions, dist.log_prob(actions), torch.FloatTensor(reward), done))

                state = next_state
                total_rewards += reward

                if done:
                    print(f"Episode: {episode} Reward: {total_rewards} Counts: {count+1}")
                    break
            if len(self.memory) > 1000:
                print(f"Updating {count+1}\n")
                count = count + 1
                self.update()
                self.memory.clear()
            self.rewards.append(total_rewards.sum())

            if early_stopping and early_stopping(self.rewards):
                print("Early stopping criterion met")
                if checkpoint_path:
                    self.save(checkpoint_path)
                break
            if (episode) % 100 == 0:
                self.save(checkpoint_path)
        print(f"total count: {count+1}")

        env.close()



    def test(self, env, num_episodes=10):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_rewards = np.zeros(2)
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                dist = self.policy_old.pi(state_tensor)
                actions = dist.sample()
                action1 = actions[0][0].item()
                action2 = actions[0][1].item()
                state, reward, done, _ = env.step([action1, action2])
                total_rewards += reward
            print(f"Episode {episode + 1}: Total Rewards: {total_rewards}")
            self.rewards.append(total_rewards.sum())
        env.close()

    def plot(self, plot_path):
        data = self.rewards

        # Calculate the moving average
        window_size = 1000
        moving_avg = pd.Series(data).rolling(window=window_size).mean()

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot the moving average line
        sns.lineplot(data=moving_avg, color='red')

        # Shade the area around the moving average line to represent the range of values
        plt.fill_between(range(len(moving_avg)),
                         moving_avg - np.std(data),
                         moving_avg + np.std(data),
                         color='blue', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Moving Average of Rewards')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(plot_path)
        # Show the plot
        plt.show()


