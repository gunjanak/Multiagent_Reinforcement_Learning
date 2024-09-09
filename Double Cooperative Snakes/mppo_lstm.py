import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.init as init

from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


class ActorCritic(nn.Module):
    def __init__(self, height=10, width=10, hidden_dim=128, action_dim=4, num_agents=2):
        super(ActorCritic, self).__init__()
        
        # LSTM takes the input as (batch_size, seq_length, input_dim)
        self.lstm_input_size = height * width  # The input for each time step (flattened)
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for policy and value function
        self.fc_pi = nn.Linear(hidden_dim, action_dim * num_agents)
        self.fc_v = nn.Linear(hidden_dim, num_agents)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        
    def pi(self, x):
        batch_size = x.size(0)
        
        # Reshape input for LSTM: (batch_size, seq_length, input_size)
        x = x.view(batch_size, -1, self.lstm_input_size)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = torch.tanh(self.fc1(lstm_out))
        x = torch.tanh(self.fc2(x))
        
        # Policy output (logits for actions)
        pi_out = self.fc_pi(x)
        return Categorical(logits=pi_out.view(batch_size, -1, 4))  # Output logits for each agent

    def v(self, x):
        batch_size = x.size(0)
        
        # Reshape input for LSTM: (batch_size, seq_length, input_size)
        x = x.view(batch_size, -1, self.lstm_input_size)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = torch.tanh(self.fc1(lstm_out))
        x = torch.tanh(self.fc2(x))
        
        # Value function output
        v_out = self.fc_v(x)
        return v_out




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
        print(f"Model and rewards loaded from {filename}")




    def train(self, env, num_episodes, early_stopping=None, checkpoint_path=None):
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
                    print(f"Episode: {episode} Reward: {total_rewards}")
                    break
            if len(self.memory) > 100:
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


