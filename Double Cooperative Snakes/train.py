
import gym
import numpy as np
import torch

from snake_double import SnakeMultiAgentEnv

# from mppo_cnn import PPOAgent
# from mppo_lstm import PPOAgent
# from mppo_cnn_3 import PPOAgent
# from mppo_cnn_resnet import PPOAgent
from mppo_cnn_dilated import PPOAgent

import pygame
# Initialize Pygame
# pygame.init()
# pygame.font.init()
torch.set_num_threads(4)

HEIGHT = 10
WIDTH = 10
env = SnakeMultiAgentEnv(width=600, height=600, rows=HEIGHT, cols=HEIGHT)
agent = PPOAgent(height=HEIGHT,width=WIDTH)
checkpoint = 'ppo_double_10_cnn_dilated_snake_nov_5.pth'
# checkpoint = 'test_f.pth'

try:
    agent.load(checkpoint)
except Exception as e:
    print(e)


# Train the agent
agent.train(env, num_episodes=500000,checkpoint_path=checkpoint)


agent.plot("ppo_double_10_cnn_dilated_snake_nov_5.png")
# agent.plot("test.png")

agent.test(env)
