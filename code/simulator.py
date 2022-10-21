import gym
import numpy as np
from itertools import count

def _simulate_full_episode(agent, rm=None):
    """
        Simulate a full episode
    """
    # Sim init
    env = gym.make("LunarLander-v2", render_mode=rm)
    observation, info = env.reset()
    
    # Data init
    total_reward = 0
    observations = []
    actions = []
    rewards = []

    for i in count():
        # Simulate an episode
        action = agent.policy(observation)
        
        # Log the observation and action taken
        observations.append(observation)
        actions.append(action)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Log the reward received
        rewards.append(reward)

        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return observations, actions, rewards, total_reward, i+1

def visualize(agent, num_sim=1, rm="human"):
    """
        Wrapper to simulate an arbitrary number of episodes and fetch the reward
    """
    return [_simulate_full_episode(agent, rm)[3] for i in range(num_sim)]

def simulate(agent):
    return _simulate_full_episode(agent)

def evaluate_fitness(agent, eps=5):
    scores = [_simulate_full_episode(agent)[3] for i in range(5)]
    return np.mean(scores), np.median(scores)