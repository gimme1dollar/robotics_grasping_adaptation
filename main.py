import pybullet as p
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count

import gym
import agent
from agent.common import io_utils

config = io_utils.load_yaml("config/gripper.yaml")
env = gym.make("gripper-env-v0", config=config)
total_timestep=100_000

img_h, img_w, img_c = 64, 64, 3
action_shape = env.action_space.shape
action_min = env.action_space.low
action_max = env.action_space.high

last_obs = env.reset()
with tqdm(total=total_timestep) as pbar:
    for t in count():
        # update progress bar
        pbar.n = t
        pbar.refresh()

        if t >= total_timestep:
            break

        action = np.random.rand(5)
        action = action * (action_max - action_min) + action_min
        obs, reward, done, _ = env.step(action)
        
        if done:
            obs = env.reset()
            #print(reward)
        last_obs = obs
