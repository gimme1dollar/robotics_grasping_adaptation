import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count

import gym
import agent
from model.DDPG import DDPG
from agent.gripperEnv.encoder import AutoEncoder, embed_state
from agent.common import io_utils

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb
import warnings
wandb.init(project="grasping_ddpg")
warnings.filterwarnings("ignore")

# config
config = io_utils.load_yaml("config/gripper.yaml")
visualize = config.get('visualize', True) 

# build env
env = gym.make("gripper-env-v0", config=config)
cur_state = env.reset()

# encoder
plt.ion()
encoder = AutoEncoder(config).to(device)
encoder.load_weight("./checkpoints/encoder_000018.pth")
enc_state = encoder.encode(embed_state(cur_state))
state_dim = enc_state.flatten().shape[0]

# model
img_h, img_w, img_c = 64, 64, 3
action_shape = env.action_space.shape
action_dim = action_shape[0]
action_min = float(env.action_space.low[0])
action_max = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, action_max)
wandb.watch(agent.actor_agent)
wandb.watch(agent.critic_agent)

# main loop
total_step, max_reward = 0, -float("inf")
total_timestep, max_episode = 5_000, 20_000_000
eval_steps, eval_rewards, eval_maxs = [], [], []
for epoch in range(max_episode):
    step, mean_reward = 0, 0.0
    agent.update_epsilone(epoch)

    cur_state = env.reset()
    cur_state = encoder.encode(embed_state(cur_state))
    cur_state = cur_state.flatten()

    for t in count():
        if t >= total_timestep:
            break

        # take action
        action = agent.epsilon_greedy_action(cur_state, action_dim, action_min, action_max)

        nxt_state, reward, done, info = env.step(action)
        nxt_state = encoder.encode(embed_state(nxt_state))
        nxt_state = nxt_state.flatten()

        agent.replay_buffer.push((cur_state, nxt_state, action, reward, np.float(done)))

        cur_state = nxt_state
        if done: obs = env.reset()

        # logging
        step += 1
        mean_reward += reward
        mean_reward /= total_timestep
        wandb.log({"reward/step": reward})
        wandb.log({"epsilon"    : agent.epsilon})
        wandb.log({"reward/max" : max_reward})
    total_step += step
    eval_steps.append(total_step)
    eval_rewards.append(mean_reward)
    eval_maxs.append(max_reward) if max_reward != -float("inf") else eval_maxs.append(mean_reward)
    print(f"Episode: \t{epoch} Total T:{total_step} Mean Reward: \t{mean_reward:0.2f} Epsilon: \t{agent.epsilon}")

    # agent traning
    agent.update(100)
    if max_reward < mean_reward:
        max_reward = mean_reward
        agent.save_weight(f"./checkpoints/agent_{epoch:06d}.pth")
    
plt.figure(figsize=(15, 15))
plt.plot(eval_steps, eval_rewards, 'r')
plt.plot(eval_steps, eval_maxs, 'b')
plt.savefig('./demo/ddpg_grasping.png')

