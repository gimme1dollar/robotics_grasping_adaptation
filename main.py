import os
import numpy as np
import pybullet as p
from tqdm import tqdm
from itertools import count

import gym
import agent
from model.DDPG import DDPG
from agent.common import io_utils
from agent.gripperEnv.robot import RobotEnv
from agent.gripperEnv.encoder import AutoEncoder, embed_state

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

agent = DDPG(state_dim, action_dim, action_max, config.get('DDPG'))
wandb.watch(agent.actor_agent)
wandb.watch(agent.critic_agent)

# main loop
max_episode = 1_000_000
total_step, success, save_flag = 0, [], -float("inf")
for epoch in range(max_episode):
    agent.update_epsilone(epoch)
    step, total_reward = 0, 0.0
    
    cur_state = env.reset()
    cur_state = encoder.encode(embed_state(cur_state))
    cur_state = cur_state.flatten()

    # take action 
    while True:
        action = agent.epsilon_greedy_action(cur_state, action_dim, action_min, action_max)

        nxt_state, reward, done, info = env.step(action)
        nxt_state = encoder.encode(embed_state(nxt_state))
        nxt_state = nxt_state.flatten()

        agent.replay_buffer.push((cur_state, nxt_state, action, reward, np.float(done)))

        cur_state = nxt_state
        if done: 
            if info['status'] == RobotEnv.Status.SUCCESS : success.append(1)
            else: success.append(0)
            break
            
        step += 1
        total_reward += reward
    total_step += step
    if len(success) < 100:
        success_rate = np.mean(success)
    else:
        success_rate = np.mean(success[-50:])
    
    # logging
    print(f"Episode: {epoch:7d} Total Reward: {total_reward:7.2f}\t", end=" ")
    print(f"Epsilon: {agent.epsilon:1.2f} \t", end=" ")
    print(f"Position: {info['position']} \tSuccess: {success_rate:0.2f} \t{info['status']}")
    wandb.log({"epoch"        : epoch})
    wandb.log({"total_step"   : total_step})
    wandb.log({"success_rate" : success_rate})
    wandb.log({"epsilon"      : agent.epsilon})
    wandb.log({"reward/total" : total_reward})

    # agent traning with warm start
    if epoch > 50:
        agent.update(1000) 
    
    # save
    if save_flag < success_rate: 
        save_flag = success_rate
        agent.save_weight(f"./checkpoints/agent_{epoch:06d}.pth")
    if success_rate > 0.8: exit()


