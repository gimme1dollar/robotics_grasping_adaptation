import os
import numpy as np
import pybullet as p
from tqdm import tqdm
from itertools import count

import gym
import agent
from model import DDPG, SAC
from agent.utils import io_utils
from agent.robot.robot import RobotEnv
from agent.robot.encoder import AutoEncoder, embed_state

import stable_baselines3 as sb
from stable_baselines3.sac.policies import MlpPolicy 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import wandb
import warnings

def manual_test() :
    # config
    config = io_utils.load_yaml("config/robot.yaml")
    visualize = config.get('visualize', True) 

    # build env
    env = gym.make('grasping-env-v0', config=config)
    env.reset()

    # test
    passed = 0    
    while (True) :

        def get_grasp_position_angle(object_id):
            position, grasp_angle = np.zeros((3, 1)), 0
            # ========= PART 2============
            # Get position and orientation (yaw in radians) of the gripper for grasping
            # ==================================
            position, orientation = p.getBasePositionAndOrientation(object_id)
            grasp_angle = p.getEulerFromQuaternion(orientation)[2]
            return position, grasp_angle

        
        for _ in range(100):
            object_id = env.objects[0]
            position, grasp_angle = get_grasp_position_angle(object_id)
            env.manual_control(position, grasp_angle)
            env.reset()
        print()

def ddpg_test():
    #wandb.init(project="grasping_ddpg")
    warnings.filterwarnings("ignore")

    # config
    config = io_utils.load_yaml("config/robot.yaml")

    # build env
    env = gym.make("grasping-env-v0", config=config)
    cur_state = env.reset()

    # encoder
    encoder = AutoEncoder(config).to(device)
    encoder.load_weight("./checkpoints/ddpg/encoder_000018_.pth")
    enc_state = encoder.encode(embed_state(cur_state))
    state_dim = enc_state.flatten().shape[0]

    # model
    img_h, img_w, img_c = 64, 64, 3
    action_shape = env.action_space.shape
    action_dim = action_shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])

    #state_dim = img_h * img_w * img_c
    agent = DDPG.DDPG(state_dim, action_dim, action_max, config['DDPG'])
    #agent.load_weight("./checkpoints/ddpg/agent_000195_.pth")
    #wandb.watch(agent.actor_agent)
    #wandb.watch(agent.critic_agent)

    # main loop
    max_episode = 500_000
    success, total_step = [], 0
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

            step += 1; total_reward += reward
            cur_state = nxt_state
            if done: success.append(1) if info['status'] == RobotEnv.Status.SUCCESS else success.append(0); break
        total_step += step

        # success rate
        if len(success) < 100:
            success_rate = np.mean(success)
        else:
            success_rate = np.mean(success[-20:])
        
        # logging
        print(f"Episode: {epoch:7d} Total Reward: {total_reward:7.2f}\t", end=" ")
        print(f"Epsilon: {agent.epsilon:1.2f} \t", end=" ")
        print(f"Position: {info['position']} \tSuccess: {success_rate:0.2f} \t{info['status']}")
        #wandb.log({"epoch"        : epoch})
        #wandb.log({"total_step"   : total_step})
        #wandb.log({"success_rate" : success_rate})
        #wandb.log({"epsilon"      : agent.epsilon})
        #wandb.log({"reward/total" : total_reward})

        # agent traning with warm start
        if epoch > 0:
            agent.update(1000) 
        
        # save
        if success_rate > 0.95 and len(success) > 20:
            agent.save_weight(f"./checkpoints/ddpg/agent_{epoch:06d}.pth")
            #exit()

def sac_test():
    #wandb.init(project="grasping_sac")
    warnings.filterwarnings("ignore")

    # config
    config = io_utils.load_yaml("config/robot.yaml")

    # build env
    env = gym.make('grasping-env-v0', config=config)

    # encoder
    sac = SAC.SAC(env, env, config)
    sac.learn()
    env.close()

def main(args):
    if args.exp_algo == "manual":
        manual_test()
    elif args.exp_algo == "ddpg":
        ddpg_test()
    elif args.exp_algo == "sac":
        sac_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-algo', type=str, default='manual',
                        help='Name of the algorithm (default: manual)')

    args = parser.parse_args()
    main(args)