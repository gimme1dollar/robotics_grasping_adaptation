import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count

import gym
import agent
from model import DDPG, SAC
from agent.utils import io_utils
from agent.utils.augment_utils import *
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
warnings.filterwarnings("ignore")


def default_test(config, is_training):
    # build env
    env = gym.make('gripper-env-v0', config=config)
    cur_state = env.reset()
    wandb.init(project="grasping_ddpg")

    # model
    img_h, img_w, img_c = 64, 64, 4
    state_dim = img_h * img_w * img_c
    
    action_shape = env.action_space.shape
    action_dim = action_shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])

    agent = DDPG.DDPG(state_dim, action_dim, action_max, config.get('DDPG'))
    #agent.load_weight("./checkpoints/gripper/agent_000195_encoder.pth")
    wandb.watch(agent.actor_agent)
    wandb.watch(agent.critic_agent)

    # main loop
    max_episode = 1_000_000
    total_step, success, save_flag = 0, [], -float("inf")
    for epoch in range(max_episode):
        agent.update_epsilone(epoch)
        step, total_reward = 0, 0.0
        
        cur_state = env.reset()
        cur_state = cur_state.flatten()

        # take action 
        while True:
            if is_training == True:
                action = agent.epsilon_greedy_action(cur_state, action_dim, action_min, action_max)
            else:
                action = agent.act(cur_state)
                
            nxt_state, reward, done, status = env.step(action)
            nxt_state = nxt_state.flatten()

            agent.replay_buffer.push((cur_state, nxt_state, action, reward, np.float(done)))

            step += 1; total_reward += reward
            cur_state = nxt_state
            if done: success.append(1) if status == RobotEnv.Status.SUCCESS else success.append(0); break
        total_step += step
        if len(success) < 100:
            success_rate = np.mean(success)
        else:
            success_rate = np.mean(success[20:])
        
        # logging
        print(f"Episode: {epoch:7d} Total Reward: {total_reward:7.2f}\t", end=" ")
        print(f"Epsilon: {agent.epsilon:1.2f} \t", end=" ")
        print(f"Success: {success_rate:0.2f} \t{status}")
        wandb.log({"epoch"        : epoch})
        wandb.log({"total_step"   : total_step})
        wandb.log({"status"       : status.value})
        wandb.log({"success_rate" : success_rate})
        wandb.log({"epsilon"      : agent.epsilon})
        wandb.log({"reward/total" : total_reward})

        # agent traning with warm start
        if epoch > 0:
            agent.update(100) 
        
        # save
        #if save_flag < success_rate: 
        #    save_flag = success_rate
        if success_rate > 0.8 and len(success) > 10:
            agent.save_weight(f"./checkpoints/agent_{epoch:06d}.pth")
            exit()
    return

def image_test(config) :
    return

def dynamics_test(config) :
    return

def manual_test(config) :
    # build env
    env = gym.make('grasping-env-v0', config=config)
    env.reset()

    # test 
    while (True):
        def get_grasp_position_angle(object_id):
            position, orientation = p.getBasePositionAndOrientation(object_id)
            grasp_angle = p.getEulerFromQuaternion(orientation)[2]
            return position, grasp_angle

        for _ in range(100):
            object_id = env.objects[0]
            position, grasp_angle = get_grasp_position_angle(object_id)
            env.manual_control(position, grasp_angle)
            env.reset()
        print()

def ddpg_test(config, is_training):
    wandb.init(project="grasping_ddpg")

    # build env
    env = gym.make("grasping-env-v0", config=config)
    cur_state = env.reset()

    # encoder
    #encoder = AutoEncoder(config).to(device)
    #encoder.load_weight("./checkpoints/ddpg/encoder_000018_.pth")
    #enc_state = encoder.encode(embed_state(cur_state))
    #state_dim = enc_state.flatten().shape[0]

    # model
    img_h, img_w, img_c = 64, 64, 4
    action_shape = env.action_space.shape
    action_dim = action_shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])
    state_dim = img_h * img_w * img_c
    agent = DDPG.DDPG(state_dim, action_dim, action_max, config['policy']['DDPG'])
    #agent.load_weight("./checkpoints/ddpg/agent_000195_.pth")
    wandb.watch(agent.actor_agent)
    wandb.watch(agent.critic_agent)

    # main loop
    max_episode = env.config['simulation']['max_episode']
    success, total_step = [], 0
    for epoch in range(max_episode):
        agent.update_epsilone(epoch)
        step, total_reward = 0, 0.0
        
        cur_state = env.reset()
        #cur_state = encoder.encode(embed_state(cur_state))
        cur_state = cur_state.flatten()

        # take action 
        while True:
            if is_training:
                action = agent.epsilon_greedy_action(cur_state, action_dim, action_min, action_max)
                
                # Warm start
                if env.epoch < env.config['policy']['warm_start']:
                    position, angle = p.getBasePositionAndOrientation(env.objects[0])
                    orientation = p.getEulerFromQuaternion(angle)[2]
                    answer = np.asarray(position)
                    answer = np.append(answer, np.asarray(orientation))
                    answer = np.append(answer, [0])
                    action = tuple(map(sum, zip(action, answer)))
            else:
                action = agent.act(cur_state)

            nxt_state, reward, done, status = env.step(action)

            # Visualize state
            #nxt_state[:,:,-1] = image_noise(nxt_state[:,:,-1])
            #plt.imshow(nxt_state[:,:,-1], cmap='gray')
            #plt.show()
            #plt.pause(0.1) 

            #nxt_state = encoder.encode(embed_state(nxt_state))
            nxt_state = nxt_state.flatten()

            agent.replay_buffer.push((cur_state, nxt_state, action, reward, np.float(done)))

            step += 1; total_reward += reward
            cur_state = nxt_state
            if done: success.append(1) if status == RobotEnv.Status.SUCCESS else success.append(0); break
        total_step += step

        # success rate
        if len(success) < 100:
            success_rate = np.mean(success)
        else:
            success_rate = np.mean(success[-20:])
        
        # logging
        print(f"Episode: {epoch:7d} Total Reward: {total_reward:7.2f}\t", end=" ")
        print(f"Epsilon: {agent.epsilon:1.2f} \t", end=" ")
        print(f"Success: {success_rate:0.2f} \t{status}")
        wandb.log({"epoch"        : epoch})
        wandb.log({"total_step"   : total_step})
        wandb.log({"success_rate" : success_rate})
        wandb.log({"status"       : status.value})
        wandb.log({"epsilon"      : agent.epsilon})
        wandb.log({"reward/total" : total_reward})
        

        # agent traning with warm start
        if epoch > 0:
            agent.update(10) 
        
        # save
        if success_rate > 0.7 and len(success) > 20:
            agent.save_weight(f"./checkpoints/robot/agent_{epoch:06d}.pth")
            #exit()

def sac_test(config):
    wandb.init(project="grasping_sac")

    # build env
    env = gym.make('grasping-env-v0', config=config)

    # encoder
    sac = SAC.SAC(env, env, config['policy'])
    sac.learn()
    
    env.close()

def main(args):
    env_name = args.env_name
    
    config = io_utils.load_yaml(f"config/{env_name}.yaml")
    exp_algo = config['policy']['algo_type']
    exp_mode = config['simulation']['mode']
    is_training = True if args.env_mode == 'training' else False
        
    if env_name == "robot_env":
        if exp_algo == "manual":
            print("manual")
            manual_test(config)
        elif exp_algo == "DDPG":
            print("ddpg")
            ddpg_test(config, is_training)
        elif exp_algo == "SAC":
            print("sac")
            sac_test(config)
            
            
    elif env_name == "gripper_env":
        if exp_mode == "default":
            print("default")
            default_test(config, is_training)
        elif exp_mode == "image":
            print("image")
            image_test(config)
        elif exp_mode == "dynamics":
            print("dynamics")
            dynamics_test(config)
            
        elif exp_mode == "encoder":
            print("image")
            image_test(config)
        elif exp_mode == "augmented":
            print("image")
            image_test(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='robot_env',
                        help='Name of the environment (default: robot_env)')
    parser.add_argument('--env-mode', type=str, default='test',
                        help='Mode of the program (default: test)')
    args = parser.parse_args()
    main(args)