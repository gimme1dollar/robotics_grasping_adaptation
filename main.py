import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count

import gym
import agent
from model import DDPG
from agent.utils import io_utils
from agent.utils.augment_utils import *
from agent.robot.robot import ArmEnv, GripperEnv
from agent.robot.encoder import AutoEncoder, embed_state

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")

def manual(config, env):
    done = False
    while(True):
        action = np.zeros((5,), dtype=np.float)
        query = input()

        if (query == "w"): #front
            action[0] = 1.
        elif (query == "s"): #back 
            action[0] = -1.
        elif (query == "d"): #right
            action[1] = 1.
        elif (query == "a"): #left
            action[1] = -1.
        elif (query == "t"): #down
            action[2] = 1.
        elif (query == "r"): #up
            action[2] = -1.
        elif (query == "q"): #rotate clockwise
            action[3] = 1.
        elif (query == "e"): #rotate counterclockwise
            action[3] = -1.
        elif (query == "f"): #close gripper
            action[4] = 1.
        elif (query == "g"): #open gripper
            action[4] = -1.

        obs, reward, done, _ = env.step(action)
        obs = obs.astype(np.uint8)

        if done:
            obs = env.reset()

def baseline(config, env,
            exp_algo='DDPG', exp_ckpt=None, exp_enco=False, exp_mode='test',
            exp_logg=False):
    '''
        DDPG baseline training
    '''

    # model
    img_h, img_w, img_c = 64, 64, 4
    state_dim = img_h * img_w * img_c
    
    action_shape = env.action_space.shape
    action_dim = action_shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])

    agent = DDPG.DDPG(state_dim, action_dim, action_max, config.get('DDPG'))
    if exp_mode == 'test':
        agent.load_weight(f"{exp_ckpt}")
    if exp_logg == True:
        wandb.init(project="grasping_ddpg")
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
            if exp_mode == 'train':
                action = agent.epsilon_greedy_action(cur_state, action_dim, action_min, action_max)
            elif exp_mode == 'test':
                action = agent.act(cur_state)
                
            nxt_state, reward, done, status = env.step(action)
            nxt_state = nxt_state.flatten()

            agent.replay_buffer.push((cur_state, nxt_state, action, reward, np.float(done)))

            step += 1; total_reward += reward
            cur_state = nxt_state
            if done: success.append(1) if status == ArmEnv.Status.SUCCESS else success.append(0); break

        total_step += step
        if len(success) < 100:
            success_rate = np.mean(success)
        else:
            success_rate = np.mean(success[20:])
        
        # logging
        print(f"Episode: {epoch:7d} Total Reward: {total_reward:7.2f}\t", end=" ")
        print(f"Epsilon: {agent.epsilon:1.2f} \t", end=" ")
        print(f"Success: {success_rate:0.2f} \t{status}")
        if exp_logg == True:
            wandb.log({"epoch"        : epoch})
            wandb.log({"total_step"   : total_step})
            wandb.log({"status"       : status.value})
            wandb.log({"success_rate" : success_rate})
            wandb.log({"epsilon"      : agent.epsilon})
            wandb.log({"reward/total" : total_reward})

        # agent traning with warm start
        if epoch > config['policy']['warm_start']:
            agent.update(100) 
        
        # save
        if success_rate > 0.8 and len(success) > 10:
            agent.save_weight(f"./checkpoints/agent_{epoch:06d}.pth")
            exit()
    return

def image_aug(config, 
            exp_algo='DDPG', exp_enco=False, exp_mode='test',
            exp_logg=False):
    return

def dynamics_aug(config, 
            exp_algo='DDPG', exp_enco=False, exp_mode='test',
            exp_logg=False):
    return

def hybrid_aug(config, 
            exp_algo='DDPG', exp_enco=False, exp_mode='test',
            exp_logg=False):
    return


def main(args):
    env_name = args.env_name
    env_type = args.env_type
    exp_algo = args.exp_algorithm
    exp_mode = args.exp_mode
    exp_ckpt = args.exp_checkpoint
    exp_enco = args.exp_encoder
    exp_logg = args.exp_logging
    config = io_utils.load_yaml(f"config/{env_name}_env.yaml")

    assert( exp_mode == "train" or exp_mode == "test")
    if exp_mode == "test":
        assert(exp_ckpt is not None)

    print()
    print(f"=======================================================")
    print(f"========== env_name: {env_name} ")
    print(f"========== env_type: {env_type} ")
    print(f"========== exp_algo: {exp_algo} ")
    print(f"========== exp_mode: {exp_mode} ")
    print(f"========== exp_ckpt: {exp_ckpt} ")
    print(f"========== exp_enco: {exp_enco} ")
    print(f"========== exp_logg: {exp_logg} ")
    print(f"=======================================================")
    print()


    env = gym.make(f'{env_name}-env-v0', config=config)
    
    if env_type == "manual":
        print("manual control")
        manual(config, env)    

    elif env_type == "default":
        print("environment for basline model")
        baseline(config, env,
                    exp_algo, exp_ckpt, exp_enco, exp_mode, exp_logg)

    elif env_type == "image":
        print("image augmentation")
        image_aug(config, 
                    exp_algo, exp_enco, exp_mode, exp_logg)

    elif env_type == "dynamics":
        print("dynamics augmentation")
        dynamics_aug(config, 
                    exp_algo, exp_enco, exp_mode, exp_logg)

    elif env_type == "hybrid":
        print("hybrid augmentation")
        hybrid_aug(config, 
                    exp_algo, exp_enco, exp_mode, exp_logg)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='robot',
                        help='Env name (default: robot_env)')
    parser.add_argument('--env-type', type=str, default='default',
                        help='Env type (default/image/dynamics/hybrid) (default: default)')
    parser.add_argument('--exp-algorithm', type=str, default='DDPG',
                        help='Exp algorithm (default: DDPG)')
    parser.add_argument('--exp-mode', type=str, default='test',
                        help='Exp mode (train/test) (default: test)')
    parser.add_argument('--exp-checkpoint', type=str, default='None',
                        help='Exp checkpoint')
    parser.add_argument('--exp-encoder', type=str, default='False',
                        help='Exp with encoder (default: False)')
    parser.add_argument('--exp-logging', type=str, default='False',
                        help='Exp with logging on wandb (default: False)')
    args = parser.parse_args()

    main(args)