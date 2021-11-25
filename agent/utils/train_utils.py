import argparse
import logging
import os
import gym
import time

import numpy as np
import stable_baselines3 as sb

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import agent
from agent.utils.wrapper_utils import TimeFeatureWrapper
from agent.utils import io_utils
from agent.utils import sb_utils
from agent.utils.agent_utils import run_agent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(args):
    env_name = f"robot-env-v0"
    algo = "SAC"

    # open config file
    config = io_utils.load_yaml(args.config)

    # make directory for save files
    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.model_dir = f"{args.model_dir}/{env_name}_{time_str}"
    os.mkdir(args.model_dir)
    os.mkdir(args.model_dir + "/best_model") # Folder for best models

    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))
    

    config['robot']['discrete'] = False    
    config[algo]['save_dir'] = args.model_dir
    
    # make env
    config_ = config
    config_['simulation']['real_time'] = False
    config_['simulation']['visualize'] = False

    train_env = DummyVecEnv([lambda: Monitor(gym.make(env_name, config=config), os.path.join(args.model_dir, "log_file"))])
    test_env = DummyVecEnv([lambda: gym.make(env_name, config=config_, evaluate=True, validate=True)])
    
    # run ehlper
    trainer = sb_utils.SBPolicy(train_env, test_env, config, 
                                args.model_dir, args.load_dir, algo)
    trainer.learn()

    print("training is over")

    train_env.close()
    test_env.close()
    return

def run(args):
    config = io_utils.load_yaml(args.config)

    env_name_idx = args.config.rfind('/')
    env_name_str = args.config[env_name_idx+1:]
    env_name = env_name_str.split("_")[0]
    if env_name == "robot":
        env_name_v = f"{env_name}-env-v0"
    elif env_name == "gripper":
        env_name_v = f"{env_name}-env-v1"
    env_name_v = f"gripper-env-v1"
    print(env_name_v)

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    task = DummyVecEnv([lambda: gym.make(env_name_v, config=config, evaluate=True, test=True)])
    task = VecNormalize.load(os.path.join('checkpoints/_final/baseline', 'vecnormalize.pkl'), task)
        
    # task = gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)
    agent = sb.SAC.load(args.model)

    print("Run the agent")
    run_agent(task, agent, args.stochastic)
    task.close()

