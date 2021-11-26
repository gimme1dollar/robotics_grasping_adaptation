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
    config = io_utils.load_yaml(args.config)
    env_name = "gripper-env-v1"

    time_str = time.strftime("%Y%m%d-%H%M%S")
    args.model_dir = f"{args.model_dir}/{env_name}_{time_str}"
    os.mkdir(args.model_dir)
    os.mkdir(args.model_dir + "/best_model") # Folder for best models
    algo = args.algo

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True
    if args.simple:
        logging.info("Simplified environment is set")
        config['simplified'] = True
    if args.shaped:
        logging.info("Shaped reward function is being used")
        config['reward']['shaped'] = True
    if args.timestep:
        config[algo]['total_timesteps'] = args.timestep
    if not args.algo == 'DQN':
        config['robot']['discrete'] = False    
    else:
        config['robot']['discrete'] = True
    
    config[algo]['save_dir'] = args.model_dir
    if args.timefeature:
        train_env = DummyVecEnv([lambda:  TimeFeatureWrapper(gym.make(env_name, config=config))])
    else:
        train_env = DummyVecEnv([lambda: Monitor(gym.make(env_name, config=config), os.path.join(args.model_dir, "log_file"))])
    
    config["algorithm"] = args.algo.lower()
    config_eval = config
    config_eval['simulation']['real_time'] = False
    config_eval['simulation']['visualize'] = False
    if args.timefeature:
        test_env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make(env_name, config=config_eval, evaluate=True, validate=True))])
    else:
        test_env = DummyVecEnv([lambda: gym.make(env_name, config=config_eval, evaluate=True, validate=True)])


    sb_help = sb_utils.SBPolicy(train_env, test_env, config, args.model_dir,
                                 args.load_dir, algo)
    sb_help.learn()


    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))
    train_env.close()
    test_env.close()

def run(args):
    env_name = "gripper-env-v1"
    config = io_utils.load_yaml("./config/gripper_env.yaml")
    normalize = config.get("normalize", False)

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    task = DummyVecEnv([lambda: gym.make(env_name, config=config, evaluate=False, test=args.test)])

    if normalize:
        task = VecNormalize(task, training=False, norm_obs=True, norm_reward=True,
                            clip_obs=10.)
        task = VecNormalize.load(os.path.join("./checkpoints/_final/baseline", 'vecnormalize.pkl'), task)
        
    # task = gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)
    model_lower = args.model.lower() 
    agent = sb.SAC.load(args.model)
    
    print("Run the agent")
    run_agent(task, agent, args.stochastic)
    task.close()

