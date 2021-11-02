import argparse
import logging
import os
import gym
import time

import numpy as np
import stable_baselines as sb
import tensorflow as tf

from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sacMLP
from stable_baselines.bench import Monitor

import agent
from agent.utils.wrapper_utils import TimeFeatureWrapper
from agent.utils import io_utils
from agent.utils import sb_utils
from agent.utils.agent_utils import run_agent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(args):
    config = io_utils.load_yaml(args.config)

    env_name_idx = args.config.rfind('/')
    env_name_str = args.config[env_name_idx+1:]
    env_name = env_name_str.split("_")[0]
    if env_name == "robot":
        env_name_v = f"{env_name}-env-v0"
    elif env_name == "gripper":
        env_name_v = f"{env_name}-env-v1"
    print(env_name_v)

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
        env = DummyVecEnv([lambda:  TimeFeatureWrapper(gym.make(env_name_v, config=config))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make(env_name_v, config=config), os.path.join(args.model_dir, "log_file"))])
    
    config["algorithm"] = args.algo.lower()
    config_eval = config
    config_eval['simulation']['real_time'] = False
    config_eval['simulation']['visualize'] = False

    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))

    if args.timefeature:
        test_env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make(env_name_v, config=config_eval, evaluate=True, validate=True))])
    else:
        test_env = DummyVecEnv([lambda: gym.make(env_name_v, config=config_eval, evaluate=True, validate=True)])

    sb_help = sb_utils.SBPolicy(env, test_env, config, args.model_dir,
                                 args.load_dir, algo)
    sb_help.learn()
    env.close()
    test_env.close()

def run(args):
    top_folder_idx = args.model.rfind('/')
    top_folder_str = args.model[0:top_folder_idx]
    config_file = top_folder_str + '/config.yaml'
    config = io_utils.load_yaml(config_file)
    normalize = config.get("normalize", False)

    env_name_idx = args.config.rfind('/')
    env_name_str = args.config[env_name_idx+1:]
    env_name = env_name_str.split("_")[0]
    if env_name == "robot":
        env_name_v = f"{env_name}-env-v0"
    elif env_name == "gripper":
        env_name_v = f"{env_name}-env-v1"
    print(env_name_v)

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    task = DummyVecEnv([lambda: gym.make(env_name_v, config=config, evaluate=True, test=args.test)])

    if normalize:
        task = VecNormalize(task, training=False, norm_obs=True, norm_reward=True,
                            clip_obs=10.)
        task = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), task)
        
    # task = gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)
    model_lower = args.model.lower() 
    if 'trpo' == config["algorithm"]: 
        agent = sb.TRPO.load(args.model)
    elif 'sac' == config["algorithm"]:
        agent = sb.SAC.load(args.model)
    elif 'ppo' == config["algorithm"]:
        agent = sb.PPO2.load(args.model)
    elif 'dqn' == config["algorithm"]:
        agent = sb.DQN.load(args.model)
    elif 'bdq' == config["algorithm"]:
        agent = sb.BDQ.load(args.model)
    else:
        raise Exception
    print("Run the agent")
    run_agent(task, agent, args.stochastic)
    task.close()

