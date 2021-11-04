import argparse
import logging
import os
import gym

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
from agent.utils import io_utils, sb_utils, train_utils
from agent.utils.agent_utils import run_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Sub-command for training a policy
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--load_dir', type=str)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')

    train_parser.set_defaults(func=train_utils.train)

    # Sub-command for running a trained policy
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--model', type=str)
    run_parser.add_argument('--config', type=str, required=True)
    run_parser.add_argument('-v', '--visualize', action='store_true')
    run_parser.add_argument('-t', '--test', action='store_true')
    run_parser.add_argument('-s', '--stochastic', action='store_true')

    run_parser.set_defaults(func=train_utils.run)

    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()
    args.func(args)