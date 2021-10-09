import os
import time
import logging
import numpy as np
import stable_baselines3 as sb
import wandb

from stable_baselines3.sac.policies import MlpPolicy as sacMlp
from stable_baselines3.common.vec_env import  VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class LogCallback(BaseCallback):
    """
    Custom callback
    Success rate is integrated
    """
    def __init__(self, task):
        super(LogCallback, self).__init__()
        self.task = task
        self.log_freq = 1000
        self.old_epoch = 0
        self.rewards = []

    def _on_step(self) -> bool:
        # Log additional tensor
        history = self.task.history
        self.rewards.append(self.task.episode_rewards)

        epoch = len(history)
        if self.old_epoch != epoch:
            #wandb.log({"epoch"        : epoch})
            #wandb.log({"success_rate" : np.mean(history[:-20]) if epoch > 20 else np.mean(history)})
            #wandb.log({"reward"       : self.rewards[-2] if len(self.rewards) > 1 else 0}) 
            print(f"epoch : {epoch}")
            print(f"success_rate: {np.mean(history[:-20]) if epoch > 20 else np.mean(history)}")
            print(f"reward: {self.rewards[-2] if len(self.rewards) > 1 else 0}") 
            print()
            self.rewards = []
            self.old_epoch = epoch
        return True


class SAC:
    def __init__(self, env, test_env, config, 
                model_dir='./checkpoints/sac'):
        self.config = config
        self.model_dir = model_dir

        # Environment
        self.env = env
        self.test_env = test_env
        if config['normalize']:
            # Don't normalize the reward for test env
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False,
                                        clip_obs=10.)

        # Model
        self.model = sb.SAC(sacMlp,
                        self.env,
                        policy_kwargs={},
                        verbose=2,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config['SAC']['buffer_size'],
                        batch_size=self.config['SAC']['batch_size'],
                        learning_rate=self.config['SAC']['step_size'])
 
        # Callbacks
        self.eval_callback = EvalCallback(self.test_env, best_model_save_path=self.model_dir,
                                    log_path=self.model_dir, eval_freq=5_000,
                                    n_eval_episodes=10,
                                    deterministic=True, render=False)
        

    def learn(self):
        self.model.learn(total_timesteps=int(self.config['SAC']['total_timesteps']), 
                        callback=[LogCallback(self.env),
                                  self.eval_callback])

    def load(self):
        print("Loading the model")
        model = sb.SAC(sacMlp,
                        self.env,
                        policy_kwargs={},
                        verbose=1,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config['SAC']['buffer_size'],
                        batch_size=self.config['SAC']['batch_size'],
                        learning_rate=self.config['SAC']['step_size'])
        model_load = sb.SAC.load(self.model_dir, self.env)
        params = model_load.get_parameters()
        model.load_parameters(params, exact_match=False)
        return model
