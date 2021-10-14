import os
import time
import logging
import numpy as np
import stable_baselines3 as sb
import wandb

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import  VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

class LogCallback(BaseCallback):
    """
    Custom callback
    Success rate is integrated
    """
    def __init__(self, task):
        super(LogCallback, self).__init__()
        self.task = task
        self.log_freq = 1000
        self.old_epoch = 1
        self.rewards = []

    def _on_step(self) -> bool:
        # Log additional tensor
        history = self.task.history
        self.rewards.append(self.task.episode_rewards)

        epoch = len(history)
        if self.old_epoch != epoch:
            wandb.log({"epoch"        : epoch})
            wandb.log({"success_rate" : np.mean(history[:-20]) if epoch > 20 else np.mean(history)})
            wandb.log({"reward"       : self.rewards[-2] if len(self.rewards) > 1 else 0}) 
            print(f"epoch : {epoch-1}", end="\t")
            print(f"success_rate: {np.mean(history[:-20]) if epoch > 20 else np.mean(history)}", end="\t")
            print(f"reward: {self.rewards[-2] if len(self.rewards) > 1 else 0}", end="\n")
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
        #noise_mean = [0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.05]
        #noise_std  = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #action_noise = NormalActionNoise(mean=noise_mean, sigma=noise_std)
        self.model = sb.SAC(MlpPolicy,
                        self.env,
                        policy_kwargs={},
                        verbose=1,
                        gamma=self.config['discount_factor'],
                        gradient_steps=self.config['SAC']['gradient_steps'],
                        learning_starts=self.config['warm_start']*self.env._time_horizon,
                        #action_noise=action_noise,
                        buffer_size=self.config['SAC']['buffer_size'],
                        batch_size=self.config['SAC']['batch_size'],
                        learning_rate=self.config['SAC']['step_size'])
 
        # Callbacks
        self.eval_callback = EvalCallback(self.test_env, 
                                    best_model_save_path=self.model_dir,
                                    log_path=self.model_dir, eval_freq=5_000,
                                    n_eval_episodes=10,
                                    deterministic=True, render=False)
        

    def learn(self):
        self.model.learn(total_timesteps=int(self.config['SAC']['total_epoch']*self.env._time_horizon), 
                        callback=[LogCallback(self.env), self.eval_callback])

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
