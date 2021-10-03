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
        self.epoch = 0
        self.rewards = []

    def _on_step(self) -> bool:
        # Log additional tensor
        stps = self.task.episode_step
        rew = self.task.episode_rewards
        sr = self.task.sr_mean

        if self.epoch % self.log_freq == 0:
            wandb.log({"epoch"        : self.epoch})
            wandb.log({"success_rate" : sr})
            wandb.log({"reward"       : np.mean(self.rewards)})
            wandb.log({"steps"        : stps})
            self.rewards=[]
        else:
            self.rewards.append(sum(rew))
            
        self.epoch += 1
        return True


class SBPolicy:
    def __init__(self, env, test_env, config, 
                model_dir='./checkpoints/sac'):
        self.env = env
        self.test_env = test_env
        self.config = config
        self.model_dir = model_dir
        self.norm = False #config['normalize']
 
    def learn(self):
        eval_path = self.model_dir

        # TODO save checkpoints with vecnormalize callback pkl file
        if self.norm:
            # Don't normalize the reward for test env
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False,
                                        clip_obs=10.)

        # TODO Use deterministic actions for evaluation
        eval_callback = EvalCallback(self.test_env, best_model_save_path=eval_path,
                                    log_path=eval_path, eval_freq=5_000,
                                    n_eval_episodes=10,
                                    deterministic=True, render=False)
        #checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=self.model_dir+'/logs/',
        #                                 name_prefix='rl_model')
        #time_callback = TrainingTimeCallback()

        policy = sacMlp
        model = sb.SAC(policy,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        verbose=2,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config['SAC']['buffer_size'],
                        batch_size=self.config['SAC']['batch_size'],
                        learning_rate=self.config['SAC']['step_size'])

        try:
            model.learn(total_timesteps=int(self.config['SAC']['total_timesteps']), 
                        callback=[LogCallback(self.env),
                                  eval_callback])
        except KeyboardInterrupt:
            pass

    def load(self):
        print("Loading the model")
        policy_kwargs = {}
        model = sb.SAC(sacMlp,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config['SAC']['buffer_size'],
                        batch_size=self.config['SAC']['batch_size'],
                        learning_rate=self.config['SAC']['step_size'])
        model_load = sb.SAC.load(self.model_dir, self.env)
        params = model_load.get_parameters()
        model.load_parameters(params, exact_match=False)
        return model
