import os
import time
import logging
import numpy as np
import stable_baselines3 as sb

from stable_baselines3.sac.policies import MlpPolicy as sacMlp
from stable_baselines3.common.vec_env import  VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class LogCallback(BaseCallback):
    """
    Custom callback
    Success rate is integrated
    """
    def __init__(self, task, algo, log_freq, model_name, verbose=0):
        self.is_tb_set = False
        self.task = task
        self.algo = algo
        self.log_freq = log_freq
        self.old_timestep = -1
        self.model_name = model_name
        super(LogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        history = self.task.get_attr("history")[0]
        rew = self.task.get_attr("episode_rewards")[0]
        sr = self.task.get_attr("sr_mean")[0]
        curr = self.task.get_attr("curriculum")[0]

        if len(history) is not 0 and self.num_timesteps is not self.old_timestep:            
            if self.num_timesteps % self.log_freq == 0:
                print(f"model: {self.model_name} Success Rate: {sr} Timestep Num: {self.num_timesteps} lambda: {curr._lambda}")
            self.old_timestep = self.num_timesteps
        return True


class SBPolicy:
    def __init__(self, env, test_env, config, 
                model_dir='./checkpoints/', load_dir=None, algo='SAC', log_freq=1000):
        self.env = env
        self.test_env = test_env
        self.algo = algo
        self.config = config
        self.load_dir = load_dir
        self.model_dir = model_dir
        self.log_freq = log_freq
        self.norm = False #config['normalize']
 
    def learn(self):
        # Use deterministic actions for evaluation
        eval_path = self.model_dir + "/best_model"
        # TODO save checkpoints with vecnormalize callback pkl file
        if self.norm:
            # Don't normalize the reward for test env
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False,
                                        clip_obs=10.)
        eval_callback = EvalCallback(self.test_env, best_model_save_path=eval_path,
                                    log_path=eval_path+'/logs', eval_freq=50000,
                                    n_eval_episodes=10,
                                    deterministic=True, render=False)
        
        #checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=self.model_dir+'/logs/',
        #                                 name_prefix='rl_model')
        #time_callback = TrainingTimeCallback()

        policy_kwargs = {}
        policy = sacMlp
            
        model = sb.SAC(policy,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        verbose=2,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config[self.algo]['buffer_size'],
                        batch_size=self.config[self.algo]['batch_size'],
                        learning_rate=self.config[self.algo]['step_size'])

        try:
            model.learn(total_timesteps=int(self.config[self.algo]['total_timesteps']), 
                        callback=[#LogCallback(self.env, self.algo, self.log_freq, self.model_dir),
                                  eval_callback])
        except KeyboardInterrupt:
            pass

        self.save(model, self.model_dir)

    def load_params(self):
        print("Loading the model")
        for key, value in pars.items():
            if not 'action_value' in key and '2' in key:
                usable_params.update({key:value})
        model = sb.SAC(sacMlp,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        gamma=self.config['discount_factor'],
                        buffer_size=self.config[self.algo]['buffer_size'],
                        batch_size=self.config[self.algo]['batch_size'],
                        learning_rate=self.config[self.algo]['step_size'])
        model_load = sb.SAC.load(self.load_dir, self.env)
        params = model_load.get_parameters()
        model.load_parameters(params, exact_match=False)
        return model
    
    def save(self, model, model_dir):
        if '/' in model_dir:
            top_folder, model_name = model_dir.split('/')
        else:
            model_name = model_dir
        folder_path = model_dir + '/' + model_name

        if os.path.isfile(folder_path):
            print('File already exists \n')
            i = 1
            while os.path.isfile(folder_path + '.zip'):
                folder_path = '{}_{}'.format(folder_path, i)
                i += 1
            model.save(folder_path)
        else:
            print('Saving model to {}'.format(folder_path))
            model.save(folder_path)

        if self.norm:
            model.get_vec_normalize_env().save(os.path.join(model_dir, 'vecnormalize.pkl'))

