import unittest
import pytest
import numpy as np

import gym
import agent


CONFIG_TEST = 'test/config/test.yaml'
env_test = gym.make('grasping-env-v0', config=CONFIG_TEST)
ENVS_LIST = [env_test]

@pytest.mark.parametrize("env", ENVS_LIST)
def test_scene(env):
    env.reset()
    assert len(env.models) > 1
    
@pytest.mark.parametrize("env", ENVS_LIST)
def test_observation_space(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs_shape = env.observation_space.shape
    assert obs_shape == (64, 64, 1) or (64, 64, 3) or (64, 64, 5)

@pytest.mark.parametrize("env", ENVS_LIST)
def test_action_spaces(env):
    assert env.action_space == gym.spaces.Box(-1, 1, shape=(6,))

@pytest.mark.parametrize("env", ENVS_LIST)
def test_reset_return(env):
    obs = env.reset()
    assert obs.shape == (64, 64, 1) or (64, 64, 3) or (64, 64, 5)

@pytest.mark.parametrize("env", ENVS_LIST)
def test_step_return(env):
    env.reset()
    zero_action = np.zeros_like(env.action_space.sample())
    _, reward, done, _ = env.step(zero_action)
    
    assert done == True or done == False
    assert reward is not None

@pytest.mark.parametrize("env", ENVS_LIST)
def test_position(env):
    env.reset()
    pos_old, _ = env.get_pose()
    env.step(env.action_space.sample())
    pos_new, _ = env.get_pose()
    assert np.isclose(pos_new[2], pos_old[2], 4)

@pytest.mark.parametrize("env", ENVS_LIST)
def test_gripper(env):
    env.reset()
    env.close_gripper()
    print(env.get_gripper_width())
    assert (env.get_gripper_width() > 0)

    env.open_gripper()
    print(env.get_gripper_width())
    assert (env.get_gripper_width() < 0)


