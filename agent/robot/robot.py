import math
import os
import time
import numpy as np
import functools
import collections
from enum import Enum

import pybullet as p

import gym
from gym import spaces 

from agent.utils import io
from agent.utils import transform
from agent.utils import curriculum

from agent.world.model import Model_gripper
from agent.robot import sensor, actuator
from agent.robot.reward import GripperCustomReward
from agent.world.world import World 


def _reset(robot, actuator, depth_sensor):
    """Reset until an object is within the fov of the camera."""
    ok = False
    while not ok:
        robot.reset_sim() #world + scene reset
        robot.reset_model() #robot model
        actuator.reset()
        _, _, mask = depth_sensor.get_state()
        
        #ok = len(np.unique(mask)) > 2  # plane and gripper are always visible
        ok = True

class GripperEnv(World):
    class Events(Enum):
        START_OF_EPISODE = 0
        END_OF_EPISODE = 1
        CLOSE = 2
        CHECKPOINT = 3

    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3
        
    def __init__(self, config, evaluate=False, test=False, validate=False):
        if not isinstance(config, dict):
            config = io.load_yaml(config)
        super().__init__(config, evaluate=evaluate, test=test, validate=validate)
        self._step_time = collections.deque(maxlen=10000)
        self._workspace = {'lower': np.array([-1., -1., -1]),
                           'upper': np.array([1., 1., 1.])}
        # Get configurations
        self.time_horizon = config['simulation']['time_horizon']
        self.model_path = config['robot']['model_path']
        self.depth_obs = config.get('depth_observation', False)
        self.full_obs = config.get('full_observation', False)

        # Assign models
        self._model = None
        self._joints = None
        self._initial_height = 0.3
        self._init_ori = transform.quaternion_from_euler(np.pi, 0., 0.)
        self.main_joints = [0, 1, 2, 3] #FIXME make it better
        
        # Assign actuator
        self._left_finger_id = 7
        self._right_finger_id = 9
        self._fingers = [self._left_finger_id, self._right_finger_id]
        self._left_finger, self._right_finger = None, None
        self._actuator = actuator.Gripper(self, config)

        # Assign the sensors
        self._camera = sensor.RGBDSensor(config['sensor'], self)
        if self.depth_obs or self.full_obs:
            self._sensors = [self._camera]

        # Assign the reward fn
        self._reward_fn = GripperCustomReward(config['reward'], self)

        # Assign curriculum (for better training)
        self.curriculum = curriculum.WorkspaceCurriculum(config['curriculum'], self, evaluate)
        self.history = self.curriculum._history

        # Assign callbacks
        self.sr_mean = 0.
        self._callbacks = {GripperEnv.Events.START_OF_EPISODE: [],
                        GripperEnv.Events.END_OF_EPISODE: [],
                        GripperEnv.Events.CLOSE: [],
                        GripperEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)
        self.setup_spaces()

    def setup_spaces(self):
        self.action_space = self._actuator.setup_action_space()
        if not self.depth_obs and not self.full_obs:
            low, high = np.array([]), np.array([])
            for sensor in self._sensors:
                low = np.append(low, sensor.state_space.low)
                high = np.append(high, sensor.state_space.high)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        else:
            shape = self._camera.state_space.shape
            
            if self.full_obs: # RGB + Depth + Actuator
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(shape[0], shape[1], 5))
            else: # Depth + Actuator obs
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(shape[0], shape[1], 2))

    def register_events(self, evaluate, config):
        # Setup the reset function
        reset = functools.partial(_reset, self, self._actuator, self._camera)

        # Register callbacks
        self.register_callback(GripperEnv.Events.START_OF_EPISODE, reset)
        self.register_callback(GripperEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(GripperEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(GripperEnv.Events.END_OF_EPISODE, self.curriculum.update)
        self.register_callback(GripperEnv.Events.CLOSE, super().close)

    def reset(self):
        self._trigger_event(GripperEnv.Events.START_OF_EPISODE)
        self.episode_step = 0
        self.episode_rewards = np.zeros(self.time_horizon)
        self.status = GripperEnv.Status.RUNNING
        self.obs = self._observe()

        return self.obs

    def reset_model(self):
        """Reset the task.

        Returns:
            Observation of the initial state.
        """
        self.endEffectorAngle = 0.
        start_pos = [0., 0., self._initial_height]
        self._model = self.add_model(self.model_path, start_pos, self._init_ori)
        self._joints = self._model.joints
        self.robot_id = self._model.model_id
        self._left_finger = self._model.joints[self._left_finger_id]
        self._right_finger = self._model.joints[self._right_finger_id]

    def add_model(self, path, start_pos, start_orn, scaling=1.):
        model = Model_gripper(self.physics_client)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model

    def _trigger_event(self, event, *event_args):
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))

    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns:
            A tuple (obs, reward, done, info), where done is a boolean flag
            indicating whether the current episode finished.
        """
        if self._model is None:
            self.reset()

        self._actuator.step(action)

        new_obs = self._observe()

        reward, self.status = self._reward_fn(self.obs, action, new_obs)
        self.episode_rewards[self.episode_step] = reward

        if self.status != GripperEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            done, self.status = True, GripperEnv.Status.TIME_LIMIT
        else:
            done = False

        if done:
            self._trigger_event(GripperEnv.Events.END_OF_EPISODE, self)

        self.episode_step += 1
        self.obs = new_obs
        if len(self.curriculum._history) != 0:
            self.sr_mean = np.mean(self.curriculum._history)
        super().step_sim(1)
        
        return self.obs, reward, done, {"is_success":self.status==GripperEnv.Status.SUCCESS,
                                         "episode_step": self.episode_step, 
                                         "episode_rewards": self.episode_rewards, 
                                         "lambda": self.curriculum._lambda,
                                         "status": self.status}

    def _observe(self):
        if not self.depth_obs and not self.full_obs:
            obs = np.array([])
            for sensor in self._sensors:
                obs = np.append(obs, sensor.get_state())
            return obs
        else:
            rgb, depth, _ = self._camera.get_state()
            sensor_pad = np.zeros(self._camera.state_space.shape[:2])

            sensor_pad[0][0] = self._actuator.get_state()
            if self.full_obs:
                obs_stacked = np.dstack((rgb, depth, sensor_pad))
            else:
                obs_stacked = np.dstack((depth, sensor_pad))
            return obs_stacked

    def absolute_pose(self, target_pos, target_orn):
        # target_pos = self._enforce_constraints(target_pos)

        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)

        # _, _, yaw = transform_utils.euler_from_quaternion(target_orn)
        # yaw *= -1
        yaw = target_orn
        comp_pos = np.r_[target_pos, yaw]

        for i, joint in enumerate(self.main_joints):
            self._joints[joint].set_position(comp_pos[i])
        
        self.run(0.1)

    def relative_pose(self, translation, yaw_rotation):
        pos, orn = self._model.get_pose()
        _, _, yaw = transform.euler_from_quaternion(orn)
        #Calculate transformation matrices
        T_world_old = transform.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transform.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform.to_pose(T_world_new)
        self.absolute_pose(target_pos, self.endEffectorAngle)

    def close_gripper(self):
        self.gripper_close = True
        self._target_joint_pos = 0.05
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def open_gripper(self):
        self.gripper_close = False
        self._target_joint_pos = 0.0
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def _enforce_constraints(self, position):
        """Enforce constraints on the next robot movement."""
        if self._workspace:
            position = np.clip(position,
                               self._workspace['lower'],
                               self._workspace['upper'])
        return position
    
    def get_gripper_width(self):
        """Query the current opening width of the gripper."""
        left_finger_pos = 0.05 - self._left_finger.get_position()
        right_finger_pos = 0.05 - self._right_finger.get_position()

        return left_finger_pos + right_finger_pos

    def object_detected(self, tol=0.005):
        """Grasp detection by checking whether the fingers stalled while closing."""
        return self._target_joint_pos == 0.05 and self.get_gripper_width() > tol

    def get_pose(self):
        return self._model.get_pose()

    def camera_pose(self):
        return self._model.get_pose()

    def gripper_pose(self):
        return self._model.get_pose()

    def get_pose_cam(self):
        return self._model.get_pose()

    def is_simplified(self):
        return False
        
    def is_discrete(self):
        return self._actuator.is_discrete()