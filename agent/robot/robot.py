import os
import time
import numpy as np
import functools
import collections
from enum import Enum

import pybullet as p

import gym
from gym import spaces 

from agent.utils import io_utils
from agent.utils import transform_utils

from agent.robot import sensor, encoder, actuator
from agent.robot.rewards import CustomReward
from agent.world.world import World 

class RobotEnv(World):
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
        OUT_BOUND = 4

    def __init__(self, config, evaluate=False, test=False, validate=False):
        if not isinstance(config, dict):
            config = io_utils.load_yaml(config)
        self.config = config
        
        super().__init__(config, evaluate=evaluate, test=test, validate=validate)
        self.status = None
        self._step_time = collections.deque(maxlen=10000)
        self.history = []
        self.sr_mean = 0.
        self._workspace = {'lower': np.array([-1., -1., -1]),
                           'upper': np.array([1., 1., 1.])}

        # Model models
        self._body, self._mount, self._gripper = None, None, None
        self._robot_body_asset = config['robot']['body_path']
        self._robot_mount_asset = config['robot']['mount_path']
        self._robot_gripper_asset = config['robot']['gripper_path']

        self._robot_body_id = None
        self._robot_mount_id = None
        self._robot_gripper_id = None

        # Assign the actuators
        self._actuator = actuator.Actuator(config['policy'], self)
        self.action = None

        self._robot_body_joint_indices = []
        self._joint_epsilon = 1e-3

        self.gripper_close = False

        # Assign the sensors
        self._camera = sensor.RGBDSensor(config['sensor'], self)
        self._sensors = [self._camera]
        self.state = None

        # Assign the reward fn
        self._reward_fn = CustomReward(config['reward'], self)
        self.episode_step = 0
        self.episode_rewards = -5000

        # Assign callbacks
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],
                        RobotEnv.Events.END_OF_EPISODE: [],
                        RobotEnv.Events.CLOSE: [],
                        RobotEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)

        # Setup for spaces
        shape = self._camera.state_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], 4))
        self.action_space = gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32)


    ## Reset
    def reset(self):
        """Reset the task.

        Returns:
            Observation of the initial state.
        """
        self.trigger_event(RobotEnv.Events.START_OF_EPISODE)

        # enable torque control
        #
        if self.status == RobotEnv.Status.SUCCESS:
            self.history.append(1)
        else:
            self.history.append(0)

        self.episode_step = 0
        self.episode_rewards = 0
        self.status = RobotEnv.Status.RUNNING
        self.state = self._observe()

        return self.state

    def reset_model(self):
        # Robot body
        self.endEffectorAngle = 0.
        #start_pos = [0., 0., self._initial_height]
        self._body = self.add_model(
            self._robot_body_asset, [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._robot_body_id = self._body.model_id
        robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(
            p.getNumJoints(self._robot_body_id))]
        self._robot_body_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        
        # Robot mount  
        self._mount = self.add_model(
            self._robot_mount_asset, [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))
        self._robot_mount_id = self._mount.model_id

        # Robot gripper     
        self._gripper = self.add_model(
            self._robot_gripper_asset, [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))
        self._robot_gripper_id = self._gripper.model_id
        self._robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        
        p.createConstraint(self._robot_body_id, self._robot_end_effector_link_index, self._robot_gripper_id, 0, jointType=p.JOINT_FIXED, 
                        jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        return

    def reset_positions(self):
        robot_home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        p.setJointMotorControlArray(
            self._robot_body_id, self._robot_body_joint_indices,
            p.POSITION_CONTROL, robot_home_joint_config,
            positionGains=0.03 * np.ones(len(self._robot_body_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            # Keep moving until joints reach the target configuration
            current_joint_state = [
                p.getJointState(self._robot_body_id, i)[0]
                for i in self._robot_body_joint_indices]

            if all([np.abs(current_joint_state[i] - robot_home_joint_config[i]) < self._joint_epsilon
                    for i in range(len(self._robot_body_joint_indices))]):
                break

            if time.time()-timeout_t0 > 10:
                print(
                    "Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                p.setJointMotorControlArray(
                    self._robot_body_id, self._robot_body_joint_indices,
                    p.POSITION_CONTROL, robot_home_joint_config,
                    positionGains=np.ones(len(self._robot_body_joint_indices))
                )
                break
            self.step_sim(1)
        
    ## Step
    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns:
            A tuple (obs, reward, done, info), where done is a boolean flag
            indicating whether the current episode finished.
        """
        if self._body is None or self._mount is None or self._gripper is None:
            print(f"body {self._body}, mount {self._mount}, gripper {self._gripper}")
            self.reset()

        # action
        target = self.position_to_joints(action)
        if action[-1] > 0.5: self.open_gripper()
        else: self.close_gripper()
        self._act(target)

        # observe
        new_state = self._observe()
        #print(f"state: {new_obs}")

        # reward
        reward, self.status = self._reward_fn(self.state, action, new_state)
        self.episode_rewards += reward
        #print(f"reward: {reward}")

        # state update
        if self.status == RobotEnv.Status.SUCCESS: done = True
        elif self.episode_step == self._time_horizon - 1: done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else: done = False
        
        self.episode_step += 1
        self.state = new_state

        # return
        if done: self.trigger_event(RobotEnv.Events.END_OF_EPISODE, self)
        return self.state, reward, done, {"status": self.status, "episode_step": self.episode_step, "episode_rewards": self.episode_rewards}
        
    def step_sim(self, num_steps):
        """Advance the simulation by one step."""
        for _ in range(int(num_steps)):
            p.stepSimulation()
            if self._robot_gripper_id is not None:
                # Constraints
                gripper_joint_positions = np.array([p.getJointState(self._robot_gripper_id, i)[0] 
                                                    for i in range(p.getNumJoints(self._robot_gripper_id))])
                p.setJointMotorControlArray(
                    self._robot_gripper_id, 
                    [6, 3, 8, 5, 10], 
                    p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], 
                        -gripper_joint_positions[1], 
                        -gripper_joint_positions[1], 
                        gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )

    ## Observe
    def _observe(self):
        rgb, depth, mask = self._camera.get_state()

        observation = np.dstack((rgb, depth))
        return observation

    def gripper_pose(self):
        return self._gripper.get_pose()

    def camera_pose(self):
        pos, orn = self._gripper.get_pose()
        pos = tuple(map(sum, zip(pos, (0.00, 0.00, 0.25))))
        orn = tuple(map(sum, zip(orn, (0.00, 0.00, 0.00, 0.00))))
        return (pos, orn)

    def object_detected(self, tol=0.5):
        """Grasp detection by checking whether the fingers stalled while closing."""
        rgb, depth, mask = self._camera.get_state()
        return mask

    ## Action
    def _act(self, target, speed=0.03):
        assert len(self._robot_body_joint_indices) == len(target)

        p.setJointMotorControlArray(
            self._robot_body_id, self._robot_body_joint_indices,
            p.POSITION_CONTROL, target,
            positionGains=speed * np.ones(len(self._robot_body_joint_indices))
        )

        self.step_sim(1)
        return

    def open_gripper(self):
        self.gripper_close = False
        p.setJointMotorControl2(
            self._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=100)
        return

    def close_gripper(self):
        self.gripper_close = True
        p.setJointMotorControl2(
            self._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=100)
        return

    def position_to_joints(self, arr):
        pos, ori = arr[:3], arr[3]
        target_joint_state = p.calculateInverseKinematics(self._robot_body_id,
                                                          self._robot_end_effector_link_index,
                                                          pos, ori,
                                                          maxNumIterations=100, residualThreshold=1e-4)
        return target_joint_state

    ## Misc.
    def trigger_event(self, event, *event_args):
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))

    def register_events(self, evaluate, config):
        # Register callbacks
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self.reset_sim)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self.reset_objects)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self.reset_model)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self.reset_positions)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._actuator.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(RobotEnv.Events.CLOSE, super().close)