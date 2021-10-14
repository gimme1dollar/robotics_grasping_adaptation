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
        
        super().__init__(config, evaluate=evaluate, test=test, validate=validate)
        self.status = None
        self._step_time = collections.deque(maxlen=10000)
        self.history = []
        self.sr_mean = 0.
        self._workspace = {'lower': np.array([-1., -1., -1]),
                           'upper': np.array([1., 1., 1.])}

        # Model models
        self.model_path = config['robot']['model_path']        
        self._body, self._mount, self._gripper = None, None, None
        self._robot_body_id = None
        self._robot_mount_id = None
        self._robot_gripper_id = None

        # Assign the actuators
        self._actuator = actuator.Actuator(config['robot'], self)
        self.action = None

        # Assign the sensors
        self._camera = sensor.RGBDSensor(config['sensor'], self)
        self._sensors = [self._camera]
        self.state = None

        # Assign the reward fn
        self._reward_fn = CustomReward(config['reward'], self)
        self.episode_step = 0
        self.episode_rewards = 0

        # Assign callbacks
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],
                        RobotEnv.Events.END_OF_EPISODE: [],
                        RobotEnv.Events.CLOSE: [],
                        RobotEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)

        # Setup for spaces
        shape = self._camera.state_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], 6))
        self.action_space = self._actuator.action_space


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

        return self.state

    def reset_model(self):
        # Robot body
        self.endEffectorAngle = 0.
        #start_pos = [0., 0., self._initial_height]
        self._body = self.add_model(
            "assets/body/ur5.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._robot_body_id = self._body.model_id
        robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(
            p.getNumJoints(self._robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        
        # Robot mount  
        self._mount = self.add_model(
            "assets/body/mount.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))
        self._robot_mount_id = self._mount.model_id

        # Robot gripper     
        self._gripper = self.add_model(
            "assets/gripper/robotiq_2f_85.urdf", [0, 0, 1], p.getQuaternionFromEuler([np.pi, 0, 0]))
        self._robot_gripper_id = self._gripper.model_id
        self._robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        p.createConstraint(self._robot_body_id, self._robot_end_effector_link_index, self._robot_gripper_id, 0, jointType=p.JOINT_FIXED, 
                        jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        return

    def reset_positions(self):
        for _ in range(100):
            self._actuator._act([-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0, 1.0])
        print("reset positions")

        return
        
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

        self._act(action)
        self.step_sim(1)
        #print(f"action: {action}")

        new_state = self._observe()
        #print(f"state: {new_obs}")

        reward, self.status = self._reward_fn(self.state, action, new_state)
        #print(f"reward: {reward}")
        self.episode_rewards += reward

        if self.status == RobotEnv.Status.SUCCESS:
            done = True
        elif self.episode_step == self._time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False
        
        self.episode_step += 1
        self.state = new_state

        if done:
            self.trigger_event(RobotEnv.Events.END_OF_EPISODE, self)
        return self.state, reward, done, {"status": self.status, "episode_step": self.episode_step, "episode_rewards": self.episode_rewards}
        

    ## Observe
    def _observe(self):
        rgb, depth, mask = self._camera.get_state()

        # TODO: better joints matrix
        joints = self._actuator.get_state()

        observation = np.dstack((rgb, depth, mask))
        return observation

    def robot_pose(self):
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
    def _act(self, target):
        self._actuator._act(target)

    # Manual control
    def position_to_joints(self, position, orientation, speed=0.03):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        target_joint_state = np.zeros((6,))  # this should contain appropriate joint angle values
        # ========= Part 1 ========
        # Using inverse kinematics (p.calculateInverseKinematics), find out the target joint configuration of the robot
        # in order to reach the desired end_effector position and orientation
        # HINT: p.calculateInverseKinematics takes in the end effector **link index** and not the **joint index**. You can use 
        #   self.robot_end_effector_link_index for this 
        # HINT: You might want to tune optional parameters of p.calculateInverseKinematics for better performance
        # ===============================
        target_joint_state = p.calculateInverseKinematics(self._robot_body_id,
                                                          self._robot_end_effector_link_index,
                                                          position, orientation,
                                                          maxNumIterations=100, residualThreshold=1e-4)
        return target_joint_state

    def manual_control(self, grasp_position, grasp_angle):
        print(f"excute_grasp : {grasp_position}")
        """
            Execute grasp sequence
            @param: grasp_position: 3d position of place where the gripper jaws will be closed
            @param: grasp_angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        gripper_orientation = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        
        # move body
        print(f"moving init")
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        for _ in range(10): self.step(np.append(self.position_to_joints(pre_grasp_position_over_bin, gripper_orientation), [1.0]))

        print("over object") 
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.1])
        for _ in range(10): self.step(np.append(self.position_to_joints(pre_grasp_position_over_object, gripper_orientation), [1.0]))

        print(f"close gripper")
        joints = self._actuator.get_state()
        joints[-1] = 0.0
        for _ in range(10): self.step(joints)

        print("grasping end")        
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        for _ in range(10): self.step(np.append(self.position_to_joints(post_grasp_position, None), [0.0]))

        print()   
        return

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
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._actuator.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self.reset_positions)
        self.register_callback(RobotEnv.Events.CLOSE, super().close)