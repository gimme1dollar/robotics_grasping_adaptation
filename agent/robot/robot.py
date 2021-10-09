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
from agent.utils import transformations
from agent.utils import transform_utils

from agent.robot import sensor, encoder, actuator
from agent.robot.rewards import CustomReward
from agent.simulation.simulation import World 

def _reset(robot, actuator, camera):
    ok = True
    while True:
        robot.reset_sim() #world + scene reset
        robot.reset_model() #robot model
        actuator.reset()

        # TODO: may reset when there exists no objects detected
        #_, _, mask = camera.get_state()
        #ok = len(np.unique(mask)) > 2  # plane and gripper are always visible
        
        if ok == True:
            break

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
        #self._simplified = config['simplified']

        # Model models
        self.model_path = config['robot']['model_path']        
        self._body, self._mount, self._gripper = None, None, None
        self._robot_body_id = None
        self._robot_mount_id = None
        self._robot_gripper_id = None

        # Assign the actuators
        self._actuator = actuator.Actuator(self, config)

        # Assign the sensors
        self._camera = sensor.RGBDSensor(config['sensor'], self)
        self._sensors = [self._camera]

        # Assign the reward fn
        self._reward_fn = CustomReward(config['reward'], self)

        # Assign callbacks
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],
                        RobotEnv.Events.END_OF_EPISODE: [],
                        RobotEnv.Events.CLOSE: [],
                        RobotEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)

        # Setup for spaces
        shape = self._camera.state_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], 5))
        self.action_space = self._actuator.action_space

    def _trigger_event(self, event, *event_args):
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))

    def register_events(self, evaluate, config):
        # Register callbacks
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, functools.partial(_reset, self, self._actuator, self._camera))
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(RobotEnv.Events.CLOSE, super().close)

    def reset(self):
        self._trigger_event(RobotEnv.Events.START_OF_EPISODE)

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
        """Reset the task.

        Returns:
            Observation of the initial state.
        """

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
            "assets/gripper/robotiq_2f_85.urdf", [0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))
        self._robot_gripper_id = self._gripper.model_id
        self._robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])
        p.createConstraint(self._robot_body_id, self._robot_end_effector_link_index, self._robot_gripper_id, 0, jointType=p.JOINT_FIXED, 
                        jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

    def step(self, action):
        #print("step")
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

        self._actuator.step(action)
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
        
        if done:
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)

        position, _ = self.get_pose()
        is_in_bound = "in" if (position[0] < 0.3) and (position[1] < 0.3) and (position[2] < 0.25) and (position[2] > 0) else "out"

        self.episode_step += 1
        self.state = new_state
        self.step_sim(1)

        #return self.obs, reward, done, {"is_success":self.status==RobotEnv.Status.SUCCESS, "episode_step": self.episode_step, "episode_rewards": self.episode_rewards, "status": self.status}
        return self.state, reward, done, {"status": self.status, "position": is_in_bound}
        
    def step_sim(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()

            # gripper constraint
            if self._robot_gripper_id is not None:
                gripper_joint_positions = np.array([p.getJointState(self._robot_gripper_id, i)[
                                                0] for i in range(p.getNumJoints(self._robot_gripper_id))])
                p.setJointMotorControlArray(
                    self._robot_gripper_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], -gripper_joint_positions[1], 
                        -gripper_joint_positions[1], gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )
            # time.sleep(1e-3)

    def get_pose(self):
        return self._gripper.get_pose()

    def _observe(self):
        rgb, depth, mask = self._camera.get_state()
        #print(f"mask : {mask}, sensing {len(np.unique(mask))} objects")

        sensor_pad = np.zeros(self._camera.state_space.shape[:2])
        sensor_pad[0][0] = self._actuator.get_state()
        obs_stacked = np.dstack((rgb, depth, sensor_pad))
        return obs_stacked

    def close_gripper(self):
        self.gripper_close = True
        p.setJointMotorControl2(
            self._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        
        self.step_sim(4e2)

    def open_gripper(self):
        self.gripper_close = False
        p.setJointMotorControl2(
            self._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        
        self.step_sim(4e2)
    
    def get_gripper_width(self):
        #print("get_gripper_width")
        """Query the current opening width of the gripper."""
        return p.getJointState(self._robot_gripper_id, 1)[0]

    def object_detected(self, tol=0.5):
        """Grasp detection by checking whether the fingers stalled while closing."""
        #print(f"gripper_close : {self.gripper_close}")
        #print(f"gripper_width() : {self.get_gripper_width()}")
        return self.get_gripper_width() > tol

    def render(self, mode='human'):
        pass

    '''
    def _enforce_constraints(self, position):
        """Enforce constraints on the next robot movement."""
        if self._workspace:
            position = np.clip(position,
                               self._workspace['lower'],
                               self._workspace['upper'])
        return position

    def is_simplified(self):
        return self._simplified

    def is_discrete(self):
        return self._actuator.is_discrete()

    def get_inverse(self, position, orientation, speed=0.03):
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
    '''