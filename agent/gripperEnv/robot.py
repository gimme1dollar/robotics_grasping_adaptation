import os
import time
import numpy as np
import functools
import collections
from enum import Enum

import pybullet as p

import gym
from gym import spaces 

from agent.common import io_utils
from agent.common import transformations
from agent.common import transform_utils

from agent.gripperEnv import sensor, encoder, actuator
from agent.simulation.simulation import World 
from agent.gripperEnv.rewards import CustomReward

def _reset(robot, actuator, camera, skip_empty_states=False):
    ok = True
    while True:
        robot.reset_sim() #world + scene reset
        robot.reset_model() #robot model
        actuator.reset()
        #_, _, mask = camera.get_state()
        #ok = len(np.unique(mask)) > 2  # plane and gripper are always visible

        if not skip_empty_states:
            ok = True
        
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
        self.time_horizon = config['time_horizon']
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
        self.action_space = self._actuator.setup_action_space()
       
        shape = self._camera.state_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], 5))

    def _trigger_event(self, event, *event_args):
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))

    def register_events(self, evaluate, config):
        # Setup the reset function
        skip_empty_states = True if evaluate else config['skip_empty_initial_state']
        reset = functools.partial(_reset, self, self._actuator, self._camera,
                                skip_empty_states)

        # Register callbacks
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(RobotEnv.Events.CLOSE, super().close)

    def reset(self):
        #print("reset")
        self._trigger_event(RobotEnv.Events.START_OF_EPISODE)

        if self.status == RobotEnv.Status.SUCCESS:
            self.history.append(1)
        else:
            self.history.append(0)

        self.episode_step = 0
        self.episode_rewards = 0
        self.status = RobotEnv.Status.RUNNING
        self.obs = self._observe()

        #print("reset_out")
        return self.obs

    def reset_model(self):
        #print("reset_model")
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
        print(f"gripper id : {self._robot_gripper_id}")
        self._robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])
        print(f"body idx : {self._robot_body_id}")
        p.createConstraint(self._robot_body_id, self._robot_end_effector_link_index, self._robot_gripper_id, 0, jointType=p.JOINT_FIXED, 
                        jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        #self._left_finger = self._model.joints[self._left_finger_id]
        #self._right_finger = self._model.joints[self._right_finger_id]

        #print("reset_model out")

    def step(self, action):
        print("step")
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

        new_obs = self._observe()

        reward, self.status = self._reward_fn(self.obs, action, new_obs)
        print(f"reward: {reward}")
        self.episode_rewards += reward

        if self.status == RobotEnv.Status.SUCCESS:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False
        
        if done:
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)

        position, _ = self.get_pose()
        is_in_bound = "in" if (position[0] < 0.3) and (position[1] < 0.3) and (position[2] < 0.25) and (position[2] > 0) else "out"

        self.episode_step += 1
        self.obs = new_obs
        self.step_sim(1)
        #return self.obs, reward, done, {"is_success":self.status==RobotEnv.Status.SUCCESS, "episode_step": self.episode_step, "episode_rewards": self.episode_rewards, "status": self.status}
        return self.obs, reward, done, {"status": self.status, "position": is_in_bound}
        
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
        print(f"mask : {mask}, sensing {len(np.unique(mask))} objects")

        sensor_pad = np.zeros(self._camera.state_space.shape[:2])
        sensor_pad[0][0] = self._actuator.get_state()
        obs_stacked = np.dstack((rgb, depth, sensor_pad))
        return obs_stacked

    '''
    def reset_robot_pose(self, target_pos, target_orn):
        """ Reset the world coordination of the robot base. Useful for test purposes """
        self.reset_base(self._model.model_id, target_pos, target_orn)
        self.run(0.1)

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
        _, _, yaw = transform_utils.euler_from_quaternion(orn)
        #Calculate transformation matrices
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        self.absolute_pose(target_pos, self.endEffectorAngle)
    '''

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

    def object_detected(self, tol=0.834 - 0.001):
        """Grasp detection by checking whether the fingers stalled while closing."""
        return self.get_gripper_width() < tol

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
    '''

    '''
    def execute_grasp(self, grasp_position, grasp_angle):
        """
            Execute grasp sequence
            @param: grasp_position: 3d position of place where the gripper jaws will be closed
            @param: grasp_angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        # Adjust grasp_position to account for end-effector length
        print(f"execute_grasp : {grasp_position}")
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        gripper_orientation = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.1])
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        
        # ========= PART 2============
        # Implement the following grasp sequence:
        # 1. open gripper
        # 2. Move gripper to pre_grasp_position_over_bin
        # 3. Move gripper to pre_grasp_position_over_object
        # 4. Move gripper to grasp_position
        # 5. Close gripper
        # 6. Move gripper to post_grasp_position
        # 7. Move robot to robot_home_joint_config
        # 8. Detect whether or not the object was grasped and return grasp_success
        # ============================
        self.open_gripper()
        self.move_tool(pre_grasp_position_over_bin, None)
        self.move_tool(pre_grasp_position_over_object, gripper_orientation)
        self.move_tool(grasp_position, gripper_orientation)
        self.close_gripper()
        self.move_tool(post_grasp_position, None)
        #self.robot_go_home(speed=0.03)
        #grasp_success = self.check_grasp_success()
        #return grasp_success
        return
    
    def move_tool(self, position, orientation, speed=0.03):
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
        self._actuator.step(target_joint_state)
    '''

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