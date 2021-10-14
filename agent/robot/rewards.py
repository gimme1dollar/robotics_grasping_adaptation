import numpy as np
import pybullet as p

from agent.robot import robot

class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, config, robot):
        self._robot = robot

        self._terminal_reward = config['terminal_reward']
        self._detect_reward = config['detect_reward']
        self._grasp_reward = config['grasp_reward']
        self._lift_reward = config['lift_reward']

        self._time_penalty = config['time_penalty']
        self._out_penalty = config['out_penalty']
        self._close_penalty = config['close_penalty']
        
        # Placeholders
        self._grasping = False
        self._lifting = False

        # requisites
        self._start_height = None
        self._old_robot_height = None
        self._old_gripper_close = True

    def __call__(self, obs, action, new_obs):
        reward = 0.
        status = robot.RobotEnv.Status.RUNNING

        return reward, status

    def reset(self):
        position, _ = self._robot.robot_pose()
        self._grasping = False
        self._old_robot_height = position[2]

        return


class CustomReward(Reward):
    def __call__(self, obs, action, new_obs):
        reward = 0.

        # prerequisites
        self._target_id = self._robot.objects[0]
        object_pos, _ = p.getBasePositionAndOrientation(self._target_id)
        robot_pos, _ = self._robot.robot_pose()

        # Reward on detection
        det_mask = self._robot.object_detected()
        det_objects = np.unique(det_mask)
        det_target = self._target_id in det_objects
        reward += self._detect_reward * np.clip(len(det_objects), 0, 3)
        if det_target: 
            reward += self._detect_reward * 5 
            #print(f"reward on detection: {reward}")

        # Reward on grasping
        dist_object = np.linalg.norm(np.asarray(object_pos)-np.asarray(robot_pos))
        if dist_object < 0.07:
            self._grasping = True
            reward += self._grasp_reward
            #print(f"reward on grasping: {reward} by {dist_object}")
        else:
            self._grasping = False

        # Reward on lifting
        if self._grasping and robot_pos[2] > self._old_robot_height:
            self._lifting = True
            reward += self._lift_reward
            #print(f"reward on lifting: {reward} by {robot_pos[2] - self._old_robot_height}")
        else:
            self._lifting = False

        if self._lifting and robot_pos[2] > 0.1 and object_pos[2] > 0.1:
            #print(f"lifting up to {object_pos[2]}")
            #print(f"*** success, rewarding {reward} ***")
            return self._terminal_reward, robot.RobotEnv.Status.SUCCESS

        # Penalty on poor grasping
        if self._old_gripper_close == False and self._robot._actuator.gripper_close == True:
            reward -= self._close_penalty
            #print(f"penalty on poor grasping: {reward}")

        # Penalty on time
        reward -= self._time_penalty
        #print(f"penalty on time: {reward}")

        # return
        self._old_gripper_close = self._robot._actuator.gripper_close
        self._old_robot_height = robot_pos[2]
        return reward, robot.RobotEnv.Status.RUNNING