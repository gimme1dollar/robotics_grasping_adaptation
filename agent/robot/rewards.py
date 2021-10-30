import numpy as np
import pybullet as p

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
        position, _ = self._robot.gripper_pose()
        self._grasping = False
        self._old_robot_height = position[2]

        return


class CustomReward(Reward):
    def __call__(self, obs, action, new_obs):
        reward, status = 0., self._robot.RobotEnv.Status.RUNNING

        # prerequisites
        self._target_id = self._robot.objects[0]
        object_pos, _ = p.getBasePositionAndOrientation(self._target_id)
        robot_pos, _ = self._robot.gripper_pose()

        # Reward on detection
        det_mask = self._robot.object_detected()
        det_objects = np.unique(det_mask)
        det_target = self._target_id in det_objects
        reward += self._detect_reward * np.clip(len(det_objects), 0, 3)
        if det_target: 
            reward += self._detect_reward * 5 
            status = self._robot.RobotEnv.Status.DETECT
            #print(f"reward on detection: {reward}")

        # Reward on grasping
        dist_object = np.linalg.norm(np.asarray(object_pos)-np.asarray(robot_pos))
        if dist_object < 0.07:
            self._grasping = True
            reward += self._grasp_reward
            status = self._robot.RobotEnv.Status.GRASP
            #print(f"reward on grasping: {reward} by {dist_object}")
        else:
            self._grasping = False

        # Reward on lifting
        if self._grasping and robot_pos[2] > self._old_robot_height:
            self._lifting = True
            reward += self._lift_reward
            status = self._robot.RobotEnv.Status.LIFT
            #print(f"reward on lifting: {reward} by {robot_pos[2] - self._old_robot_height}")
        else:
            self._lifting = False

        if self._lifting and robot_pos[2] > 0.1 and object_pos[2] > 0.1:
            #print(f"lifting up to {object_pos[2]}")
            #print(f"*** success, rewarding {reward} ***")
            return self._terminal_reward, self._robot.RobotEnv.Status.SUCCESS

        # Penalty on poor grasping
        if self._old_gripper_close == False and self._robot.gripper_close == True:
            reward -= self._close_penalty
            #print(f"penalty on poor grasping: {reward}")

        # Penalty on time
        reward -= self._time_penalty
        #print(f"penalty on time: {reward}")

        # return
        self._old_gripper_close = self._robot.gripper_close
        self._old_robot_height = robot_pos[2]
        return reward, status

        
class GripperReward(Reward):
    def __call__(self, obs, action, new_obs):
        reward, status = 0., self._robot.Status.RUNNING
        
        position, _ = self._robot.get_pose()
        robot_height = position[2]
        
        # Range out of bound penalty
        if (position[0] > 0.3) or (position[1] > 0.3):
            reward -= self._out_penalty        
            if (position[2] > 0.25) or (position[2] < 0):
                reward -= self._out_penalty * 3
        else:
            status = self._robot.Status.IN_BOX

        if self._robot.object_detected():
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True
                status = self._robot.Status.GRASP

            if robot_height - self._start_height > 0.15:
                return self._terminal_reward, self._robot.Status.SUCCESS

                '''    
                if self._table_clearing:
                    # Object was lifted by the desired amount
                    grabbed_obj = self._robot.find_highest()
                    if grabbed_obj is not -1:
                        self._robot.remove_model(grabbed_obj)
                    
                    # Multiple object grasping
                    # grabbed_objs = self._robot.find_higher(self.lift_dist)
                    # if grabbed_objs:
                    #     self._robot.remove_models(grabbed_objs)

                    self._robot.open_gripper()
                    if self._robot.get_num_body() == 2: 
                        return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                    return self._lift_success, robot.RobotEnv.Status.RUNNING
                else:
                    if not self._shaped:
                        return 1., robot.RobotEnv.Status.SUCCESS
                    return self._terminal_reward, robot.RobotEnv.Status.SUCCESS
                '''

            # Intermediate rewards for grasping
            reward += self._grasp_reward

            # Intermediate rewards for lifting
            delta_z = robot_height - self._old_robot_height
            reward += 10 * delta_z
        else:
            self._lifting = False


        # Time penalty
        reward -= self._time_penalty
        
        # Poor grasp
        if self._old_gripper_close ^ self._robot.gripper_close:
            reward -= self._close_penalty

        self._old_gripper_close = self._robot.gripper_close
        self._old_robot_height = robot_height
        return reward, status