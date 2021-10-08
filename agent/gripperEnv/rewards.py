from agent.gripperEnv import robot

class Reward:
    """Simple reward function reinforcing upwards movement of grasped objects."""

    def __init__(self, config, robot):
        self._robot = robot
        self._shaped = config.get('shaped', True)

        self._max_delta_z = robot._actuator._max_translation
        self._terminal_reward = config['terminal_reward']
        self._grasp_reward = config['grasp_reward']
        self._delta_z_scale = config['delta_z_scale']
        self._lift_success = config.get('lift_success', self._terminal_reward)
        self._time_penalty = config['time_penalty']
        self._out_penalty = config['out_penalty']
        self._close_penalty = config['close_penalty']
        self.lift_dist = 0

        # Placeholders
        self._lifting = False
        self._start_height = None
        self._old_robot_height = None
        self._old_gripper_close = True

    def __call__(self, obs, action, new_obs):
        reward = 0.
        status = robot.RobotEnv.Status.RUNNING

        position, _ = self._robot.get_pose()
        robot_height = position[2]
        
        if self._robot.object_detected():
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:
                status = robot.RobotEnv.Status.SUCCESS
                return self._terminal_reward, status
        else:
            self._lifting = False

        return reward, status

    def reset(self):
        position, _ = self._robot.get_pose()
        self._old_robot_height = position[2]


class CustomReward(Reward):
    def __call__(self, obs, action, new_obs):
        reward = 0.
        
        position, _ = self._robot.get_pose()
        robot_height = position[2]

        if self._robot.object_detected():
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True

            if robot_height - self._start_height > self.lift_dist:
                return self._terminal_reward, robot.RobotEnv.Status.SUCCESS

            # Intermediate rewards for lifting
            reward += self._grasp_reward

            delta_z = robot_height - self._old_robot_height
            reward += self._delta_z_scale * delta_z
        else:
            self._lifting = False

        # Time penalty
        reward -= self._time_penalty

        # Range out of bound penalty
        if (position[0] > 0.3) or (position[1] > 0.3):
            reward -= self._out_penalty
        if (position[2] > 0.25) or (position[2] < 0):
            reward -= self._out_penalty * 3

        # Poor grasp
        if self._old_gripper_close == False and self._robot.gripper_close == True:
            reward -= self._close_penalty

        self._old_gripper_close = self._robot.gripper_close
        self._old_robot_height = robot_height
        return reward, robot.RobotEnv.Status.RUNNING