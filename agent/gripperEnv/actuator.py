import numpy as np
import gym
import pybullet as p
from sklearn.preprocessing import MinMaxScaler


class Actuator:
    def __init__(self, robot, config):
        self.robot = robot
        #self._include_robot_height = config.get('include_robot_height', False)
        #self._simplified = simplified

        # Define action and state spaces
        self._max_translation = config['robot']['max_translation']
        self._max_yaw_rotation = config['robot']['max_yaw_rotation']
        self._max_force = config['robot']['max_force']

        # Discrete action step sizes
        #self._discrete = config['robot']['discrete']
        #self._discrete_step = config['robot']['step_size']
        #self._yaw_step = config['robot']['yaw_step']
        #if self._discrete:
        #    self.num_actions_pad = config['robot']['num_actions_pad']
        #    self.num_act_grains = self.num_actions_pad - 1
        #    self.trans_action_range = 2 * self._max_translation
        #    self.yaw_action_range = 2 * self._max_yaw_rotation

        # Last gripper action
        self._gripper_open = True
        self.state_space = None

    def setup_action_space(self):
        print("actuator setup_action_space")
        #high = np.r_[[self._max_translation] * 4, self._max_yaw_rotation, 1.]
        #self._action_scaler = MinMaxScaler((-1, 1))
        #self._action_scaler.fit(np.vstack((-1. * high, high)))
        self.action_space = gym.spaces.Box(-1., 1., shape=(6,), dtype=np.float32)

        print("actuator setup_action_space out")
        return self.action_space

    def reset(self):
        self.robot.open_gripper()
        self._gripper_open = True

    def step(self, action):
        return self._act(action)

    def get_state(self):
    #    """Return the current opening width scaled to a range of [0, 1]."""
    #    if self._include_robot_height:
    #        gripper_width = self.robot.get_gripper_width()
    #        position, _ = self.robot.get_pose()
    #        height = position[2]
    #        state = self._obs_scaler * np.r_[gripper_width, height]
    #    else:
    #        state = self._obs_sclaer * self.robot.get_gripper_width()
        return self.robot.get_gripper_width()

    #def _clip_translation_vector(self, translation, yaw):
    #    """Clip the vector if its norm exceeds the maximal allowed length."""
    #    length = np.linalg.norm(translation)
    #    if length > self._max_translation:
    #        translation *= self._max_translation / length
    #    if yaw > self._max_yaw_rotation:
    #        yaw *= self._max_yaw_rotation / yaw
    #    return translation, yaw

    def _act(self, target_joint_state, speed=0.03):
        print("actuator _full_act")
        print(f"target_joint_state : {target_joint_state}")

        assert len(self.robot._robot_joint_indices) == len(target_joint_state)
        
        p.setJointMotorControlArray(
            self.robot._robot_body_id, self.robot._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self.robot._robot_joint_indices))
        )

        # Keep moving until joints reach the target configuration
        current_joint_state = [
            p.getJointState(self.robot._robot_body_id, i)[0]
            for i in self.robot._robot_joint_indices
        ]
        print(f"current_joint_state : {current_joint_state}")
        self.robot.step_sim(1)

        print("actuator _full_act out")

    #def is_discrete(self):
    #    return self._discrete
