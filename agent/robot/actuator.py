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

        self.action_space = gym.spaces.Box(-1., 1., shape=(6,), dtype=np.float32)

    def reset(self):
        self.robot.open_gripper()

    def step(self, action):
        return self._act(action)

    def get_state(self):
        '''Return gripper state [0, 1].'''

        # Body joint state
        current_joint_state = [
            p.getJointState(self.robot._robot_body_id, i)[0]
            for i in self.robot._robot_joint_indices
        ]
        #print("current_joint_state: ", end=" ")
        #for i in current_joint_state : print(f"{i:0.2}", end=" ") 
        #print()

        # Gripper state
        self.robot.get_gripper_width()

        return self.robot.get_gripper_width()

    #def _clip_translation_vector(self, translation, yaw):
    #    """Clip the vector if its norm exceeds the maximal allowed length."""
    #    length = np.linalg.norm(translation)
    #    if length > self._max_translation:
    #        translation *= self._max_translation / length
    #    if yaw > self._max_yaw_rotation:
    #        yaw *= self._max_yaw_rotation / yaw
    #    return translation, yaw

    def _act(self, target, speed=0.03):
        assert len(self.robot._robot_joint_indices) == len(target)
        
        # Body torque control
        p.setJointMotorControlArray(
            self.robot._robot_body_id, self.robot._robot_joint_indices,
            p.TORQUE_CONTROL, target
        )

        self.robot.step_sim(1)

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

    #def is_discrete(self):
    #    return self._discrete
