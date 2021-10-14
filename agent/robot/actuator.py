import numpy as np
import gym
import pybullet as p
from sklearn.preprocessing import MinMaxScaler


class Actuator:
    def __init__(self, config, robot):
        self.robot = robot
        self.gripper_close = False

        # Define action and state spaces
        self._max_translation = config['max_translation']
        self._max_yaw_rotation = config['max_yaw_rotation']
        self._max_force = config['max_force']

        self.action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)

    ## simulation
    def step(self, action):
        return self._act(action)

    def reset(self):
        # enable torque control
        for joint in self.robot._robot_joint_indices:
            p.setJointMotorControl2(self.robot._robot_body_id, joint, p.VELOCITY_CONTROL, force=0)
        return

    def _act(self, target, speed=0.03):
        #print(target)
        assert len(self.robot._robot_joint_indices) + 1 == len(target)
        
        # Body torque control
        p.setJointMotorControlArray(
            self.robot._robot_body_id, self.robot._robot_joint_indices,
            #p.TORQUE_CONTROL, target[:-1]
            p.POSITION_CONTROL, target[:-1]
        )

        # Gripper control
        if target[-1] > 0.5:
            self.open_gripper()
        else:
            self.close_gripper()

        # Gripper constraint
        gripper_joint_positions = np.array([p.getJointState(self.robot._robot_gripper_id, i)[0] 
                                                for i in range(p.getNumJoints(self.robot._robot_gripper_id))])
        p.setJointMotorControlArray(
            self.robot._robot_gripper_id, [3, 5, 6, 8, 10], p.POSITION_CONTROL,
            [
                -gripper_joint_positions[1],
                gripper_joint_positions[1],
                gripper_joint_positions[1],  
                -gripper_joint_positions[1], 
                gripper_joint_positions[1]
            ],
            positionGains=np.ones(5)
        )

        # Simulate
        self.robot.step_sim(1)
        return

    ## Status of actuators
    def get_state(self):
        '''Return gripper state [0, 1].'''

        # Body joint state
        current_joint_state = [
            p.getJointState(self.robot._robot_body_id, i)[0]
            for i in self.robot._robot_joint_indices
        ]

        # Gripper state
        current_gripper_state = self.get_gripper_width()
        
        current_joint_state.append(current_gripper_state)
        return current_joint_state

    ## Gripper
    def close_gripper(self):
        self.gripper_close = True
        p.setJointMotorControl2(self.robot._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=10, force=10)

        return

    def open_gripper(self):
        self.gripper_close = False
        p.setJointMotorControl2(self.robot._robot_gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=-10, force=10)

        return
    
    def get_gripper_width(self):
        #print("get_gripper_width")
        """Query the current opening width of the gripper."""
        return p.getJointState(self.robot._robot_gripper_id, 1)[0]
