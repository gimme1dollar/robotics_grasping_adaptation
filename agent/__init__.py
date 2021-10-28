from gym.envs.registration import register

register(
    id='grasping-env-v0',
    entry_point='agent.robot.robot:RobotEnv',
)

register(
    id='gripper-env-v0',
    entry_point='agent.robot.gripperEnv:GripperEnv',
)