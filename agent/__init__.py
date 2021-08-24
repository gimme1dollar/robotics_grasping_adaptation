from gym.envs.registration import register

register(
    id='gripper-env-v0',
    entry_point='agent.gripperEnv.robot:RobotEnv',
)