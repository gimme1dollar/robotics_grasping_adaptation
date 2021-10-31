from gym.envs.registration import register

register(
    id='robot-env-v0',
    entry_point='agent.robot.robot:ArmEnv',
)

register(
    id='gripper-env-v0',
    entry_point='agent.robot.robot:GripperEnv',
)