from gym.envs.registration import register

register(
    id='gripper-env-v1',
    entry_point='agent.robot.robot:GripperEnv',
)