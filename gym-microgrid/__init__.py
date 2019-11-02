from gym.envs.registration import register

register(
    id='microgrid-v0',
    entry_point='gym_microgrid.envs:MicrogridEnv',
)
register(
    id='microgrid-extrahard-v0',
    entry_point='gym_microgrid.envs:MicrogridExtraHardEnv',
)
