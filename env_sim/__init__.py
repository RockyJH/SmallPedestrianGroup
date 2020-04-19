from gym.envs.registration import register

register(
    id='EnvSim-v1',
    entry_point='env_sim.envs:EnvSim',
)
