from gym.envs.registration import register

register(
    id='EnvSim-v1',
    entry_point='env_sim.envs:EnvSim',
)
'''
openAI gym接口必须的步骤， 将环境注册到gym 中，这样以后在使用的时候直接
env = gym.make('env_name-v0') # 参数为环境的id
'''