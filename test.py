# import gym
# import logging
# from env_sim.config_file.config import BaseEnvConfig
# from env_sim.envs.utils.robot import Robot
# from env_sim.envs.policy.policy_factory import policy_factory
#
# logging.basicConfig(level=logging.INFO)
# env = gym.make('EnvSim-v1')
# env.configure(BaseEnvConfig)
#
# robot = Robot(BaseEnvConfig, 'robot')
# env.set_robot(robot)
# robot.time_step = env.time_step
# policy = policy_factory['orca']()
# robot.set_policy(policy)
#
# ob = env.reset()
# rewards = []
# done = False
#
# for j in range(100):
#     action = robot.act(ob)
#     ob, _, done, info = env.step(action)
#     rewards.append(_)
#     if done:
#         env.render('video', None)  # render函数只负责绘制出一幅动图
#         print(info, '\n')
#         print(rewards)
#         env.close()
#         break


list1 = [1, 2, 3]
print(list1)
#list1[1] = 3
print(list1[1:])
