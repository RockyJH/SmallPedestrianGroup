import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
from crowd_nav.modules.explorer import Explorer
from crowd_nav.rgl_group_control import RglGroupControl
from crowd_nav.rvo_group_control import RvoGroupControl
from crowd_nav.tvcg_group_control import TvcgGroupControl
from env_sim.envs.modules.group import Group


def main(args):
    # 配置日志和运行设备
    level = logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    logging.info('[test.py]:使用的设备是： %s。', device)

    # policy_name = input('input a policy name \'rgl\' or \'rvo\' to compare--:')
    policy_name = 'tvcg'

    if policy_name == 'rgl':
        # 载入网络权重
        policy = RglGroupControl()
        project_path = os.getcwd()  # 项目路径
        data_dir = 'crowd_nav/data-3/rl_model_6.pth'
        model_weights = str(os.path.join(project_path, data_dir))
        policy.load_model(model_weights)
        logging.info('load model_weights from: ' + model_weights)

        # configure environment
        env = gym.make('EnvSim-v1')
        group = Group()
        group.policy = policy
        env.set_group(group)
        seed = input('input a random seed to generate the same random test scene --:')
        env.set_seed(seed)
        explorer = Explorer(env, group, device, None, gamma=0.9)
        epsilon_end = 0.1
        policy.set_epsilon(epsilon_end)
        policy.set_phase(args.phase)
        policy.set_device(device)
        policy.set_env(env)

        if args.visualize:
            rewards = []
            ob = env.reset()
            done = False
            step_count = 0
            while not done:
                action = group.get_action(ob)
                ob, _, done, info = env.step(action)
                step_count += 1
                rewards.append(_)
            gamma = 0.9
            cumulative_reward = sum([pow(gamma, t * group.time_step * group.v_pref)
                                     * reward for t, reward in enumerate(rewards)])

            logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f',
                         env.global_time, info, cumulative_reward)
            env.render('video')
            env.render('traj')

    elif policy_name == 'rvo':
        env = gym.make('EnvSim-v1')
        group = Group()
        policy = RvoGroupControl()
        group.policy = policy
        env.set_group(group)
        seed = input('input a random seed to generate the same random test scene --:')
        env.set_seed(seed)

        if args.visualize:
            rewards = []
            ob = env.reset()
            done = False
            step_count = 0
            while not done:
                action = group.get_action(ob)
                ob, _, done, info = env.step(action)
                step_count += 1
                rewards.append(_)

            logging.info('It takes %.2f seconds to finish. Final status is %s',
                         env.global_time, info)
            env.render('video')
            env.render('traj')
    elif policy_name == 'tvcg':
        env = gym.make('EnvSim-v1')
        group = Group()
        policy = TvcgGroupControl()
        group.policy = policy
        env.set_group(group)
        seed = input('input a random seed to generate the same random test scene --:')
        print('the seed used is : ',seed)
        ###################################################3
        env.set_seed(seed)

        if args.visualize:
            rewards = []
            ob = env.reset()
            done = False
            step_count = 0
            while not done:
                action = group.get_action(ob)
                ob, _, done, info = env.step(action)
                step_count += 1
                rewards.append(_)

            logging.info('It takes %.2f seconds to finish. Final status is %s',
                         env.global_time, info)
            env.render('video')
            env.render('traj')
    else:
        raise Exception("no such a policy!!!!!!!!!!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('-v', '--visualize', default=True, action='store_true')
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
