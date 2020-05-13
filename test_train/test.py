import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.modules.explorer import Explorer
from crowd_nav.rgl_group_control import RglGroupControl
from env_sim.envs.modules.group import Group


def main(args):
    # 配置日志和运行设备
    level = logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    logging.info('[test.py]:使用的设备是： %s。', device)

    if args.model_dir is not None:
        model_weights = os.path.join(args.model_dir, 'best_val.pth')  # 载入网络的比重
        logging.info('从 ' + model_weights + '载入net weight')
        policy = RglGroupControl()
        policy.load_model(model_weights)
    else:
        ##############注意此处的设置，仅仅为了本次测试。#####
        policy = RglGroupControl()
        model_weights = 'best_val.pth'
        logging.info('load model from:{}'.format(model_weights))
        policy.load_model(model_weights)

    # configure environment
    env = gym.make('EnvSim-v1')
    group = Group()
    group.policy = policy
    env.set_group(group)
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
        last_pos = np.array((group.cx,group.cy))
        step_count = 0
        while not done:
            action = group.get_action(ob)
            ob, _, done, info = env.step(action)
            step_count += 1
            rewards.append(_)
            current_pos = np.array((group.cx,group.cy))
            last_pos = current_pos
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * group.time_step * group.v_pref)
                                 * reward for t, reward in enumerate(rewards)])

        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time,
                     info, cumulative_reward)
        env.render('video')
    else:
        explorer.run_k_episodes(100, args.phase, print_failure=True)
        if args.plot_test_scenarios_hist:  # True
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('-v', '--visualize', default=True, action='store_true')
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
