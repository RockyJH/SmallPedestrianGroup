import argparse  # 参数解析
import copy  # 用于深克隆
import logging  # 日志
import os  # 用于调用操作系统接口
import shutil  # 主要用于文件拷贝
import sys  # 操控运行时环境
import gym
import torch
import time
from tensorboardX import SummaryWriter  # 输出日志
from crowd_nav.rgl_group_control import RglGroupControl
from crowd_nav.modules.explorer import Explorer
from crowd_nav.modules.memory import ReplayMemory
from crowd_nav.modules.trainer import MPRLTrainer  #
from env_sim.envs.modules.group import Group


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    # step1-设置随机数种子
    set_random_seeds(17)

    # step2-创建输出文件
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('the out_put_dir is already exist, '
                    'do you want to override it?\n if you input \'no\' '
                    'we will creat a new file named with current time. (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            localtime = time.asctime( time.localtime(time.time()) )
            tim_sp = localtime.split(' ')
            t = tim_sp[1]+tim_sp[3]+'-'+tim_sp[4]

            args.output_dir = 'data-'+t
            make_new_dir = True

    if make_new_dir:
        os.makedirs(args.output_dir)

    log_file = os.path.join(args.output_dir, 'output.log')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    # step4-配置日志
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    # step5-device
    device = torch.device("cpu")
    logging.info('[train.py:] use device:%s', device)
    writer = SummaryWriter(log_dir=args.output_dir)  # writer 是为了输出日志文件

    # step6- 配置控制策略
    policy = RglGroupControl()
    policy.set_device(device)

    # step7-环境
    env = gym.make('EnvSim-v1')
    # step8-group
    group = Group()
    group.policy = policy
    group.time_step = env.time_step
    env.set_group(group)

    # step9-读取训练的参数
    rl_learning_rate = 0.001
    train_batches = 100
    train_episodes = 10000
    target_update_interval = 1000
    evaluation_interval = 1000
    capacity = 100000
    epsilon_start = 0.5
    epsilon_end = 0.1
    epsilon_decay = 4000
    checkpoint_interval = 1000

    # step10-配置trainer和explorer
    memory = ReplayMemory(capacity)  # 记忆库  memory 局部变量
    model = policy.get_model()  # model[graph_model + mlp]
    batch_size = 100  # 批训练的大小
    optimizer = 'Adam'  # 优化器：Adam
    trainer = MPRLTrainer(model, memory, device, policy, writer, batch_size, optimizer, env.out_group_agents_num)
    explorer = Explorer(env, group, device, writer, memory, policy.gamma, policy)

    # reinforcement learning
    policy.set_env(env)
    trainer.set_learning_rate(rl_learning_rate)

    # fill the memory pool with some RL experience
    group.policy.set_epsilon(epsilon_end)
    explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)  # 先run100个回合
    logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    ##############3
    trainer.update_target_model(policy.get_model())
    ##########
    # reinforcement learning
    episode = 0
    best_val_reward = -1
    best_val_model = None
    while episode < train_episodes:  # 10000  1万
        if episode < epsilon_decay:  # 前4000次，epsilon要衰减(0.1-0.4)/4000 * episode
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end  # 衰减到0.1之后保持不变
        group.policy.set_epsilon(epsilon)  # 每一步都修改epsilon

        # sample k episodes into memory and optimize over the generated memory
        # run了一个episode
        explorer.run_k_episodes(1, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)

        # trainer.optimize_batch(100,n)就是每次出来一个回合的结果就 从记忆里采样100个步骤学习100次一次。
        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:  # target_update_interval ==1000
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:  # evalution_interval == 1000
            _, _, _, reward, _ = explorer.run_k_episodes(100, 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
            # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(500, 'test', episode=episode, print_failure=True)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    sys_args = parser.parse_args()

    main(sys_args)
