import sys  # 操控运行时环境
import logging  # 日志
import argparse  # 参数解析
import os  # 用于调用操作系统接口
import shutil  # 主要用于文件拷贝
import importlib.util  #
import torch
import gym
import copy  # 用于深克隆
import git  # 导入git 所谓何故？ repo = git.Repo(search_parent_directories=True)
import re
from tensorboardX import SummaryWriter  # 类似tensorboard
from crowd_nav.rgl_group_control import RglGroupControl
from crowd_sim.envs.utils.robot import Robot  # robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer  #
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from env_sim.envs.policy.policy_factory import policy_factory
from env_sim.envs.utils.group import Group


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


'''
'--policy' rgl
'--config' mp_separate.py 使用的是这个config
'--output_dir' 'data/output'
--overwrite' False
--weights'
--resume' False
--gpu' False
--debug' False
--test_after_every_eval' False
'--randomseed' 17
'''


def main(args):

    # step1-设置随机数种子
    set_random_seeds(args.randomseed)

    # step2-创建输出文件
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('输出文件的路径已经存在！是否删除重建？ (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False

    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    # step3-导入config文件
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # step4-配置日志
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    # step5-device
    device = torch.device("cpu")
    logging.info('CPU和GPU使用情况: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

    # step6- 配置控制策略
    policy = RglGroupControl
    policy.configure(policy_config)
    policy.set_device(device)

    # step7-环境
    env_config = config.EnvConfig(args.debug)
    env = gym.make('EnvSim-v1')
    env.configure(env_config)
    # step8-group
    group = Group(env_config, 'group')
    group.time_step = env.time_step
    env.set_group(group)

    # step9-读取训练的参数
    train_config = config.TrainConfig(args.debug)  # debuge == false会影响日志级别
    rl_learning_rate = train_config.train.rl_learning_rate  # 0.001 学习率
    train_batches = train_config.train.train_batches  # 100 批训练100
    train_episodes = train_config.train.train_episodes  # 10000步  训练总的回合数
    sample_episodes = train_config.train.sample_episodes  # 1
    target_update_interval = train_config.train.target_update_interval  # 1000
    evaluation_interval = train_config.train.evaluation_interval  # 1000
    capacity = train_config.train.capacity  # 100000 十万
    epsilon_start = train_config.train.epsilon_start  # 0.5
    epsilon_end = train_config.train.epsilon_end  # 0.1
    epsilon_decay = train_config.train.epsilon_decay  # 衰退 4000
    checkpoint_interval = train_config.train.checkpoint_interval  # 1000

    # step10-配置trainer和explorer
    memory = ReplayMemory(capacity)  # 记忆库
    model = policy.get_model()
    batch_size = train_config.trainer.batch_size  # 100
    optimizer = train_config.trainer.optimizer  # Adam
    if policy_config.name == 'model_predictive_rl':
        trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer,
                              env.human_num,
                              reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                              freeze_state_predictor=train_config.train.freeze_state_predictor,
                              detach_state_predictor=train_config.train.detach_state_predictor,
                              share_graph_model=policy_config.model_predictive_rl.share_graph_model)
    else:
        trainer = VNRLTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    explorer = Explorer(env, robot, device, writer, memory, policy.gamma, target_policy=policy)

    # imitation learning
    # 模仿学习
    if args.resume:  # args.resume ==False
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):  # 不存在， 这里因该是模仿学习一次给别的策略用
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    # 直接跳转到else执行：
    else:
        il_episodes = train_config.imitation_learning.il_episodes  # 2000个步骤
        il_policy = train_config.imitation_learning.il_policy  # orca
        il_epochs = train_config.imitation_learning.il_epochs  # 50回合
        il_learning_rate = train_config.imitation_learning.il_learning_rate  # 0.001
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible:  # Fasle
            safety_space = 0
        else:
            safety_space = train_config.imitation_learning.safety_space  # 0.15
        il_policy = policy_factory[il_policy]()  # orca
        il_policy.multiagent_training = policy.multiagent_training  # ORCA True
        il_policy.safety_space = safety_space  # 0.15
        robot.set_policy(il_policy)

#### 关键是跳转到这里， explorer.run_k_episodes # 2000
#### trainer.optimize_epoch(il_epochs)
#### 然后把模仿学习的模型保存起来
#### trainer 更新model
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, leaimitation_rning=True)
        trainer.optimize_epoch(il_epochs)
        policy.save_model(il_weight_file)  # 把model保存起来了
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    trainer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)  # 此处的policy是model_predictive_RL
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)

    # fill the memory pool with some RL experience
    if args.resume:  # False
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    best_val_reward = -1
    best_val_model = None

    # evaluate the model after imitation learning

    if episode % evaluation_interval == 0:  # 1000步整一下
        logging.info('Evaluate the model instantly after imitation learning on the validation cases')
        explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        explorer.log('val', episode // evaluation_interval)

        if args.test_after_every_eval:  # False
            explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
            explorer.log('test', episode // evaluation_interval)

    episode = 0
    while episode < train_episodes:  # 10000  1万
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:  # 前5000次，epsilon要衰减
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end  # 衰减到0.1之后后5000次保持不变
        robot.policy.set_epsilon(epsilon)  # 每一步都修改epsilon

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)

        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:  # target_update_interval ==1000
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:  # evalution_interval == 1000
            _, _, _, reward, _ = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
            # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode, print_failure=True)
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
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/mp_separate.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)

    sys_args = parser.parse_args()

    main(sys_args)
