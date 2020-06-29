import logging
import torch
from tqdm import tqdm  # Tqdm 是一个快速，可扩展的Python进度条
from env_sim.envs.modules.info import *

class Explorer(object):
    def __init__(self, env, group, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.group = group
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None
        self.info = None

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.group.policy.set_phase(phase)
        success_times = []  # 成功的次数
        collision_times = []  # 碰撞的次数
        timeout_times = []  # 超时次数
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []  # 累积的reward
        average_returns = []  # 平均 returns
        collision_cases = []  # 碰撞的案例
        timeout_cases = []  # 超时的案例

        if k != 1:
            pbar = tqdm(total=k) # 进度条
        else:
            pbar = None

        for i in range(k):
            ob = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.group.get_action(ob)
                ob, reward, done, self.info = self.env.step(action)
                states.append(self.group.policy.last_state)
                actions.append(action)
                rewards.append(reward)

            if isinstance(self.info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(self.info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(self.info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            # 向记忆里存的时候是给一个回合的信息，但在update_memory的时候是将这些记忆分解成一个一的动作存到记忆里
            if update_memory:
                if isinstance(self.info, ReachGoal) or isinstance(self.info, Collision) or isinstance(self.info,Timeout):
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.group.time_step * self.group.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.group.time_step * self.group.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'.format(phase.upper(), extra_info, success_rate, collision_rate,
                                                      avg_nav_time, average(cumulative_rewards),
                                                      average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))

        if print_failure: # false
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(
            average_returns)

        return self.statistics

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        # 给到即记忆里的是一个回合的信息，将他们拆解成一系列的动作存到记忆里。
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i + 1])
                value = sum([pow(self.gamma, (t - i) * self.group.time_step * self.group.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i + 1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'RglGroupControl':
                self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
