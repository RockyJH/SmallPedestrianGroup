import logging
import torch
import numpy as np
import numpy.linalg

import math

from numpy.linalg import norm
import itertools
from env_sim.envs.policy.policy import Policy
from env_sim.envs.utils.action import ActionRot, ActionXY
from env_sim.envs.utils.state import tensor_to_joint_state
from env_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.utils.state_predictor import LinearStatePredictor
from crowd_nav.utils.graph_model import RGL
from crowd_nav.utils.value_estimator import ValueEstimator
from env_sim.envs.utils.agent import Agent


class RglGroupControl(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TieQinrui'
        self.trainable = True
        self.multiagent_training = True
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.agent_radius = 0.3

    def configure(self, config):
        self.state_predictor = LinearStatePredictor(config, self.time_step)
        graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
        self.value_estimator = ValueEstimator(config, graph_model)
        self.model = [graph_model, self.value_estimator.value_network]
        # value_network 是后面的哪个多层感知机

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        return {
            'graph_model': self.value_estimator.graph_model.state_dict(),
            'value_network': self.value_estimator.value_network.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
        self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    # 遍历动作空间--计算执行该动作能获得的value，选取value最大的动作
    def predict(self, state, action_space, group_members):
        action_space = action_space
        group_members = group_members

        # 如果可能性小于贪婪度则随机选取动作
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')

            # 遍历动作空间--计算执行该动作能获得的value，选取value最大的动作
            for action in action_space:

                # 状态估计：此处状态估计是为小组做的，故组内成员不用管，对agent线性预测，对group直接放到目标位置。
                next_state = self.state_predictor(state, action)
                # 此处用上一步得到的状态估计，获得value
                value_return = self.value_estimate(next_state)
                # 由碰撞检测获得reward_est,这个工作由 self.estimate_reward函数来负责
                reward_est = self.estimate_reward(state, action, group_members)

                value = reward_est + self.get_normalized_gamma() * value_return
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        # 训练时，执行到此，选出了动作，那么当前的state就成了last_state
        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    # 对于一个状态估值
    def value_estimate(self, state):
        value = self.value_estimator(state)
        return value

    # 输入（状态，动作），线性推演碰撞检测，返回reward
    def estimate_reward(self, state, action, group_members):
        ##########################################################################################
        #### form的参照点外插, 这里暂时不做这个实现，后期可在选择实现，当前的实现是直接按照当前速度走下一步即可。###
        ##########################################################################################

        agents_states = state.agents_state
        group_state = state.group_state

        # 由group的action（v，formation）获得每个group_memeber的action（一个速度）
        vx, vy = action[0]
        formation = action[1]

        p = formation.get_ref_point()

        p = p[0] + vx * self.time_step, p[1] * vy * self.time_step
        relation = formation.get_relation()  # [[a,b],[a,b],[a,b]]
        vn, vc = formation.get_vn_vc()  # vn = (x,y) vc = (x,y)

        # 获取新的位置,np_array的形式
        new_p1 = np.array(p) + relation[0][0] * np.array(vn) + relation[0][1] * np.array(vc)
        new_p2 = np.array(p) + relation[1][0] * np.array(vn) + relation[1][1] * np.array(vc)
        new_p3 = np.array(p) + relation[2][0] * np.array(vn) + relation[2][1] * np.array(vc)
        new_p_array = [new_p1, new_p2, new_p3]  # list里面是数组类型的新位置

        # 一对一地分配谁去哪里
        member1, member2, member3 = group_members

        old_p1 = np.array(member1.get_position())
        old_p2 = np.array(member2.get_position())
        old_p3 = np.array(member3.get_position())

        old_p_arr = [old_p1, old_p2, old_p3]

        all_premutations = [[1, 2, 3], [1, 3, 2],
                            [2, 1, 3], [2, 3, 2],
                            [3, 1, 2], [3, 2, 1]]
        min_dis_index = 0
        min_dis = float('inf')

        for i, premutation in enumerate(all_premutations):
            dis = np.linalg.norm(new_p_array[premutation[0]] - old_p_arr[0] +
                                 new_p_array[premutation[1]] - old_p_arr[1] +
                                 new_p_array[premutation[2]] - old_p_arr[2])
            if dis < min_dis:
                min_dis = dis
                min_dis_index = i

        final_premutation = all_premutations[min_dis_index]
        # 即 1，2，3个member 分别去 final_premutaion 指示的地方

        three_velocity = [new_p_array[final_premutation[0]] - old_p_arr[0],
                          new_p_array[final_premutation[1]] - old_p_arr[1],
                          new_p_array[final_premutation[2]] - old_p_arr[2]]

        # collision detection
        dmin = float('inf')
        collision = False
        for j, member in enumerate(group_members):
            for i, agent in enumerate(agents_states):
                px = agent.px - old_p_arr[j][0]
                py = agent.py - old_p_arr[j][1]
                vx = agent.vx - three_velocity[j][0]
                vy = agent.vy - three_velocity[j][1]
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - 2 * agent.radius
                if closest_dist < 0:
                    collision = True
                    break

        # 检查是否到达终点，检查的是小组是否到达终点
        px = group_state.px + action[0][0] * self.time_step
        py = group_state.py + action[0][1] * self.time_step
        end_position = np.array((px, py))
        reaching_goal = norm(end_position - np.array([group_state.gx, group_state.gy])) < 3 * self.agent_radius

        # 碰撞、到达的奖励
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        else:
            reward = 0
        # formation、速度


        return reward

    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]). \
            to(self.device)

        return robot_state_tensor, human_states_tensor
