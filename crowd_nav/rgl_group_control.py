import torch
import numpy as np
import numpy.linalg

from numpy.linalg import norm
from env_sim.envs.policy.policy import Policy
from env_sim.envs.modules.utils import point_to_segment_dist
from crowd_nav.modules.state_predictor import LinearStatePredictor
from crowd_nav.modules.graph_model import RGL
from crowd_nav.modules.value_estimator import ValueEstimator


class RglGroupControl(Policy):

    def __init__(self):
        super().__init__()
        self.name = 'RglGroupControl'
        self.trainable = True
        self.gamma = 0.9
        self.group_state_dim = 8
        self.agent_state_dim = 5
        self.time_step = 0.25
        self.v_pref = 1
        self.agent_radius = 0.3
        self.state_predictor = LinearStatePredictor(self.time_step)
        self.graph_model = RGL()
        self.value_estimator = ValueEstimator(self.graph_model)
        self.model = [self.graph_model, self.value_estimator.value_network]  # value_network 是后面的哪个多层感知机

        # 需要set的属性
        self.epsilon = None
        self.phase = None
        self.device = None
        self.last_state = None

        # reward function ！！！！！！！！！！！！！！！
        # 在env_sim.py里用于获得真实reward，rgl_group_control.py里用于预测，一并更改！！！！！
        self.success_reward = 1
        self.collision_penalty = -1
        self.k1 = 0.08  # 速度偏向的权重
        self.k2 = 0.04  # 队形差异权重
        self.k3 = 3  # 到达终点判定: 距离 <=  K3 * self.agent_radius

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_phase(self, phase):
        self.phase = phase

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator  # value_estiator包括两个部分，【self.graph_model,self.value_netword】

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
    def predict(self, joint_state, action_space, group_members):
        # 如果可能性小于随机概率则随机选取动作
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = action_space[np.random.choice(len(action_space))]
        else:
            max_action = None
            max_value = float('-inf')

            # 遍历动作空间--计算执行该动作能获得的value，选取value最大的动作
            for action in action_space:
                # 组外agent线性预测，小组直接放到动作的位置上
                next_state = self.state_predictor(joint_state, action)
                # next_state是join_state,
                state_tensor = next_state.to_tensor(True, self.device)
                # 这样转换出来的shape.state == 3 transform出来的是2
                # 此处用上一步得到的状态估计，获得value
                value_return = self.value_estimator(state_tensor)
                # 由碰撞检测获得reward_est,这个工作由 self.estimate_reward函数来负责
                reward_est = self.estimate_reward(joint_state, action, group_members)
                value = reward_est + self.get_normalized_gamma() * value_return
                if value > max_value:
                    max_value = value
                    max_action = action

            if max_action is None:
                raise ValueError('Value network is not well trained.')

        # 训练时，执行到此，选出了动作，那么当前的state就成了last_state
        if self.phase == 'train':
            self.last_state = self.transform(joint_state)

        return max_action

    # 输入（状态，动作），线性推演碰撞检测，返回reward
    def estimate_reward(self, joint_state, action, group_members):
        ##########################################################################################
        #### 参照点外插,当前的实现是直接按照当前速度走下一步。###
        ##########################################################################################

        agents_state = joint_state.agents_states
        group_full_state = joint_state.self_state

        # 目的：由group的action（v，formation）获得每个group_memeber的action（一个速度）
        vx, vy = action.v
        formation = action.formation

        # 参照点向外扩展一个时间片
        p = formation.get_ref_point()
        p = p[0] + vx * self.time_step, p[1] + vy * self.time_step
        relation = formation.get_relation_horizontal()  # [[a,b],[a,b],[a,b]]
        vn, vc = formation.get_vn_vc()  # vn = (x,y) vc = (x,y)

        # 获取新的位置,np_array的形式
        new_p1 = np.array(p) + relation[0][0] * np.array(vn) + relation[0][1] * np.array(vc)
        new_p2 = np.array(p) + relation[1][0] * np.array(vn) + relation[1][1] * np.array(vc)
        new_p3 = np.array(p) + relation[2][0] * np.array(vn) + relation[2][1] * np.array(vc)
        new_p = [new_p1, new_p2, new_p3]  # list里面是数组类型的新位置

        # 一对一地分配谁去哪里
        member1, member2, member3 = group_members

        old_p1 = np.array(member1.get_position())
        old_p2 = np.array(member2.get_position())
        old_p3 = np.array(member3.get_position())
        old_p = [old_p1, old_p2, old_p3]

        orders = [[0, 1, 2], [0, 2, 1],
                  [1, 0, 2], [1, 2, 0],
                  [2, 0, 1], [2, 1, 0]]

        min_dis_index = 0
        min_dis = float('inf')

        for i, order in enumerate(orders):
            dis = np.linalg.norm(new_p[order[0]] - old_p[0]) \
                  + np.linalg.norm(new_p[order[1]] - old_p[1]) \
                  + np.linalg.norm(new_p[order[2]] - old_p[2])
            if dis < min_dis:
                min_dis = dis
                min_dis_index = i

        final_order = orders[min_dis_index]
        # 即 1，2，3个member 分别去 final_premutaion 指示的地方

        v_3 = [new_p[final_order[0]] - old_p[0],
               new_p[final_order[1]] - old_p[1],
               new_p[final_order[2]] - old_p[2]]

        # collision detection
        collision = False
        for j, member in enumerate(group_members):
            for i, agent in enumerate(agents_state):
                tem_px = agent.px - old_p[j][0]
                tem_py = agent.py - old_p[j][1]
                tem_vx = agent.vx - v_3[j][0]
                tem_vy = agent.vy - v_3[j][1]
                ex = tem_px + tem_vx * self.time_step
                ey = tem_py + tem_vy * self.time_step
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(tem_px, tem_py, ex, ey, 0, 0) - 2 * agent.radius
                if closest_dist < 0:
                    collision = True
                    break

        # 检查是否到达终点，检查的是小组是否到达终点
        px = group_full_state.px + vx * self.time_step
        py = group_full_state.py + vy * self.time_step
        end_position = np.array((px, py))
        reaching_goal = norm(
            end_position - np.array([group_full_state.gx, group_full_state.gy])) < self.k3 * self.agent_radius

        # 碰撞、到达的奖励
        reward = 0
        if collision:
            reward = self.collision_penalty
        elif reaching_goal:
            reward = self.success_reward
        else:
            reward = 0
        # formation reward

        des_v = np.array((group_full_state.gx - group_full_state.px, group_full_state.gy - group_full_state.py))
        des_s = np.linalg.norm(des_v)
        des_v = des_v / des_s
        cur_velocity = (group_full_state.vx, group_full_state.vy)
        v_deviation = np.linalg.norm(np.array(cur_velocity) - des_v) / 2
        velocity_deviation_reward = self.k1 * (0.5 - v_deviation)

        cur_width = formation.get_width()
        form_deviation = np.math.fabs(8 * self.agent_radius - cur_width) / (6 * self.agent_radius)
        form_deviation_reward = self.k2 * (0.5 - form_deviation)

        reward = reward + velocity_deviation_reward + form_deviation_reward

        return reward

    def transform(self, joint_state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        group_state_tensor = torch.Tensor([joint_state.self_state.to_tuple()]).to(self.device)
        agents_states_tensor = torch.Tensor([agent_state.to_tuple() for agent_state in joint_state.agents_states]). \
            to(self.device)

        return group_state_tensor, agents_states_tensor
