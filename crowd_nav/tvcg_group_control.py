import math

from env_sim.envs.modules.formation import Formation
from env_sim.envs.modules.group_action import GroupAction
from env_sim.envs.modules.utils import normalize_vector, compute_vn_vc
from env_sim.envs.policy.policy import Policy
from crowd_nav.modules.box import Box
import numpy as np


class TvcgGroupControl(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TvcgGroupControl'
        self.time_step = 0.25
        self.v_pref = 1
        self.agent_radius = 0.3
        self.tc_min = 2.5
        self.tc_mid = 10
        self.tc_max = 18
        self.delta_mid = np.pi / 3
        self.delta_max = np.pi / 2
        self.k1 = 1
        self.k2 = 1
        self.k3 = 0.05

    def set_phase(self, phase):
        self.phase = phase

    def predict(self, joint_state, candidate_formation, group_members):
        # 要返回的值
        cost = float('inf')
        formation_selected = None
        v_selected = None

        # step1 在gourp类中已经解决

        agents_state = joint_state.agents_states
        group_state = joint_state.self_state
        candidate_formation = candidate_formation
        group_members = group_members

        '''
        step2 获得每个队形的个人空间，个人空间是 在局部系下的宽度 X agent_radius
        '''
        for formation in candidate_formation:
            x, y, w, h = self.compute_formation_xywh(formation)
            formation_box = Box(x, y, w, h, group_state.vx, group_state.vy)

            # 局部坐标系下的vn,vc
            vn, vc = compute_vn_vc(group_state.vx, group_state.vy)
            out_group_boxes = []
            for state in agents_state:
                x, y, w, h = self.compute_agent_xywh_under_vn_vc(group_state.position, state, vn, vc)
                box = Box(x, y, w, h, state.vx, state.vy)  # 将Agent变成box
                out_group_boxes.append(box)

            ttc = self.compute_ttc(out_group_boxes, formation_box, vn, vc)

            # 计算角度域
            delt_theta_max = self.compute_theta_max(ttc)  # 角度偏转域
            orientations = self.compute_orientation_domain(group_state.velocity, delt_theta_max)
            # 计算速度域
            speed_domain = self.compute_speed_domain(ttc, self.v_pref)

            for angle in orientations:
                for speed in speed_domain:
                    v_candidate = (speed * math.cos(angle), speed * math.sin(angle))
                    new_vn, new_vc = compute_vn_vc(v_candidate[0], v_candidate[1])

                    # 新速度下的 box
                    formation = self.compute_formation_box_on_new_v(formation, new_vn, new_vc)
                    x, y, w, h = self.compute_formation_xywh(formation)
                    new_box = Box(x, y, w, h, v_candidate[0], v_candidate[1])

                    out_group_boxes = []
                    for state in agents_state:
                        x, y, w, h = self.compute_agent_xywh_under_vn_vc(group_state.position, state, new_vn, new_vc)
                        box = Box(x, y, w, h, state.vx, state.vy)  # 将Agent变成box
                        out_group_boxes.append(box)

                    new_ttc = self.compute_ttc(out_group_boxes, new_box, new_vn, new_vc)
                    # 计算cost
                    velocity_deviation_cost = self.k1 * normalize_vector(
                        (np.array(group_state.goal_position) - np.array(group_state.position)) - np.array(v_candidate))
                    ttc_cost = self.k2 * ((self.tc_max - new_ttc) / self.tc_max)

                    form_deviation_cost = self.k3 * (
                            np.math.fabs(8 * self.agent_radius - formation.get_width()) / (6 * self.agent_radius))
                    cost_temp = velocity_deviation_cost + ttc_cost + form_deviation_cost

                    if cost_temp < cost:
                        cost = cost_temp
                        formation_selected = formation
                        v_selected = v_candidate

        group_action = GroupAction(v_selected, formation_selected)
        return group_action

    def compute_formation_box_on_new_v(self, formation, new_vn, new_vc):
        p1 = np.array(formation.get_ref_point()) + \
             formation.relation[0][0] * np.array(formation.vn) + \
             formation.relation[0][1] * np.array(formation.vc)
        p2 = np.array(formation.get_ref_point()) + \
             formation.relation[1][0] * np.array(formation.vn) + \
             formation.relation[1][1] * np.array(formation.vc)
        p3 = np.array(formation.get_ref_point()) + \
             formation.relation[2][0] * np.array(formation.vn) + \
             formation.relation[2][1] * np.array(formation.vc)
        global_p = [p1, p2, p3]  # 全局坐标位置
        ref_point = formation.get_ref_point()

        new_formation = Formation()
        new_formation.set_vn_vc(new_vn, new_vc)
        new_formation.set_ref_point(ref_point)
        for p in global_p:
            vector1 = p - np.array(ref_point)
            r1 = np.dot(vector1, new_vn)
            r2 = np.dot(vector1, new_vc)
            r = [r1, r2]
            new_formation.relation.append(r)

        return new_formation

    def compute_speed_domain(self, ttc, prefer_speed):
        if 0 <= ttc <= self.tc_min:
            u_min = prefer_speed * (1 - np.exp(-ttc))
            u_max = prefer_speed
        elif self.tc_min < ttc:
            u_min = prefer_speed
            u_max = prefer_speed

        speed_domain = []
        speed_domain.append(u_min)
        if u_max != u_min:
            speed_domain.append(u_max)
        for i in range(1, 100):
            tem = u_min + 0.1 * i
            if tem < u_max:
                speed_domain.append(tem)
            else:
                break

        return speed_domain

    def compute_orientation_domain(self, velocity, delt_theta):
        orientations = []
        cur_angle = math.atan2(velocity[0], velocity[1])
        orientations.append(cur_angle)

        angle_step = math.pi / 12  # 15度
        tem = []
        for i in range(1, 7):  # 1,2,3,4,5,6
            if delt_theta > angle_step * i:
                tem.append(angle_step * 1)
            else:
                tem.append(delt_theta)
                break

        for angle in tem:
            orientations.append(cur_angle - angle)
            orientations.append(cur_angle + angle)

        return orientations

    def compute_theta_max(self, ttc):

        if 0 <= ttc < self.tc_min:
            return (self.delta_max - self.delta_mid) * np.exp(-ttc) + self.delta_mid
        elif self.tc_min <= ttc < self.tc_mid:
            return self.delta_mid
        elif self.tc_mid <= ttc <= self.tc_max:
            return self.delta_mid * (self.tc_mid - ttc) / (self.tc_max - self.tc_mid) + self.delta_mid
        elif self.tc_max < ttc:
            return 0

    def compute_ttc(self, boxes, box, vn, vc):
        ttc = float('inf')
        for tem_box in boxes:
            tem_ttc = self.swept_test(tem_box, box, vn, vc)
            if tem_ttc < ttc:
                ttc = tem_ttc

        return ttc

    def swept_test(self, tem_box, box, vn, vc):
        vector1 = tem_box.vx - box.vx, tem_box.vy - box.vy
        relative_vx = np.dot(vector1, vn)
        relative_vy = np.dot(vector1, vc)

        # 水平速度为0 且水平方向无重合，则永远不会碰撞,垂直方向同理
        if relative_vx == 0 and (tem_box.x + tem_box.w < box.x or tem_box.x > box.x + box.w):
            return float('inf')
        if relative_vy == 0 and (tem_box.y > box.y + box.h or box.y > tem_box.y + tem_box.h):
            return float('inf')

        # find the distance between the objects on the near and far sides for both x and y
        # 水平和竖直方向进入和退出需要走过的距离
        if relative_vx > 0:
            x_entry_dis = box.x - (tem_box.x + tem_box.w)
            x_exit_dis = (box.x + box.w) - tem_box.x
        else:  # relative_vx < 0
            x_entry_dis = (box.x + box.w) - tem_box.x
            x_exit_dis = box.x - (tem_box.x + tem_box.w)

        if relative_vy > 0:
            y_entry_dis = box.y - (tem_box.y + tem_box.h)
            y_exit_dis = (box.y + box.h) - tem_box.y
        else:
            y_entry_dis = (box.y + box.h) - tem_box.y
            y_exit_dis = box.y - (tem_box.y + tem_box.h)

        # find time of collision and time of leaving for each axis
        # ( if statement is to prevent divide by zero)

        if relative_vx == 0:
            x_entry_time = -float('inf')  # 进入时间 负无穷
            x_exit_time = float('inf')  # 退出时间正无穷
        else:
            x_entry_time = x_entry_dis / relative_vx
            x_exit_time = x_exit_dis / relative_vx

        if relative_vy == 0:
            y_entry_time = -float('inf')  # 进入时间 负无穷
            y_exit_time = float('inf')  # 退出时间正无穷
        else:
            y_entry_time = y_entry_dis / relative_vy
            y_exit_time = y_exit_dis / relative_vy

        # 碰撞发生的条件： 在某一时段，xy都处于进入而未出去的状态
        entry_time = max(x_entry_time, y_entry_time)  # x和y的最晚进入时间,因为同时进才算进
        exit_time = min(x_exit_time, y_exit_time)  # x和y的最早出去时间，因为出去一个就算出去了

        # collision has happend
        if (x_entry_time < 0 and x_exit_time > 0 and y_entry_time < 0 and y_exit_time > 0):
            return 0

        if entry_time > exit_time or (
                x_entry_time < 0 and y_entry_time < 0) or x_entry_time > self.tc_max or y_entry_time > self.tc_max:
            return float('inf')
        else:
            return entry_time

    # 用于计算每个候选队形在期望速度下的box
    def compute_formation_xywh(self, formation):
        left = float('inf')
        right = -float('inf')
        front = -float('inf')
        for rel in formation.relation:
            left = min(left, rel[1])
            right = max(right, rel[1])
            front = max(front, rel[0])

        x = left - self.agent_radius
        y = front - self.agent_radius
        w = right - left + 2 * self.agent_radius
        h = 2 * self.agent_radius
        return x, y, w, h

    # 计算每个agent在局部坐标系下的box
    def compute_agent_xywh_under_vn_vc(self, ref_point, state, vn, vc):

        vector1 = state.px - ref_point[0], state.py - ref_point[1]
        relative_x = np.dot(vector1, vn)
        relative_y = np.dot(vector1, vc)

        x = relative_x - self.agent_radius
        y = relative_y - self.agent_radius
        w = 2 * self.agent_radius
        h = 2 * self.agent_radius
        return x, y, w, h
