import math

import numpy as np
from numpy.linalg import norm

from crowd_nav.rgl_group_control import RglGroupControl
from env_sim.envs.utils.state import ObservableState, FullState, JointState
from env_sim.envs.utils.agent import Agent
from env_sim.envs.utils.formation import Formation

from env_sim.envs.utils.utils import compute_vn_vc

"""
group对象,使用时,初始需要给出一个中心点和目的位置，会在中心点生成朝向目的点的abreast的队形
与TVCG的不同在于，他以v_desire（即由当前位置指向目标位置的向量）来建局部系，
我以下一步将要采取的速度来建局部系。
"""


class Group(object):
    def __init__(self, config, section):
        self.config = config
        self.policy = RglGroupControl
        self.v_pref = getattr(config, section).v_pref
        self.agent_radius = getattr(config, 'Agent').radius  # 用于生成group_member
        self.relation = [[0, -4 * self.agent_radius], [0, 0], [0, 4 * self.agent_radius]]

        self.cx = None
        self.cy = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.width = None

        self.central_position = None
        self.vector_n = None
        self.vector_nc = None
        self.group_members = []
        self.time_step = None
        self.action_space = list()

        # 初始化模板
        # 横排
        self.abreast = [(0, -4 * self.agent_radius), (0, 0), (0, 4 * self.agent_radius)]
        # 正三角形
        self.v_like = [(2 * self.agent_radius * math.tan(math.pi / 6), -2 * self.agent_radius),
                       (- 4 * self.agent_radius / math.sqrt(3), 0),
                       (2 * self.agent_radius * math.tan(math.pi / 6), -2 * self.agent_radius)]
        # 竖排
        self.river = [(4 * self.agent_radius, 0), (0, 0), (-4 * self.agent_radius, 0)]

    # 每次给出速度，之后才能算出局部坐标系，而小组的v 由全部小组成员来决定
    def set(self, cx, cy, gx, gy, vx, vy):
        # 给出小组的基本属性
        self.cx = cx
        self.cy = cy
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy

        # 相对位置关系
        self.group_members = self.generate_group_members(cx, cy)  # 以中心点和局部正方向的描述生成group

    def update(self):
        cx = 0
        cy = 0
        for member in self.group_members:
            cx += member.px
            cy += member.py
        self.cx = cx /3
        self.cy = cy/3

    def generate_group_members(self, cx, cy):
        group_members = []
        # 局部坐标系描述
        vn, vc = compute_vn_vc(self.gx - self.cx, self.gy - self.cy)
        for i in range(3):
            agent = Agent(self.config, 'Agent')
            position = np.array([cx, cy]) + \
                       self.relation[i][0] * np.array(vn) + \
                       self.relation[i][1] * np.array(vc)
            agent.set(position[0], position[1], 0, 0)  # 只是设置了位置和速度为0
            group_members.append(agent)

        return group_members

    def get_group_members(self):
        return self.group_members
    # 设置速度
    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    # 计算出当前的formation，以便使用relation插值得到候选formation
    # 注意，调用该方法前需要先调set_velocity方法
    def get_self_formation(self):
        current_formation = Formation()
        vector_n, vector_nc = compute_vn_vc(self.vx, self.vy)
        central = self.get_central_position()
        current_formation.set_ref_point(central)
        for agent in self.group_members:
            vector1 = np.array(agent.get_position()) - np.array(central)
            r1 = np.dot(vector1, vector_n)
            r2 = np.dot(vector1, vector_nc)
            r = r1, r2
            current_formation.relation.append(r)

        current_formation.set_vn_vc(vector_n, vector_nc)
        return current_formation

    # 获取中心位置,更新中心位置的算法在update中写
    def get_central_position(self):
        return self.cx, self.cy

    # 获取完整状态
    def get_full_state(self):
        return FullState(self.cx, self.cy, self.vx, self.vy, self.width, self.gx, self.gy, self.v_pref)

    # 获取目的地
    def get_goal_position(self):
        return self.gx, self.gy

    # 获取速度
    def get_velocity(self):
        return self.vx, self.vy

    # 获取一个动作,输入group的观察值。 group的动作包含一个速度和一个formation
    def get_action(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, self.get_action_space(), self.group_members)
        return action

    # 计算位置###################################
    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    # 执行一个动作并且更新自身的状态
    def step(self, action):
        """
        Perform an action and update the state
        """
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        self.vx = action.vx
        self.vy = action.vy

    def reached_destination(self):
        return norm(np.array(self.get_central_position()) - np.array(self.get_goal_position())) < 3 * self.agent_radius

    def get_action_space(self):

        action_space = list()

        velocities = list()
        velocities.append(self.get_velocity())  # 当前速度
        cur_speed = np.math.hypot(self.vx, self.vy)  # 根号下x^2+y^2
        cur_angle = math.atan2(self.vy, self.vx)
        delta_theta = math.pi / 6
        angles = [cur_angle + delta_theta, cur_angle + 2 * delta_theta,
                  cur_angle - delta_theta, cur_angle - 2 * delta_theta]

        # velocities里存储的是 大小等于当前的速度，角度方向是五个方向
        for angle in angles:
            velocities.append((cur_speed * math.cos(angle), cur_speed * math.sin(angle)))

        self.action_space.clear()  # 先把候选队形清空

        # 对于其中的每一个方向，先计算candidate_formation：
        for velocity in velocities:
            self.set_velocity(velocity)
            candidate_formation = list()
            cur_formation = self.get_self_formation()
            cur_relation = cur_formation.get_relation()  # 当前位置描述
            cur_central = cur_formation.get_ref_point()
            vn, vc = cur_formation.get_vn_vc()

            # 对 abreast和v-like的插值
            for i in range(5):
                # i={0,1,2,3,4}
                s = 0.25 * i
                form = Formation()
                p1 = np.array(cur_relation[0]) * (1.0 - s) + s * np.array(self.abreast[0])  # 数组形式
                p2 = np.array(cur_relation[1]) * (1.0 - s) + s * np.array(self.abreast[1])
                p3 = np.array(cur_relation[2]) * (1.0 - s) + s * np.array(self.abreast[2])
                form.add_relation_positions([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])])
                form.set_ref_point(cur_central)
                form.set_vn_vc(vn, vc)
                candidate_formation.append(form)

                p11 = np.array(cur_relation[0]) * (1.0 - s) + s * np.array(self.v_like[0])  # 数组形式
                p12 = np.array(cur_relation[1]) * (1.0 - s) + s * np.array(self.v_like[1])
                p13 = np.array(cur_relation[2]) * (1.0 - s) + s * np.array(self.v_like[2])
                form.add_relation_positions([(p11[0], p11[1]), (p12[0], p12[1]), (p13[0], p13[1])])
                candidate_formation.append(form)

            # 对于竖列的插值-----将当前的relation按照纵队排序。
            for i in range(5):
                # i={0,1,2,3,4}
                s = 0.25 * i
                form = Formation()
                p1 = np.array(cur_relation[0]) * (1.0 - s) + s * np.array(self.river[0])  # 数组形式
                p2 = np.array(cur_relation[1]) * (1.0 - s) + s * np.array(self.river[1])
                p3 = np.array(cur_relation[2]) * (1.0 - s) + s * np.array(self.river[2])
                form.add_relation_positions([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])])
                form.set_ref_point(cur_central)
                form.set_vn_vc(vn,vc)
                candidate_formation.append(form)
                # river-like 是两种插值方法
                p11 = np.array(cur_relation[0]) * (1.0 - s) + s * np.array(self.v_like[2])  # 数组形式
                p12 = np.array(cur_relation[1]) * (1.0 - s) + s * np.array(self.v_like[1])
                p13 = np.array(cur_relation[2]) * (1.0 - s) + s * np.array(self.v_like[0])
                form.add_relation_positions([(p11[0], p11[1]), (p12[0], p12[1]), (p13[0], p13[1])])
                candidate_formation.append(form)
            # 已经在某个速度方向上获取了formation，速度加减
            extend_velocity = self.speed_up_down()
            # 动作空间 action = (速度，formation)
            for v in extend_velocity:
                for form in candidate_formation:
                    action_space.append(v, form)

        return action_space

    def speed_up_down(self):

        extend_velocity = list()

        cur_speed = np.math.hypot(self.vx, self.vy)  # 根号下x^2+y^2
        if cur_speed < 0.8 * self.v_pref:
            accelerate = cur_speed + 0.2 * self.v_pref
        else:
            accelerate = self.v_pref

        if accelerate != cur_speed:  # 添加加速之后的速度
            v_accelerate = (self.vx * (accelerate / cur_speed), self.vy * (accelerate / cur_speed))
            extend_velocity.append(v_accelerate)

        if cur_speed > 0.2 * self.v_pref:
            decelerate = cur_speed - 0.2 * self.v_pref
        else:
            decelerate = 0

        if decelerate != cur_speed:  # 添加减速之后的速度
            v_decelerate = (self.vx * (decelerate / cur_speed), self.vy * (decelerate / cur_speed))
            extend_velocity.append(v_decelerate)

        extend_velocity.append((self.vx, self.vy))
        return extend_velocity
