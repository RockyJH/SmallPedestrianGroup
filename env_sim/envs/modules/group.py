import math
import numpy as np
from numpy.linalg import norm

from env_sim.envs.modules.group_action import GroupAction
from env_sim.envs.modules.state import FullState, JointState
from env_sim.envs.modules.agent import Agent
from env_sim.envs.modules.formation import Formation
from env_sim.envs.modules.utils import compute_vn_vc

"""
group对象,使用时,初始需要给出一个中心点和目的位置，会在中心点生成朝向目的点的abreast的队形
以将要采取的速度来建局部系。
"""


class Group(object):
    def __init__(self):
        # 所有状态属性
        self.cx = None
        self.cy = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.width = None  # 实际上是formation的属性
        self.v_pref = 1

        self.policy = None
        self.group_members = []
        self.action_space = list()

        self.delta_theta = math.pi / 6
        self.agent_radius = 0.3
        self.time_step = 0.25
        self.init_relation = [[0, -3 * self.agent_radius], [0, 0], [0, 3 * self.agent_radius]]

        # 初始化模板 横排
        self.abreast = [[0, -3 * self.agent_radius], [0, 0], [0, 3 * self.agent_radius]]
        # 正三角形
        self.v_like = [[2 * self.agent_radius * math.tan(math.pi / 6), -2 * self.agent_radius],
                       [- 4 * self.agent_radius / math.sqrt(3), 0],
                       [2 * self.agent_radius * math.tan(math.pi / 6), -2 * self.agent_radius]]
        # river_like
        self.river = [[3 * self.agent_radius, 0], [0, 0], [-3 * self.agent_radius, 0]]

    # 每次给出速度，之后才能算出局部坐标系，而小组的v 由全部小组成员来决定
    def set(self, cx, cy, gx, gy, vx, vy):
        # 因为只有在每个回合开始的时候会调用以次set，将环境置为0
        self.group_members.clear()
        # 给出小组的基本属性
        self.cx = cx
        self.cy = cy
        self.gx = gx
        self.gy = gy
        tem = np.math.hypot(gx - cx, gy - cy)
        self.vx = (gx - cx) / tem
        self.vy = (gy - cy) / tem

        vn, vc = compute_vn_vc(self.gx - self.cx, self.gy - self.cy)
        px = self.cx + self.vx * self.time_step
        py = self.cy + self.vy * self.time_step
        for i in range(3):
            agent = Agent()
            position = np.array([cx, cy]) + self.init_relation[i][0] * np.array(vn) \
                       + self.init_relation[i][1] * np.array(vc)
            new_position = np.array([px, py]) + self.init_relation[i][0] * np.array(vn) \
                           + self.init_relation[i][1] * np.array(vc)
            vx = new_position[0] - position[0]
            vy = new_position[1] - position[1]
            agent.set(position[0], position[1], vx, vy, new_position[0], new_position[1])  # 设置了位置,速度为0
            self.group_members.append(agent)

    def get_group_members(self):
        return self.group_members

    # 更新group的中心
    def update(self, vx, vy):
        cx = 0
        cy = 0
        for member in self.group_members:
            cx += member.px
            cy += member.py
        self.cx = cx / 3
        self.cy = cy / 3
        self.vx = vx
        self.vy = vy

    # 计算出当前的formation，以便使用relation插值得到候选formation，调用该方法前需要先调set_velocity方法
    def get_formation(self, vn, vc):
        current_formation = Formation()
        current_formation.set_vn_vc(vn, vc)
        central = self.cx, self.cy
        current_formation.set_ref_point(central)
        for agent in self.group_members:
            vector1 = np.array(agent.get_position()) - np.array(central)
            r1 = np.dot(vector1, vn)
            r2 = np.dot(vector1, vc)
            r = [r1, r2]
            current_formation.relation.append(r)

        return current_formation

    # 获取目的地
    def get_goal(self):
        return self.gx, self.gy

    def get_action_space(self):
        self.action_space.clear()  # 清空动作空间
        stop = False
        if self.vx == 0 and self.vy == 0:
            stop = True
            speed = 0.2
            cur_angle = math.atan2(self.gx - self.cx, self.gy - self.cy)

            angles = [cur_angle - 2 * self.delta_theta, cur_angle - 1 * self.delta_theta, cur_angle,
                      cur_angle + self.delta_theta, cur_angle + 2 * self.delta_theta]
            velocities = list()
            for angle in angles:
                velocities.append((speed * math.cos(angle), speed * math.sin(angle)))
        else:
            velocities = list()
            velocities.append((self.vx, self.vy))  # 当前速度
            cur_speed = np.math.hypot(self.vx, self.vy)  # 根号下x^2+y^2
            cur_angle = math.atan2(self.vy, self.vx)
            delta_theta = math.pi / 12
            angles = [cur_angle - 2 * delta_theta, cur_angle - 1 * delta_theta,
                      cur_angle + delta_theta, cur_angle + 2 * delta_theta]
            for angle in angles:
                velocities.append((cur_speed * math.cos(angle), cur_speed * math.sin(angle)))

        for velocity in velocities:  # 对于其中的每一个方向，计算candidate_formations：

            candidate_formations = list()

            vn, vc = compute_vn_vc(velocity[0], velocity[1])

            cur_formation = self.get_formation(vn, vc)
            cur_relation_horizontal = cur_formation.get_relation_horizontal()  # 水平方向的位置
            cur_relation_vertical = cur_formation.get_relation_vertical()  # 竖直方向的位置
            cur_central = (self.cx, self.cy)

            # 对 abreast和v-like的插值
            for i in range(5):
                s = 0.25 * i  # i={0,1,2,3,4}
                form = Formation()
                p1 = np.array(cur_relation_horizontal[0]) * (1.0 - s) + s * np.array(self.abreast[0])  # 数组形式
                p2 = np.array(cur_relation_horizontal[1]) * (1.0 - s) + s * np.array(self.abreast[1])
                p3 = np.array(cur_relation_horizontal[2]) * (1.0 - s) + s * np.array(self.abreast[2])
                form.set_relation([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]])
                form.set_ref_point(cur_central)
                form.set_vn_vc(vn, vc)
                candidate_formations.append(form)

                p11 = np.array(cur_relation_horizontal[0]) * (1.0 - s) + s * np.array(self.v_like[0])  # 数组形式
                p12 = np.array(cur_relation_horizontal[1]) * (1.0 - s) + s * np.array(self.v_like[1])
                p13 = np.array(cur_relation_horizontal[2]) * (1.0 - s) + s * np.array(self.v_like[2])
                form.set_relation([[p11[0], p11[1]], [p12[0], p12[1]], [p13[0], p13[1]]])
                candidate_formations.append(form)

            # 对于竖列的插值-----将当前的relation按照纵队排序。
            for i in range(5):
                s = 0.25 * i  # i={0,1,2,3,4}
                form = Formation()
                p1 = np.array(cur_relation_vertical[0]) * (1.0 - s) + s * np.array(self.river[0])  # 数组形式
                p2 = np.array(cur_relation_vertical[1]) * (1.0 - s) + s * np.array(self.river[1])
                p3 = np.array(cur_relation_vertical[2]) * (1.0 - s) + s * np.array(self.river[2])
                form.set_relation([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]])
                form.set_ref_point(cur_central)
                form.set_vn_vc(vn, vc)
                candidate_formations.append(form)

            # 获取了某一个速度方向上的所有formation，将速度加减
            if stop:
                extend_velocity = [velocity]
            else:
                extend_velocity = self.speed_up_down(velocity[0], velocity[1])
            # 动作空间 group_action = (速度，formation)
            for v in extend_velocity:
                for form in candidate_formations:
                    if isinstance(v,float):
                        print(v)
                    self.action_space.append(GroupAction(v, form))
        return self.action_space

    def speed_up_down(self, vx, vy):

        extend_velocity = list()

        cur_speed = np.math.hypot(vx, vy)  # 根号下x^2+y^2
        if cur_speed < 0.8 * self.v_pref:
            accelerate = cur_speed + 0.2 * self.v_pref
        else:
            accelerate = self.v_pref

        if accelerate != cur_speed:  # 添加加速之后的速度
            v_accelerate = (vx * accelerate / cur_speed, vy * accelerate / cur_speed)
            extend_velocity.append(v_accelerate)

        if cur_speed > 0.2 * self.v_pref:
            decelerate = cur_speed - 0.2 * self.v_pref
        else:
            decelerate = 0

        if decelerate != cur_speed:  # 添加减速之后的速度
            v_decelerate = (vx * decelerate / cur_speed, vy * decelerate / cur_speed)
            extend_velocity.append(v_decelerate)

        extend_velocity.append((vx, vy))
        return extend_velocity

    # 获取完整状态
    def get_full_state(self):
        if self.vx == 0 and self.vy == 0:
            vn,vc = compute_vn_vc(self.gx-self.cx,self.gy-self.cy)
        else:
            vn, vc = compute_vn_vc(self.vx, self.vy)
        width = self.get_formation(vn, vc).get_width() / 2
        return FullState(self.cx, self.cy, self.vx, self.vy, self.gx, self.gy, width, self.v_pref)

    # 获取一个动作,输入group的观察值。 group的动作包含一个速度和一个formation
    def get_action(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state, self.get_action_space(), self.group_members)
        return action
