import numpy as np
from numpy.linalg import norm
from env_sim.envs.policy.orca import CentralizedORCA
from env_sim.envs.modules.state import ObservableState, FullState

"""
1. Agent的部分属性写死, v_pref默认1, radius默认0.3, 控制方式默认 ORCA
2. 生成Agent的时候不需要其他配置文件, v_pref可以随机 sample_random_v_pref(self)
3. 需要设置位置,速度,目标
"""


class Agent(object):
    def __init__(self):
        self.v_pref = 1  # 期望速度大小 1
        self.radius = 0.3  # 半径0.3写死
        self.time_step = 0.25  # 时间步长

        self.px = None
        self.py = None  # 位置
        self.gx = None
        self.gy = None  # 目标
        self.vx = None
        self.vy = None  # 速度

        self.start_px = None
        self.start_py = None

    def sample_random_v_pref(self):
        self.v_pref = np.random.uniform(0.5, 1.5)

    # 设置 位置（px,py） 速度（vx,vy) 目标（gx,gy）
    def set(self, px, py, vx, vy, gx=None, gy=None):
        self.px = px
        self.py = py
        self.start_px = px
        self.start_py = py
        self.vx = vx  # 初始生成agent的时候，速度为0
        self.vy = vy
        if gx is not None:
            self.gx = gx
        if gy is not None:
            self.gy = gy

    # 获取可观测状态--返回：ObservableState对象-包含拼接属性
    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    # 获取完整状态--返回：FullState对象
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref)
    # 执行一个动作并且更新到下一个状态
    def step(self, action):
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        self.vx = action.vx
        self.vy = action.vy

    # 被step调用
    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    # 是否到达目标位置
    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    # get/set方法###############################
    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_position(self):
        return self.px, self.py

    def set_goal_position(self, position):
        self.gx = position[0]
        self.gy = position[1]

    def get_goal(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.start_px, self.start_py

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]
