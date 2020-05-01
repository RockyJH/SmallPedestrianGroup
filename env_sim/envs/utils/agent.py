import numpy as np
from numpy.linalg import norm
from env_sim.envs.policy.orca import ORCA
from env_sim.envs.utils.state import ObservableState, FullState, JointState

"""
Agent 的所有物理属性： 
act(observation): 将observation 转化成 state 传递给 policy
"""


class Agent(object):
    def __init__(self, config, section):
        self.v_pref = getattr(config, section).v_pref  # 速度
        self.radius = getattr(config, section).radius  # 半径
        self.policy = ORCA

        self.px = None
        self.py = None  # 位置
        self.gx = None
        self.gy = None  # 目标
        self.vx = None
        self.vy = None  # 速度
        self.time_step = None  # 时间步长

        self.start_px = None
        self.start_py = None

    # 从确定的分布中随即采样自己的 期望速度和半径
    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.5)
        # self.radius = np.random.uniform(0.3, 0.5)

    # 设置 位置（px,py） 目标（gx,gy）速度（vx,vy) 并且将 半径和 期望速度设置为空。
    def set(self, px, py, vx, vy, gx=None, gy=None, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.start_px = px
        self.start_py = py
        if gx is not None:
            self.gx = gx
        if gy is not None:
            self.gy = gy
        self.vx = vx  # 初始生成agent的时候，速度为0
        self.vy = vy
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # 获取可观测状态
    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    # 输入一个动作获取下一个可观测状态
    def get_next_observable_state(self, action):
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        next_vx = action.vx
        next_vy = action.vy
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    # 获取完整状态，full state 是一个对象
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref)

    # 获取位置
    def get_position(self):
        return self.px, self.py

    # 设置位置
    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_position(self):
        return self.px, self.py

    # 设置目的地
    def set_goal(self, gx, gy):
        self.gx = gx
        self.gy = gy

    # 获取目的地
    def get_goal_position(self):
        return self.gx, self.gy

    # 获取起始位置
    def get_start_position(self):
        return self.start_px, self.start_py

    # 获取速度
    def get_velocity(self):
        return self.vx, self.vy

    # 设置速度
    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    # 输入当前agent的observation(jointState+ob),由控制策略返回一个动作
    def get_action(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    # 输入动作和时间间隔返回下一个位置
    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    # 执行一个动作并且更新到下一个状态
    def step(self, action):
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        self.vx = action.vx
        self.vy = action.vy

    # 计算agent是否到达目标位置
    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius
