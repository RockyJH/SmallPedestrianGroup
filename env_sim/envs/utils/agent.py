import abc
import logging
import numpy as np
from numpy.linalg import norm
from env_sim.envs.policy.policy_factory import policy_factory
from env_sim.envs.utils.action import ActionXY, ActionRot
from env_sim.envs.utils.state import ObservableState, FullState

"""
Agent
Agent 是一个基类,有两个派生类 human 和robot
Agent 有agent的所有物理属性： 位置/速度/朝向/控制策略等
--visibility: humans 总是可见的 robot 可以设置 visible / invisible
--sensor: 可以是：visual input / coordinate input
--kinematics: 可以是： holonomic (move in any direction) / unicycle (has rotation constraints)
--act(observation): 将observation 转化成 state 传递给 policy
"""

'''
问题：
1： getattr(config, section).visible  # 获取 config 对象的 section 属性，从section属性中选择visible
section 从哪里来
2： 62 行，为啥要将agent的 期望速度和半径设置为none
'''


class Agent(object):
    def __init__(self, config, section):
        self.visible = getattr(config, section).visible  # 获取 config 对象的 section 属性，从section属性中选择visible
        self.v_pref = getattr(config, section).v_pref  # 速度
        self.radius = getattr(config, section).radius  # 半径
        self.policy = policy_factory[getattr(config, section).policy]() # 控制策略
        self.sensor = getattr(config, section).sensor # 输入方式
        self.kinematics = self.policy.kinematics if self.policy is not None else None # 运动方式
        self.px = None
        self.py = None # 位置
        self.gx = None
        self.gy = None # 目标
        self.vx = None
        self.vy = None # 速度
        self.theta = None # 角度
        self.time_step = None # 时间步长

    # 输入agent是否可见和运动状态约束。
    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    # 设置控制策略，检查time_step 是否为空，先设置time_step policy kinematics
    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    # 从确定的分布中随即采样自己的 期望速度和半径
    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    # 设置 位置（px,py） 目标（gx,gy）速度（vx,vy) 角度 theta。并且将 半径和 期望速度设置为空。
    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # 获取可观测状态
    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    # 获取下一个可观测状态，返回一个对象，由当前状态和下一个动作计算所得
    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    # 获取完整状态，full state 是一个对象
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    # 获取位置
    def get_position(self):
        return self.px, self.py
    # 设置位置
    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    # 设置目的地
    def get_goal_position(self):
        return self.gx, self.gy

    # 获取起始位置
    def get_start_position(self):
        return self.sx, self.sy

    # 获取速度
    def get_velocity(self):
        return self.vx, self.vy

    # 设置速度
    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    # 抽象方法 获取一个动作
    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy
        """
        return

    # 验证合法性 assert 断言
    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    # 计算位置
    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    # 执行一个动作并且更新到下一个状态 r 是旋转角度
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    # linalg=linear（线性）+algebra（代数），norm则表示范数 这里即两点的距离
    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius