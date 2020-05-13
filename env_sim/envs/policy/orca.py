import numpy as np
import rvo2
from env_sim.envs.policy.policy import Policy
from env_sim.envs.modules.utils import ActionXY

"""
timeStep        ：仿真的时间步长，须为正
neighborDist    ：最大邻居距离，判定哪些人需要考虑，越大运行时间越长，太小不安全，非负
maxNeighbors    ：最多考虑多少个邻居，越大运行时间越长，太少不安全
timeHorizon     ：向前看多远的时间，越远越能提前反应，同时速度变换的自由越小，必须为正
timeHorizonObst ：对障碍物能往前预判多长时间，也是越大越提前反映，自由度越小
radius          ：半径 非负
maxSpeed        ：最大速度，非负
velocity        ：初始速度，二维线性值，可选
先使用最大邻居距离和最大的邻居数目，将他们设置的足够大，以使得他们能将所有人都考虑进来
Time_horizon 必须至少在一个步长是安全的，即要大于等于时间步长喽,静态障碍物不考虑。
在每一个时间片内都创建一个rvo2 模拟器往前执行一步，输入状态，返回动作
Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp
"""


class ORCA(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False  # 不可训练
        self.safety_space = 0  # 未使用
        self.neighbor_dist = 10  # 10以外的就不是我的邻居了
        self.max_neighbors = 10  # 最多考虑10个人
        self.time_horizon = 5  # 向前考虑5个时间单位
        self.time_horizon_obst = 5  # 静态障碍物也是，不过没有用
        self.radius = 0.3  # 半径
        self.max_speed = 1.2  # 最大速度
        self.time_step = 0.25
        self.sim = None

    def predict(self, state, action_space=None, members=None):
        self_state = state.self_state
        agents_states = state.agents_states
        # params 是一个list
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        # 更新sim
        if self.sim is not None and self.sim.getNumAgents() != len(state.agents_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            # 位置, *params, 半径, v_pref(期望速度大小), 速度
            self.sim.addAgent(self_state.position, *params, self_state.radius + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for agent_state in state.agents_states:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + self.safety_space,
                                  1, agent_state.velocity)  # 自己的期望速度是最大速度
        else:  # 只是更新sim里的部分属性
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, agent_state in enumerate(state.agents_states):
                self.sim.setAgentPosition(i + 1, agent_state.position)
                self.sim.setAgentVelocity(i + 1, agent_state.velocity)

        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)  # 根号下a^2+b^2
        pref_vel = velocity / speed if speed > 0.1 else velocity  # 当和目标的距离小于0.1才开始减速
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))  # 设置机器人期望速度

        # 设置其他人期望速度，都设置为0，因为自己观测不到
        for i, agent_state in enumerate(state.agents_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))  # 其他人的期望速度都是0

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))

        return action

    def configure(self, config):  # 父类方法
        return

    def set_phase(self, phase):  # 父类方法
        return


class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()

    def predict(self, state, action_space=None, members=None):
        """ Centralized planning for all agents """
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state):
            del self.sim
            self.sim = None

        if self.sim is None:
            # 环境最大速度是1.2
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius + self.safety_space, self.max_speed)
            for i, agent_state in enumerate(state):
                if i < 3:  # 将groupmember的最大速度设置为1.2
                    self.sim.addAgent(agent_state.position, *params, agent_state.radius,
                                      self.max_speed, agent_state.velocity)
                else:  # 将其他agent的最大速度设置为1
                    self.sim.addAgent(agent_state.position, *params, agent_state.radius,
                                      1, agent_state.velocity)
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            if i < 3:
                # 位置相减得到的是0.25秒应该走完的向量，乘以4得到的是速度，这个速度的差别使得相互追赶
                velocity = np.array((4 * (agent_state.gx - agent_state.px), 4 * (agent_state.gy - agent_state.py)))
                self.sim.setAgentPrefVelocity(i, (velocity[0], velocity[1]))
            else:
                velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
                speed = np.linalg.norm(velocity)
                pref_vel = velocity / speed if speed > 1 else velocity  # 设置为1相当于提前四个步长就减速了，可设置为1
                self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        self.sim.doStep()
        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]

        return actions
