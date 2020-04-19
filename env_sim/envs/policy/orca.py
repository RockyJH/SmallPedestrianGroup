import numpy as np
import rvo2
from env_sim.envs.policy.policy import Policy
from env_sim.envs.utils.action import ActionXY


# 这个orca用来控制一个agent，将受控制的成为robot，即这里我是robot
# 输入state，返回我自己下一步的动作
class ORCA(Policy):
    def __init__(self):
        """
        timeStep        ：仿真的时间步长，必须为正
        neighborDist    ：最大邻居距离，以此来判定谁是我的邻居（即哪些人需要考虑）越大运行时间越长，太小不安全 非负        maxNeighbors : 一个agent最多可以考虑多少个邻居，越大运行时间越长，太少不安全
        timeHorizon     : 意思就是我能向前看多远的时间，看的越远我就能越提前反映，同时我速度变换的自由度就越小。必须为正 The default minimal amount of time for which
        timeHorizonObst : 对障碍物能往前预判多长时间，也是越大越提前反映，自由度越小
        radius          : 半径 非负
        maxSpeed        : 最大速度，非负
        velocity        ： 初始速度，二维线性值，可选

        ORCA 先使用最大邻居距离和最大的邻居数目找到哪些邻居是需要考虑的
        这里将他们设置的足够大，以使得他们能将所有人都考虑进来
        Time_horizon 必须至少在一个步长是安全的，即要大于等于时间步长喽
        这里静态障碍无不考虑。

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False  # 不可训练
        self.multiagent_training = True
        self.kinematics = 'holonomic'  # 运动方式
        self.safety_space = 0
        self.neighbor_dist = 10  # 10以外的就不是我的邻居了
        self.max_neighbors = 10  # 最多考虑10个人
        self.time_horizon = 5  # 向前考虑5个时间单位
        self.time_horizon_obst = 5  # 静态障碍物也是，不过没有用
        self.radius = 0.3  # 半径
        self.max_speed = 1  # 最大速度
        self.sim = None

    def configure(self, config):  # 这是父类的方法，抽象方法，没有实现
        return

    def set_phase(self, phase):  # 这个phase是真尼玛烦人
        return

    def predict(self, state):
        """
        在每一个时间步长内都创建一个rvo2 模拟器往前执行一步，输入状态，返回动作
        到达目的地也不停下来，因为一停，破坏了互反的假设
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp
        """
        robot_state = state.robot_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        # params 是一个list

        # 当当前人数和上一不人数不一样了，才删掉sim并置为空
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.01 + self.safety_space,
                              robot_state.v_pref, robot_state.velocity)  # 最大速度，所以机器人的期望速度是一个值
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)  # 自己的期望速度是最大速度
        else:  # sim不为空，但是人数也没有发生变化
            self.sim.setAgentPosition(0, robot_state.position)  # 0 是agent的编号，0 号是robot
            self.sim.setAgentVelocity(0, robot_state.velocity)  # 这一步只 设置位置和速度
            for i, human_state in enumerate(state.human_states):  # 其他人也只是速度和位置变化了
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # 设置期望速度这里要改。
        ############################################################
        velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)  # 根号下a^2+b^2
        pref_vel = velocity / speed if speed > 1 else velocity  # 最终pref_vel是一个二维速度向量。
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))  # 设置机器人期望速度

        # 设置其他人期望速度，都设置为0，因为自己观测不到
        for i, human_state in enumerate(state.human_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))  # 其他人的期望速度都是0

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action


# 这个策略用于控制人群，输入环境状态，返回所有人的下一步动作
class CentralizedORCA(ORCA):  # 从ORCA 继承而来
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """ Centralized planning for all agents """
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state):
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            for agent_state in state:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, agent_state.velocity)
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        self.sim.doStep()
        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]

        return actions
