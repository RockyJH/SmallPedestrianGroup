"""
输入的是group的全部状态+所有anent的观测状态，和一个group的action，输出下一个状态
返回的状态是group的状态和agents的联合状态
"""
from env_sim.envs.modules.state import JointState, FullState


class LinearStatePredictor(object):
    def __init__(self, time_step):
        self.trainable = False
        self.time_step = time_step

    def __call__(self, joint_state, action):
        next_group_state = self.compute_next_group_state(joint_state.self_state, action)
        next_agents_states = self.compute_next_agent_states(joint_state.agents_states)

        next_joint_state = JointState(next_group_state, next_agents_states)
        return next_joint_state

    def compute_next_group_state(self, group_state, action):
        vx, vy = action.v
        px = group_state.px + vx * self.time_step
        py = group_state.py + vy * self.time_step
        radius = action.formation.get_width() / 2
        next_state = FullState( px, py, vx, vy, radius, group_state.gx, group_state.gy, group_state.v_pref)
        return next_state

    def compute_next_agent_states(self, agent_states):
        # px, py, vx, vy, radius
        for agent_state in agent_states:
            agent_state.px = agent_state.vx * self.time_step
            agent_state.py = agent_state.vy * self.time_step

        return agent_states
