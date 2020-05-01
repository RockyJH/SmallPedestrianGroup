"""
输入的是group的全部状态+所有anent的观测状态，和一个group的action，输出下一个状态
返回的状态是group的状态和agents的联合状态
"""


class LinearStatePredictor(object):
    def __init__(self, time_step):
        self.trainable = False
        self.time_step = time_step

    def __call__(self, state, action):
        next_group_state = self.compute_next_group_state(state[0])
        next_agent_states = self.compute_next_agent_states(state[1])

        next_observation = [next_group_state, next_agent_states]
        return next_observation

    def compute_next_group_state(self, group_state, action):

        # cx, cy, vx, vy, width, gx, gy, v_pref
        next_state = group_state
        next_state[0] = next_state[0] + action[0][0] * self.time_step
        next_state[1] = next_state[1] + action[0][1] * self.time_step
        next_state[2] = action[0][0]
        next_state[3] = action[0][1]
        next_state[4] = action[1].get_width()

        return next_state

    def compute_next_agent_states(self,agent_states):
        # px, py, vx, vy, radius
        for agent_state in agent_states:
            agent_state[0] = agent_state[2] * self.time_step
            agent_state[1] = agent_state[3] * self.time_step

        return agent_states