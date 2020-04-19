from env_sim.envs.utils.agent import Agent
from env_sim.envs.utils.state import JointState
''' 
这个class定义了human 和agent不同的是 它的状态是 他的JointState联合状态
'''

class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
