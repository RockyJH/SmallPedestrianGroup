from env_sim.envs.policy.policy import Policy


class RvoGroupControl(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RvoGroupControl'
        self.time_step = 0.25
        self.v_pref = 1
        self.agent_radius = 0.3

    def predict(self, joint_state):
        return None
