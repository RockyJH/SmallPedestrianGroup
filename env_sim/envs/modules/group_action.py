from env_sim.envs.modules.formation import Formation

'''
将一个速度和一个formation封装成一个group_action
'''


class GroupAction(object):
    def __init__(self, v, formation):
        assert (len(v) == 2)
        assert (isinstance(formation, Formation))
        self.v = v
        self.formation = formation
