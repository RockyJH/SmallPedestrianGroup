from crowd_nav.rgl_group_control import RglGroupControl
from env_sim.envs.policy.orca import ORCA,CentralizedORCA


def none_policy():
    return None


policy_factory = dict()  # 存储key value 键值对
policy_factory['orca'] = ORCA
policy_factory['rl_group'] = RglGroupControl
policy_factory['none'] = none_policy
