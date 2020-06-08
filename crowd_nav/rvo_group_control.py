from abc import ABC

import torch
import numpy as np
import numpy.linalg

from numpy.linalg import norm
from env_sim.envs.policy.policy import Policy
from env_sim.envs.modules.utils import point_to_segment_dist
from crowd_nav.modules.state_predictor import LinearStatePredictor
from crowd_nav.modules.graph_model import RGL
from crowd_nav.modules.value_estimator import ValueEstimator


class RvoGroupControl(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RvoGroupControl'
        self.time_step = 0.25
        self.v_pref = 1
        self.agent_radius = 0.3

    def predict(self, joint_state):
        return None
