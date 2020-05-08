import abc  # abstract base class 抽象基类
import numpy as np
import torch

# policy 对象 是所有policy类型的基类，有一个抽象方法predict（）
'''
输入一个state 输出一个action，
ORCA: ORCA
'''


class Policy(object):
    def __init__(self):
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = 0.25
        # if agent is assumed to know the dynamics of real world
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_model(self):
        return self.model

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)  # only save the parameters

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @abc.abstractmethod
    def predict(self, state, action_space, members):
        """
        Policy takes state as input and output an action
        """
        return
