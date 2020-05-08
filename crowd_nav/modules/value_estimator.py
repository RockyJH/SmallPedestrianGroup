import torch.nn as nn
from crowd_nav.modules.mlp import mlp

'''
价值估计网络 传入的参数是 graph_model
由 graph_model 后面加一个多呈感知机构成价值网络
输入纬度是 X矩阵的纬度；mlp中每层的输入纬度写在配置文件
forward 函数中取graph_model中的代表robot状态的向量传入价值网络，返回一个价值
'''


# 输入一个状态，返回一个价值。
class ValueEstimator(nn.Module):
    def __init__(self,graph_model):
        super().__init__()  # 从nn.Module继承下来的都要有这一步了
        self.graph_model = graph_model
        self.value_network = mlp(32, [32, 100, 100, 1])

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(state)[:, 0, :]  # 输入到graph_model,然后取第一行的状态向量
        value = self.value_network(state_embedding)  # 传进 价值网络，返回价值估值
        return value
