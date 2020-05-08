import logging
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter  # 数据类型转换
from crowd_nav.modules.mlp import mlp


# 该类以状态为输入，即机器人的状态和所有人的状态输出一个矩阵，隐含了他们相互之间的关系
class RGL(nn.Module):  # 输入robot_state的纬度，和human_state的纬度
    def __init__(self):
        super().__init__()
        self.similarity_function = 'embedded_gaussian'
        self.group_state_dim = 8  # 机器人的输入纬度
        self.agents_state_dim = 5  # 人的输入纬度
        self.num_layer = 2
        self.X_dim = 32
        self.layerwise_graph = True
        self.skip_connection = False

        logging.info('[graph_model.py 提示：] Similarity_func: {}'.format(self.similarity_function))  # 字符串格式化函数
        logging.info('[graph_model.py 提示：] Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('[graph_model.py 提示：] Skip_connection: {}'.format(self.skip_connection))
        logging.info('[graph_model.py 提示：] GCN layers: {}'.format(self.num_layer))

        w_group_dims = [64, 32]
        w_agent_dims = [64, 32]
        final_state_dim = 32
        self.w_group = mlp(self.group_state_dim, w_group_dims, last_relu=True)  # 处理group的状态
        self.w_agents = mlp(self.agents_state_dim, w_agent_dims, last_relu=True)  # 处理Agent的状态

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))

        # TODO: try other dim size
        embedding_dim = self.X_dim  # 32
        self.Ws = torch.nn.ParameterList()  # 网络的权重

        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(Parameter(torch.randn(embedding_dim, embedding_dim)))

        # for visualization
        self.A = None

    def compute_similarity_matrix(self, X):
        A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
        normalized_A = softmax(A, dim=2)  # 使用softmax 激活
        return normalized_A

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        group_state, agents_state = state

        # compute feature matrix X
        group_state_embedings = self.w_group(group_state)
        agents_state_embedings = self.w_agents(agents_state)
        X = torch.cat([group_state_embedings, agents_state_embedings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            self.A = normalized_A[0, :, :].data.cpu().numpy()

        next_H = H = X
        for i in range(self.num_layer):
            if self.layerwise_graph:
                A = self.compute_similarity_matrix(H)
                next_H = relu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:
                next_H = relu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))

            if self.skip_connection:
                next_H += H
            H = next_H

        return next_H
