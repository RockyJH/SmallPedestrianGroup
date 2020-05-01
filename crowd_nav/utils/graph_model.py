import logging
import itertools  # 迭代器
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu  # softmax
from torch.nn import Parameter  # 数据类型转换
from crowd_nav.utils.helper import mlp  # 多层感知机


# 该类以状态为输入，即机器人的状态和所有人的状态输出一个矩阵，隐含了他们相互之间的关系
class RGL(nn.Module):  # 输入robot_state的纬度，和human_state的纬度
    def __init__(self, config, group_state_dim, agents_state_dim):
        super().__init__()

        num_layer = config.gcn.num_layer  # 2
        X_dim = config.gcn.X_dim  # 32
        w_group_dims = config.gcn.wr_dims  # [64,32]
        w_agent_dims = config.gcn.wh_dims  # [64,32]
        final_state_dim = config.gcn.final_state_dim  # 32
        similarity_function = config.gcn.similarity_function  # 'embedded_gaussian'
        layerwise_graph = config.gcn.layerwise_graph  # true
        skip_connection = config.gcn.skip_connection  # flase

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function  # 'embedded_gaussian'
        self.group_state_dim = group_state_dim  # 机器人的输入纬度
        self.agents_state_dim = agents_state_dim  # 人的输入纬度
        self.num_layer = num_layer  # 2
        self.X_dim = X_dim  # 32
        self.layerwise_graph = layerwise_graph  # true
        self.skip_connection = skip_connection  # false

        logging.info('Similarity_func: {}'.format(self.similarity_function))  # 字符串格式化函数
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_group = mlp(group_state_dim, w_group_dims, last_relu=True)  # w_r对robot的状态维度处理的网络
        self.w_agents = mlp(agents_state_dim, w_agent_dims, last_relu=True)  # w_h对human处理的网络
        # w_r 和 w_h 是两个mlp

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
            ''' w_a 长这样
Parameter containing:
tensor([[-0.6178, -3.0455,  0.7404,  ..., -0.1121, -1.0416,  0.3010],
        [ 3.3320,  0.4126,  1.3288,  ...,  0.1153, -1.5146, -0.2926],
        [-0.3247,  0.4398,  0.1117,  ..., -0.2912,  0.5789,  0.9129],
        ...,
        [-0.4658, -1.8522,  1.8713,  ..., -0.2603, -0.3251, -1.3233],
        [ 2.3578, -3.4380,  0.0918,  ..., -1.4142,  0.9503, -0.4044],
        [ 0.2154, -0.1300,  0.6244,  ..., -0.8157,  1.6516,  0.3914]],
        requires_grad=True)
            '''
            # torch.randn(a,b) 产生 a X b 形式的 tensor ,tensor不可训练，Parameter可训练

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

            '''执行完这一套后
            print (self.Ws) 结果是：
            ParameterList(
                (0): Parameter containing: [torch.FloatTensor of size 32x32]
                (1): Parameter containing: [torch.FloatTensor of size 32x32]
            )
            '''

        # for visualization
        self.A = None

    # 该函数以一个矩阵为输入，输出另一个矩阵
    def compute_similarity_matrix(self, X):
        # if self.similarity_function == 'embedded_gaussian':
        # torch.matul 张量相乘 permute(0,2,1) 第一行不变 第二行和第三行换位
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
        group_state, agents_states = state

        # compute feature matrix X
        group_state_embedings = self.w_group(group_state)
        agents_state_embedings = self.w_agents(agents_states)
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
