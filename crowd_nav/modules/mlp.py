import torch.nn as nn

'''
包含了一个方法用于生成多层感知机mlp,
input_dim : 接受输入的维度；
mlp_dims  : 是一个list，类似于[150, 100, 100, 50]
last_relu : 默认最后一层不使用激活函数
返回一个net
'''


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net
