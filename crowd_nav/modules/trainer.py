import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MPRLTrainer(object):
    def __init__(self, value_estimator,
                 memory, device, policy, writer, batch_size, optimizer_str, human_num,):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.device = device  # cpu
        self.writer = writer
        self.target_policy = policy  # 传进来的policy有什么用？
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)  # 误差 均方差
        self.memory = memory  # 记忆库
        self.data_loader = None  # 没有data_loader
        self.batch_size = batch_size  # 100
        self.optimizer_str = optimizer_str  # Adam
        self.reduce_sp_update_frequency = False
        self.state_predictor_update_interval = human_num  # 5
        self.freeze_state_predictor = False
        self.detach_state_predictor = False
        self.share_graph_model = False
        self.v_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # 设置学习率
    def set_learning_rate(self, learning_rate):
        self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
        logging.info('Lr: {} 的参数 {} 使用 {} 优化器'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    # 批优化批次数，和当前回合数
    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        batch_count = 0
        for data in self.data_loader:
            group_states, agents_states, _, rewards, next_group_states, next_agents_states = data
            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((group_states, agents_states))

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model((next_group_states, next_agents_states))

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        logging.info('Average loss : %.2E', average_v_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)

        return average_v_loss


def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """

    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states
