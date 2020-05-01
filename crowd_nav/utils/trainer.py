import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


'''motion predict RL Trainer
1价值估计者
2状态估计者
3记忆库
4运行设备指示
5策略
6writer
7批大小
8优化器
9人数
10 reduce_sp_update_frequency，缩减记忆库的频率？
11冻结状态预测
12detach_state_predictor
13是否共享图模型
'''


class MPRLTrainer(object):
    def __init__(self, value_estimator, state_predictor,
                 memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor,
                 share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device  # cpu
        self.writer = writer
        self.target_policy = policy  # model_predictive_rl
        self.target_model = None  # target_model none
        self.criterion = nn.MSELoss().to(device)  # 误差 均方差
        self.memory = memory  # 记忆库
        self.data_loader = None  # 没有data_loader
        self.batch_size = batch_size  # 100
        self.optimizer_str = optimizer_str  # Adam
        self.reduce_sp_update_frequency = reduce_sp_update_frequency  # False
        self.state_predictor_update_interval = human_num  # 5
        self.freeze_state_predictor = freeze_state_predictor  # False
        self.detach_state_predictor = detach_state_predictor  # False
        self.share_graph_model = share_graph_model  # False
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # 设置学习率
    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)

        # 这里用了Adam 所以后面的不用看了
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        # 对可训练的输出它的参数
        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    # 对 x 个epoch优化
    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            # 从记忆中提取batch_size 的记忆，打乱顺序。
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        # 训练价值估计和人的动作预测的网络
        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0

            for data in self.data_loader:
                robot_states, human_states, values, _, _, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                # optimizer.zero_grad()可以把所有Variable的grad成员数值变为0

                # 载入记忆后，先计算一个状态的预测值，当作估计值，然后取记忆中的value作为i实际值
                outputs = self.value_estimator((robot_states, human_states))
                values = values.to(self.device)  # 变成cpu的形式
                loss = self.criterion(outputs, values)  # 误差是预测的价值和真实的价值的差值
                loss.backward()  # 误差反向传递
                self.v_optimizer.step()
                # optimizer.step()则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值
                epoch_v_loss += loss.data.item()  # 误差

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:  # 该数值为5
                        update_state_predictor = False

                    # 对于human动作预测网络的训练也是一样，由当前动作预测一个状态然后和记忆里的状态做对比
                    # 将误差反向传递
                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
                        loss = self.criterion(next_human_states_est, next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += loss.data.item()
                    update_counter += 1

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))

        return

    # 批优化 输入批大小，和回合
    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        for data in self.data_loader:
            robot_states, human_states, _, rewards, next_robot_states, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((robot_states, human_states))

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model((next_robot_states, next_human_states))

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                    detach=self.detach_state_predictor)
                    loss = self.criterion(next_human_states_est, next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)

        return average_v_loss, average_s_loss


# value net RL Trainer 其他模型用的，所以可以不看
class VNRLTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)

        return average_loss


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
