from abc import ABC
import torch
import torch.nn as nn
import numpy as np
import config
import random
import os

import utils
from DPAMSA.models import Encoder
from DPAMSA.replay_memory import ReplayMemory


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, seq_num, max_seq_len, action_number, max_value, d_model=64):
        super(Net, self).__init__()
        self.max_value = max_value
        dim = seq_num * (max_seq_len + 1)
        self.encoder = Encoder(6, d_model, dim)
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(dim * d_model, 1028)
        self.f1 = nn.LeakyReLU()
        self.l2 = nn.Linear(1028, 512)
        self.f2 = nn.LeakyReLU()
        self.l3 = nn.Linear(512, action_number)
        self.f3 = nn.Tanh()

        self.mask = lambda x, y: (x != y).unsqueeze(-2)

    def forward(self, x):
        x = self.encoder(x, self.mask(x, 0))
        # x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.f1(self.l1(x))
        x = self.f2(self.l2(x))
        x = self.f3(self.l3(x))
        x = torch.mul(x, self.max_value)

        return x


class DQN(ABC):
    def __init__(self, action_number, seq_num, max_seq_len, max_value):
        super(DQN, self).__init__()
        self.seq_num = seq_num
        self.max_seq_len = max_seq_len
        self.action_number = action_number
        self.eval_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.DEVICE)
        self.target_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.DEVICE)

        self.current_epsilon = config.EPSILON

        self.update_step_counter = 0
        self.epsilon_step_counter = 0

        self.replay_memory = ReplayMemory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config.ALPHA)
        self.loss_func = nn.MSELoss()

    def update_epsilon(self):
        self.epsilon_step_counter += 1
        if self.epsilon_step_counter % config.DECREMENT_ITERATION == 0:
            self.current_epsilon = max(0, self.current_epsilon - config.DELTA)

    def select(self, state):
        # random policy
        if random.random() <= self.current_epsilon:
            action = np.random.randint(0, self.action_number)
        # greedy policy
        else:
            action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
            action = torch.argmax(action_val, 1).cpu().data.numpy()[0]

        return action

    def predict(self, state):
        action_val = self.eval_net.forward(torch.LongTensor(state).unsqueeze_(0).to(config.DEVICE))
        return torch.argmax(action_val, 1).cpu().data.numpy()[0]

    def update(self):
        # updating the parameters
        self.update_step_counter += 1
        if self.update_step_counter % config.UPDATE_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.replay_memory.size < config.BATCH_SIZE:
            return

        # sampling batch from memory
        state, next_state, action, reward, done = self.replay_memory.sample(config.BATCH_SIZE)

        batch_state = torch.LongTensor(state).to(config.DEVICE)
        batch_next_state = torch.LongTensor(next_state).to(config.DEVICE)
        batch_action = torch.LongTensor(action).to(config.DEVICE)
        batch_reward = torch.FloatTensor(reward).to(config.DEVICE)
        batch_done = torch.FloatTensor(done).to(config.DEVICE)

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze_(-1)).squeeze_(1).to(config.DEVICE)

        # q_target
        q_next = self.target_net(batch_next_state).max(1)[0].to(config.DEVICE).detach_()
        q_target = batch_reward + batch_done * config.GAMMA * q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename, path=config.DPAMSA_WEIGHTS_PATH):
        torch.save(self.eval_net.state_dict(), os.path.join(path, "{}.pth".format(filename)))
        print("{} has been saved...".format(filename))

    def load(self, filename, path=config.DPAMSA_WEIGHTS_PATH):
        self.eval_net.load_state_dict(torch.load(os.path.join(path, "{}.pth".format(filename)),
                                                 map_location=torch.device(config.DEVICE)))
        # print("{} has been loaded...".format(filename))
