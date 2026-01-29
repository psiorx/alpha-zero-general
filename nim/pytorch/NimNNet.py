import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class NimNNet(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.max_stones = float(game.getMaxStones())

        super(NimNNet, self).__init__()
        input_size = self.board_x * self.board_y

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_size)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, s):
        s = s.view(-1, self.board_x * self.board_y)
        s = s / self.max_stones

        s = F.relu(self.fc1(s))
        s = F.dropout(s, p=self.args.dropout, training=self.training)
        s = F.relu(self.fc2(s))
        s = F.dropout(s, p=self.args.dropout, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
