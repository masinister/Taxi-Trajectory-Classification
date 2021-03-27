import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Trajectory_RNN(nn.Module):

    def __init__(self, input_size, num_layers = 2):
        super(Trajectory_RNN, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = 16, num_layers = self.num_layers, dropout = 0.2, batch_first = True)

        self.fc = nn.Sequential(nn.Linear(16, 16),
                                nn.ReLU(),
                                nn.Linear(16, 5),
                                nn.Dropout(0.1),)

    def forward(self, x):
        x, h = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return x

    def predict(self, x):
        x = self.forward(x)
        _, predicted = torch.max(x.data, 1)
        return predicted

    def save(self, path):
        torch.save(self.state_dict(), path)


class Trajectory_NN(nn.Module):

    def __init__(self, input_size, traj_len):
        super(Trajectory_NN, self).__init__()
        self.traj_len = traj_len
        self.input_size = input_size

        self.fc = nn.Sequential(nn.Linear(self.input_size * self.traj_len, 16),
                                nn.ReLU(),
                                nn.Linear(16, 16),
                                nn.ReLU(),
                                nn.Linear(16, 5),
                                nn.Dropout(0.2),)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        _, predicted = torch.max(x.data, 1)
        return predicted

    def save(self, path):
        torch.save(self.state_dict(), path)
